"""在线样本回流评测集（线上评估闭环）：筛选真实流量样本 → 回归池 / 修复池。

读取 eval/samples/online_*.jsonl（每行一轮真实对话的轻量样本），按能力域分流：

回归池（金标可信，直接进离线回归）：
- capability=rag（有检索命中）→ eval/online_eval_set.jsonl（RAG 评测用例）；
- capability=agent（无命中但有工具调用）→ eval/online_agent_eval_set.jsonl（Agent 行为回归用例）；
- 组成：关键分支（RAG 域 multihop/decompose/out_of_kb，全量）+ 其余按 --ratio 兜底抽样；
  金标直接沿用线上行为——但仅限「用户未点踩」的样本（线上行为未被否定的信号相对可信）。

修复池（金标不可信，待人工修正后转回归池）：
- 点踩（vote=down）样本 → eval/online_fix_set.jsonl，label=needs_fix；
- expected / must_call 金标**置空不沿用线上行为**（点踩样本的线上行为可能是错的，
  原样固化会把坏行为写进评测集）；线上观测行为记入 observed 字段供人工参照修正；
- 人工修正金标（expected/must_call/reference）后移入对应回归池文件再参与门禁。

筛选规则（回归池，两域统一）：
1. 关键分支（仅 RAG 域）→ 全量回流（在线样本量小，宁全不漏）；
2. 兜底：其余按 --ratio 比例随机抽样（默认 0.3），--limit 封顶防止评测集无限膨胀
   （--ratio 1 即全量回流）；
3. 同 query 去重：优先保留点踩样本（去重后点踩仍移交修复池）。

金标处理：
- RAG 域：expected（retrieval_need/retrieval_mode/complexity/generation_mode）沿用线上
  真实语义路由决策（未点踩样本）；relevant 以真实检索 top-k 命中为基础；
- Agent 域：mode（实际架构模式）+ must_call（线上真实工具轨迹）+ branch=normal，作行为回归；
- 两域 answer_keywords / reference 均留空待人工复核补全（产出 origin 溯源到样本行）。

用法（在 backend/ 目录下）：
    python scripts/eval_online.py                  # 全量样本按默认比例回流两域回归池 + 修复池
    python scripts/eval_online.py --date 20260906  # 只回流指定日期的样本
    python scripts/eval_online.py --ratio 0.2      # 兜底随机抽样比例（默认 0.3）
    python scripts/eval_online.py --limit 50       # 每域回归池用例上限（超出优先保留关键分支）
    python scripts/eval_online.py --dry-run        # 只打印筛选统计，不写评测集
    python scripts/eval_online.py --out <p> --out-agent <p> --out-fix <p>  # 自定义输出路径
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# 允许在 backend 任意相对路径下执行：把 backend/ 加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

# 关键分支（仅 RAG 域）：在线样本量小，全量回流（不随机抽样）
_CRITICAL_BRANCHES = {"multihop", "decompose", "out_of_kb"}


def _sample_dir() -> Path:
    return Path(settings.eval_sample_dir)


def _load_rows(date: str | None) -> list[dict[str, Any]]:
    """读取全部（或指定日期）online_*.jsonl 样本行。"""
    rows: list[dict[str, Any]] = []
    for path in sorted(_sample_dir().glob("online_*.jsonl")):
        if date and date not in path.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 损坏行跳过，不影响回流
    return rows


def _domain_of(row: dict[str, Any]) -> str:
    """样本能力域：capability 显式标记优先；老样本（无标记）按 retrieved_ids 回退。"""
    cap = row.get("capability")
    if cap:
        return cap
    return "rag" if row.get("retrieved_ids") else "agent"


def _branch_of(row: dict[str, Any]) -> str:
    """从线上真实路由决策还原分支（仅 RAG 域）：insufficient（检索不足/库外）优先判 out_of_kb。"""
    if row.get("insufficient") or not row.get("retrieved_ids"):
        return "out_of_kb"
    complexity = row.get("complexity") or "simple"
    if complexity in _CRITICAL_BRANCHES:
        return complexity
    return "simple"


def _expected_of(row: dict[str, Any]) -> dict[str, Any]:
    """还原语义路由决策为评测集 expected 结构（检索命中才落 RAG 域 → retrieval_need 恒 true）。"""
    return {
        "retrieval_need": True,
        "retrieval_mode": row.get("retrieval_mode") or "hybrid",
        "complexity": row.get("complexity") or "simple",
        "generation_mode": row.get("generation_mode") or "citation",
    }


def _origin_of(row: dict[str, Any]) -> dict[str, Any]:
    """溯源块：样本 id / 用户反馈 / 时间 / 版本。"""
    return {
        "sample_id": row.get("sample_id"),
        "vote": row.get("vote"),
        "reason": row.get("reason"),
        "ts": row.get("ts"),
        "version": row.get("version"),
    }


# ---------------------------------------------------------------------------
# 回归池用例（金标可信）
# ---------------------------------------------------------------------------
def _build_case(seq: int, row: dict[str, Any]) -> dict[str, Any]:
    """RAG 样本 → RAG 评测用例（answer_keywords/reference 留空待人工复核，origin 溯源）。"""
    return {
        "id": f"on_{seq:04d}",
        "branch": _branch_of(row),
        "query": row.get("query") or row.get("effective_query") or "",
        "expected": _expected_of(row),
        "relevant": row.get("retrieved_ids") or [],
        "answer_keywords": [],
        "reference": "",
        "origin": _origin_of(row),
    }


def _build_agent_case(seq: int, row: dict[str, Any]) -> dict[str, Any]:
    """Agent 样本 → 行为回归用例：mode + must_call（线上真实工具轨迹），金标留空待人工补。"""
    return {
        "id": f"agent_on_{seq:04d}",
        "capability": "agent",
        "mode": row.get("mode") or "react",
        "branch": "normal",  # 线上无显式分支标签，统一作行为回归（验证真实轨迹可复现）
        "query": row.get("query") or row.get("effective_query") or "",
        "must_call": row.get("tool_sequence") or [],
        "forbidden_tools": [],
        "answer_keywords": [],
        "reference": "",
        "origin": _origin_of(row),
    }


# ---------------------------------------------------------------------------
# 修复池用例（金标不可信，待人工修正）
# ---------------------------------------------------------------------------
def _build_fix_case(seq: int, row: dict[str, Any]) -> dict[str, Any]:
    """点踩/异常样本 → 修复池用例：金标置空不沿用线上行为，observed 记录观测行为供参照修正。"""
    return {
        "id": f"fix_{seq:04d}",
        "capability": _domain_of(row),
        "label": "needs_fix",  # 待人工修正后移入回归池（修正 expected/must_call/reference 后置 label=ready）
        "query": row.get("query") or row.get("effective_query") or "",
        "reason": row.get("reason"),  # 用户点踩原因，人工复核第一手线索
        # 金标一律置空待人工修正：点踩样本的线上行为可能是错的（路由误判/工具误选），
        # 原样固化会把坏行为写进评测集；线上观测行为保留在 observed 供人工参照。
        "expected": None,  # rag 域金标待修正
        "must_call": None,  # agent 域金标待修正
        "observed": {  # 线上观测行为（仅供参考，不可直接当金标）
            "mode": row.get("mode"),
            "retrieved_ids": row.get("retrieved_ids") or [],
            "tool_sequence": row.get("tool_sequence") or [],
            "answer": row.get("answer"),
        },
        "answer_keywords": [],
        "reference": "",
        "origin": _origin_of(row),
    }


# ---------------------------------------------------------------------------
# 筛选
# ---------------------------------------------------------------------------
def _select(
    rows: list[dict[str, Any]],
    rng: random.Random,
    ratio: float,
    limit: int,
    rag_domain: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """单域回归池筛选：去重（优先点踩）→ 分层（关键分支 + 兜底抽样）→ limit 封顶。

    返回 (regression, down_rows, stats)：
    - regression：可信金标样本（关键分支 + 兜底抽样），**不含点踩**（点踩移交修复池）；
    - down_rows：点踩样本，全部移交修复池（金标待人工修正，不沿用线上行为）。
    """
    # 按 query 去重：优先保留点踩样本（同一问题被多人问时，负面信号优先级最高）
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        q = (row.get("query") or row.get("effective_query") or "").strip()
        if not q:
            continue
        prev = dedup.get(q)
        if prev is None or (row.get("vote") == "down" and prev.get("vote") != "down"):
            dedup[q] = row

    # 分层筛选：关键分支（仅 RAG 域）→ 兜底随机抽样；点踩独立移交修复池
    critical: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    down_rows: list[dict[str, Any]] = []
    for row in dedup.values():
        if row.get("vote") == "down":
            down_rows.append(row)
        elif rag_domain and _branch_of(row) in _CRITICAL_BRANCHES:
            critical.append(row)
        else:
            fallback.append(row)

    sampled_fallback = [r for r in fallback if rng.random() < ratio]
    regression = critical + sampled_fallback
    # 总量上限：优先保留关键分支（按样本 ts 先后，保留更早的）
    if limit > 0 and len(regression) > limit:
        priority = len(critical)
        if priority >= limit:
            regression = regression[:limit]
        else:
            regression = critical + sampled_fallback[: limit - priority]

    stats = {
        "rows": len(rows),
        "dedup": len(dedup),
        "down": len(down_rows),
        "critical": len(critical),
        "sampled": len(sampled_fallback),
        "fallback": len(fallback),
        "regression": len(regression),
    }
    return regression, down_rows, stats


def _write_cases(out_path: str, cases: list[dict[str, Any]]) -> None:
    """写 JSONL 评测集（空则不创建文件）。"""
    if not cases:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在线样本回流评测集（筛选真实流量 → 回归池 / 修复池，RAG/Agent 两域）"
    )
    parser.add_argument("--date", default=None, help="只回流指定日期样本（文件名 online_YYYYMMDD.jsonl）")
    parser.add_argument("--ratio", type=float, default=0.3, help="兜底随机抽样比例（默认 0.3）")
    parser.add_argument("--limit", type=int, default=0, help="每域回归池用例总量上限（0=不限，超出优先保留关键分支）")
    parser.add_argument("--dry-run", action="store_true", help="只打印筛选统计，不写评测集")
    parser.add_argument("--out", default=None, help="RAG 回归池输出路径（默认 settings.eval_online_set_path）")
    parser.add_argument("--out-agent", default=None, help="Agent 回归池输出路径（默认 settings.eval_online_agent_set_path）")
    parser.add_argument("--out-fix", default=None, help="修复池输出路径（默认 settings.eval_online_fix_set_path）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    args = parser.parse_args()

    rows = _load_rows(args.date)
    if not rows:
        print("未找到任何在线样本（eval/samples/online_*.jsonl 为空）。")
        print("提示：真实对话一次（RAG 命中或有工具调用）后会自动落样本，再回流即可。")
        return

    # 按能力域分流
    rag_rows = [r for r in rows if _domain_of(r) == "rag"]
    agent_rows = [r for r in rows if _domain_of(r) == "agent"]

    rng = random.Random(args.seed)
    rag_selected, rag_down, rag_stats = _select(rag_rows, rng, args.ratio, args.limit, rag_domain=True)
    agent_selected, agent_down, agent_stats = _select(agent_rows, rng, args.ratio, args.limit, rag_domain=False)

    rag_cases = [_build_case(i + 1, row) for i, row in enumerate(rag_selected)]
    agent_cases = [_build_agent_case(i + 1, row) for i, row in enumerate(agent_selected)]

    # 修复池：两域点踩样本合并（金标不可信，不沿用线上行为），按 --limit 封顶（保留更早）
    fix_rows = rag_down + agent_down
    if args.limit > 0 and len(fix_rows) > args.limit:
        fix_rows = fix_rows[: args.limit]
    fix_cases = [_build_fix_case(i + 1, row) for i, row in enumerate(fix_rows)]

    # 控制台统计
    print("=== 在线样本回流评测集（回归池 / 修复池）===")
    print(f"读取样本: {len(rows)} 行（RAG {len(rag_rows)} / Agent {len(agent_rows)}）")
    print(
        f"[RAG 回归池] {rag_stats['rows']} 行去重后 {rag_stats['dedup']} 个 → 关键分支 {rag_stats['critical']} "
        f"+ 兜底抽样 {rag_stats['sampled']}/{rag_stats['fallback']} (ratio={args.ratio}) "
        f"→ 回流 {rag_stats['regression']} 条（点踩 {rag_stats['down']} 条移交修复池）"
    )
    if rag_cases:
        by_branch: dict[str, int] = {}
        for c in rag_cases:
            by_branch[c["branch"]] = by_branch.get(c["branch"], 0) + 1
        print("RAG 分支分布: " + ", ".join(f"{k}={v}" for k, v in sorted(by_branch.items())))
    print(
        f"[Agent 回归池] {agent_stats['rows']} 行去重后 {agent_stats['dedup']} 个 → "
        f"兜底抽样 {agent_stats['sampled']}/{agent_stats['fallback']} (ratio={args.ratio}) "
        f"→ 回流 {agent_stats['regression']} 条（点踩 {agent_stats['down']} 条移交修复池）"
    )
    print(f"[修复池] 点踩样本共 {len(fix_rows)} 条 → 写入 {len(fix_cases)} 条（金标待人工修正）")
    if args.limit > 0:
        print(f"（--limit {args.limit} 回归池每域封顶，超出优先保留关键分支；修复池同样封顶）")

    if args.dry_run:
        print("\n[dry-run] 未写入评测集（--dry-run）")
        return
    if not rag_cases and not agent_cases and not fix_cases:
        print("\n无满足回流条件的样本，未写入评测集。")
        return

    rag_out = args.out or settings.eval_online_set_path
    agent_out = args.out_agent or settings.eval_online_agent_set_path
    fix_out = args.out_fix or settings.eval_online_fix_set_path
    _write_cases(rag_out, rag_cases)
    _write_cases(agent_out, agent_cases)
    _write_cases(fix_out, fix_cases)
    if rag_cases:
        print(f"\n已写入 {len(rag_cases)} 条 RAG 回归用例: {rag_out}")
    if agent_cases:
        print(f"已写入 {len(agent_cases)} 条 Agent 回归用例: {agent_out}")
    if fix_cases:
        print(f"已写入 {len(fix_cases)} 条修复用例: {fix_out}")
    print("提示: 回归池 answer_keywords/reference 需人工补全；修复池需修正 expected/must_call/reference 后移入回归池。")


if __name__ == "__main__":
    main()
