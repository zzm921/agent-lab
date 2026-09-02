"""modular RAG 离线评测脚本：跑全量评测集 → 检索/路由/闸门指标 → 分支报告。

用法（在 backend/ 目录下）：
    python scripts/eval_modular.py                 # 注入期望路由的确定性回归评测（离线，无需 Key）
    python scripts/eval_modular.py --real-router   # 使用真实 LLM 路由，评测路由准确率（需配置 Key）
    python scripts/eval_modular.py --report PATH   # 指定报告输出路径（默认 eval/reports/latest.json）

输出：
    - 控制台：按分支汇总表 + 失败用例明细（召回<1 / 未可答 / 关键词未覆盖）
    - JSON 报告：全量逐用例明细 + 汇总，供后续失败样本回流分析。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许在 backend 任意相对路径下执行：把 backend/ 加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import runner  # noqa: E402

# 分支展示顺序
_BRANCH_ORDER = ["no_retrieval", "simple", "rewrite", "decompose", "multihop", "out_of_kb"]
# 分支中文名
_BRANCH_NAME = {
    "no_retrieval": "不检索",
    "simple": "单点事实",
    "rewrite": "改写/指代",
    "decompose": "分解/对比",
    "multihop": "多跳/流程",
    "out_of_kb": "库外问题",
}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _print_table(title: str, agg: dict) -> None:
    row = (
        f"{title:<10} {agg['cases']:>4}   "
        f"召回 {_fmt(agg['avg_recall']):>5}   精确 {_fmt(agg['avg_precision']):>5}   "
        f"MRR {_fmt(agg['avg_mrr']):>5}   关键词 {_fmt(agg['keyword_coverage']):>5}   "
        f"可答 {_fmt(agg['answerable_rate']):>5}   澄清 {_fmt(agg['clarify_rate']):>5}   "
        f"耗时 {_fmt(agg['avg_elapsed_ms']):>6}ms"
    )
    print(row)


def _print_failures(records: list[dict]) -> None:
    print("\n=== 失败用例明细（召回<1 或 未可答 或 关键词未覆盖，需人工核查）===")
    shown = 0
    for r in records:
        # 不检索 / 库外问题：未可答是期望行为，不算失败
        if not r["need_retrieval"] or r["branch"] == "out_of_kb":
            continue
        failed = (
            (r["recall"] is not None and r["recall"] < 1.0)
            or (r["answerable"] is False)
            or (r["keyword_hit"] is False)
        )
        if not failed:
            continue
        shown += 1
        print(f"[{r['id']}] ({_BRANCH_NAME.get(r['branch'], r['branch'])}) {r['query']}")
        print(f"    相关={r['relevant']} 命中id={r['retrieved_ids']} "
              f"召回={r['recall']} MRR={r['mrr']} 关键词命中={r['keyword_hit']}")
        print(f"    可答={r['answerable']} 建议={r['recommendation']} "
              f"缺失={r['missing_facts']} 耗时={r['elapsed_ms']}ms")
        if shown >= 20:
            print("…（仅显示前 20 条）")
            break
    if not shown:
        print("（无）")


def main() -> None:
    parser = argparse.ArgumentParser(description="modular RAG 离线评测")
    parser.add_argument("--real-router", action="store_true", help="使用真实 LLM 路由评测路由准确率（需 Key）")
    parser.add_argument("--top-k", type=int, default=3, help="最终保留命中数（默认 3）")
    parser.add_argument("--report", default="eval/reports/latest.json", help="报告输出路径")
    args = parser.parse_args()

    records, report = runner.run(top_k=args.top_k, real_router=args.real_router)
    runner.save_report(report, args.report)

    print("=== modular RAG 离线评测报告 ===")
    print(f"路由模式: {report['meta']['modules']} | top_k={args.top_k} | "
          f"语料 {report['meta']['corpus_size']} 条 | 用例 {report['meta']['eval_cases']} 条")
    print(f"路由准确率: {report['routing_accuracy']:.3f} ({_fmt(report['routing_accuracy'] * report['meta']['eval_cases'])}/{report['meta']['eval_cases']})")
    print()
    header = (
        f"{'分支':<10} {'用例':>4}   "
        f"{'召回':>5}   {'精确':>5}   {'MRR':>5}   {'关键词':>5}   {'可答':>5}   {'澄清':>5}   {'耗时':>8}"
    )
    print(header)
    print("-" * len(header))
    for branch in _BRANCH_ORDER:
        agg = report["branches"].get(branch)
        if agg:
            _print_table(_BRANCH_NAME[branch], agg)
    print("-" * len(header))
    _print_table("总计", report["overall"])

    _print_failures(records)
    print(f"\n报告已写入: {args.report}")
    if not args.real_router:
        print("提示: 使用 --real-router 可额外评测真实 LLM 路由准确率（需配置 Key）。")


if __name__ == "__main__":
    main()
