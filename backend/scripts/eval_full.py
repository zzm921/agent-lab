"""modular RAG 全面评测脚本：一份报告覆盖路由/检索/生成/拒答四类维度。

用法（在 backend/ 目录下）：
    python scripts/eval_full.py          # 真实评测：LLM 生成 + RAGAS 标准评分（需 Key）
    python scripts/eval_full.py --fake   # 离线冒烟：验证链路可跑通（分数无评测意义）
    python scripts/eval_full.py --report PATH  # 指定报告输出路径（默认 eval/reports/full.json）

输出：
    - 控制台：路由准确率 + 检索分支表 + 生成质量汇总 + 拒答率 + 失败用例明细
    - JSON 报告：全量逐用例明细（检索指标 + 答案 + 评分），供失败样本回流分析。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许在 backend 任意相对路径下执行：把 backend/ 加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import full  # noqa: E402

# 分支展示顺序与中文名
_BRANCH_ORDER = ["no_retrieval", "simple", "rewrite", "decompose", "multihop", "out_of_kb"]
_BRANCH_NAME = {
    "no_retrieval": "不检索",
    "simple": "单点事实",
    "rewrite": "改写/指代",
    "decompose": "分解/对比",
    "multihop": "多跳/流程",
    "out_of_kb": "库外问题",
}
_GEN_LABEL = {
    "faithfulness": "忠实度",
    "answer_relevancy": "相关性",
    "answer_correctness": "正确性",
    "context_precision": "上下文精确",
}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _print_retrieval_table(branches: dict, overall: dict) -> None:
    header = (
        f"{'分支':<10} {'用例':>4}   "
        f"{'召回':>5}   {'精确':>5}   {'MRR':>5}   {'关键词':>5}   {'可答':>5}   {'耗时':>8}"
    )
    print(header)
    print("-" * len(header))
    for branch in _BRANCH_ORDER:
        agg = branches.get(branch)
        if not agg:
            continue
        print(
            f"{_BRANCH_NAME[branch]:<10} {agg['cases']:>4}   "
            f"召回 {_fmt(agg['avg_recall']):>5}   精确 {_fmt(agg['avg_precision']):>5}   "
            f"MRR {_fmt(agg['avg_mrr']):>5}   关键词 {_fmt(agg['keyword_coverage']):>5}   "
            f"可答 {_fmt(agg['answerable_rate']):>5}   耗时 {_fmt(agg['avg_elapsed_ms']):>6}ms"
        )
    print("-" * len(header))
    print(f"{'总计':<10} {overall.get('cases', ''):>4}   "
          f"召回 {_fmt(overall.get('avg_recall')):>5}   精确 {_fmt(overall.get('avg_precision')):>5}   "
          f"MRR {_fmt(overall.get('avg_mrr')):>5}   关键词 {_fmt(overall.get('keyword_coverage')):>5}   "
          f"可答 {_fmt(overall.get('answerable_rate')):>5}   耗时 {_fmt(overall.get('avg_elapsed_ms')):>6}ms")


def _print_failures(records: list[dict]) -> None:
    print("\n=== 失败用例明细（检索召回<1 / 拒答未通过 / 生成或评分出错，需人工核查）===")
    shown = 0
    for r in records:
        err = r.get("error")
        if err:
            shown += 1
            print(f"[{r['id']}] ({_BRANCH_NAME.get(r['branch'], r['branch'])}) {r['query']}")
            print(f"    错误: {err}")
            continue
        if r["out_of_kb"] and r.get("grounded") is False:
            shown += 1
            print(f"[{r['id']}] (库外问题) {r['query']} 拒答未通过: {r.get('judge_reason')}")
            print(f"    回答: {r.get('answer')}")
            continue
        ret = r.get("retrieval") or {}
        if not r["out_of_kb"] and r["branch"] != "no_retrieval":
            failed = (
                (ret.get("recall") is not None and ret["recall"] < 1.0)
                or (ret.get("answerable") is False)
                or (ret.get("keyword_hit") is False)
            )
            if not failed:
                continue
            shown += 1
            print(f"[{r['id']}] ({_BRANCH_NAME.get(r['branch'], r['branch'])}) {r['query']}")
            print(f"    召回={ret.get('recall')} MRR={ret.get('mrr')} "
                  f"可答={ret.get('answerable')} 关键词命中={ret.get('keyword_hit')}")
        if shown >= 20:
            print("…（仅显示前 20 条）")
            break
    if not shown:
        print("（无）")


def main() -> None:
    parser = argparse.ArgumentParser(description="modular RAG 全面评测（路由/检索/生成/拒答）")
    parser.add_argument("--fake", action="store_true", help="离线冒烟模式（Fake 模型，分数无评测意义）")
    parser.add_argument("--top-k", type=int, default=3, help="最终保留命中数（默认 3）")
    parser.add_argument("--report", default="eval/reports/full.json", help="报告输出路径")
    args = parser.parse_args()

    records, report = full.run(top_k=args.top_k, fake=args.fake)
    full.save_report(report, args.report)

    print("=== modular RAG 全面评测报告 ===")
    print(f"模式: {report['meta']['mode']} | top_k={args.top_k} | "
          f"语料 {report['meta']['corpus_size']} 条 | 用例 {report['meta']['eval_cases']} 条")
    print()

    print(f"--- 路由 ---\n路由准确率: {_fmt(report['routing']['routing_accuracy'])}\n")
    print("--- 检索（按分支） ---")
    _print_retrieval_table(report["retrieval"]["branches"], report["retrieval"]["overall"])
    print()

    print("--- 生成质量（RAGAS 标准指标） ---")
    gen = report["generation"]
    for key, label in _GEN_LABEL.items():
        print(f"{label} {key}: {_fmt(gen[f'avg_{key}'])}")
    print(f"计分用例: {gen['scored_cases']}")
    print()

    print("--- 拒答行为（库外问题） ---")
    print(f"拒答通过率 grounded_rate: {_fmt(report['out_of_kb']['grounded_rate'])} "
          f"({report['out_of_kb']['cases']} 例)")
    _print_failures(records)
    print(f"\n报告已写入: {args.report}")


if __name__ == "__main__":
    main()
