"""真实 modular RAG 全面评测入口：真实 Qdrant/ES 检索 + 真实 LLM 生成 + RAGAS 评分。

用法（backend 目录）：
  python scripts/eval_real_full.py             # 真实评测（需 LLM_API_KEY + EMBEDDING_API_KEY + Qdrant/ES 已建库）
  python scripts/eval_real_full.py --fake      # 冒烟：真实检索链路 + 占位生成（验证链路，分数无意义）
  python scripts/eval_real_full.py --top-k 5   # 调整检索配额
  python scripts/eval_real_full.py --report path.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import real_full  # noqa: E402

DEFAULT_REPORT = "eval/reports/real_full.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="真实 modular RAG 全面评测")
    parser.add_argument("--fake", action="store_true", help="冒烟模式：真实检索 + 占位生成/评分")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    records, report = real_full.run(top_k=args.top_k, fake=args.fake)
    real_full.save_report(report, args.report)

    r = report["retrieval"]["overall"]
    print(f"\n=== 真实 modular 全面评测（{report['meta']['mode']}）===")
    print(f"用例: {report['meta']['eval_cases']}  top_k={args.top_k}")
    print(
        f"检索: recall={r['avg_recall']} precision={r['avg_precision']} mrr={r['avg_mrr']} "
        f"gate_answerable={r['gate_answerable_rate']} 平均检索耗时={r['avg_retrieve_ms']}ms "
        f"builtin过滤={report['meta']['builtin_filtered']}"
    )
    for d, v in report["retrieval"]["by_difficulty"].items():
        print(f"  {d:<10} cases={v['cases']:<3} recall={v['avg_recall']} mrr={v['avg_mrr']}")
    print(f"行为: {report['behavior']}")
    g = report["generation"]
    print(
        f"生成(RAGAS): faithfulness={g['avg_faithfulness']} relevancy={g['avg_answer_relevancy']} "
        f"correctness={g['avg_answer_correctness']} context_precision={g['avg_context_precision']} "
        f"(scored={g['scored_cases']})"
    )
    print(f"拒答: grounded_rate={report['out_of_kb']['grounded_rate']} ({report['out_of_kb']['cases']} 例)")
    print(f"报告: {args.report}")
    failed = [x for x in records if x["error"]]
    if failed:
        print(f"异常用例 {len(failed)} 个:")
        for x in failed:
            print(f"  {x['id']}: {x['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
