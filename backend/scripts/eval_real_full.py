"""真实 RAG 方案全面评测入口（modular / agentic）：真实 Qdrant/ES 检索 + 真实 LLM 生成 + RAGAS 评分。

用法（backend 目录）：
  python scripts/eval_real_full.py                        # 真实评测 modular（默认方案）
  python scripts/eval_real_full.py --scheme agentic       # 真实评测 agentic（含 agent 轨迹指标）
  python scripts/eval_real_full.py --fake                 # 冒烟：真实检索链路 + 占位生成（验证链路，分数无意义）
  python scripts/eval_real_full.py --top-k 5              # 调整检索配额
  python scripts/eval_real_full.py --report path.json     # 自定义报告路径
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import real_full  # noqa: E402

DEFAULT_REPORT = "eval/reports/real_full.json"


def _setup_logging() -> None:
    """实时观测日志：app 层（语义路由/执行计划/多跳/重排/LLM 调用耗时）INFO 实时刷出；
    三方库（ragas/httpx/qdrant 等）保持 WARNING 防刷屏。"""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("app").setLevel(logging.INFO)


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(description="真实 RAG 方案全面评测（modular/agentic）")
    parser.add_argument("--scheme", default="modular", choices=["modular", "agentic"], help="被评测方案")
    parser.add_argument("--fake", action="store_true", help="冒烟模式：真实检索 + 占位生成/评分")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--report", default=None, help="报告路径（缺省按方案：real_full[_scheme].json）")
    args = parser.parse_args()

    report_path = args.report or (
        DEFAULT_REPORT if args.scheme == "modular" else f"eval/reports/real_full_{args.scheme}.json"
    )
    records, report = real_full.run(top_k=args.top_k, fake=args.fake, scheme_id=args.scheme)
    real_full.save_report(report, report_path)

    r = report["retrieval"]["overall"]
    print(f"\n=== 真实 {args.scheme} 全面评测（{report['meta']['mode']}）===")
    print(f"用例: {report['meta']['eval_cases']}  top_k={args.top_k}")
    print(
        f"检索: recall={r['avg_recall']} precision={r['avg_precision']} mrr={r['avg_mrr']} "
        f"gate_answerable={r['gate_answerable_rate']} 平均检索耗时={r['avg_retrieve_ms']}ms "
        f"builtin过滤={report['meta']['builtin_filtered']}"
    )
    for d, v in report["retrieval"]["by_difficulty"].items():
        print(f"  {d:<10} cases={v['cases']:<3} recall={v['avg_recall']} mrr={v['avg_mrr']}")
    print(f"行为: {report['behavior']}")
    if report.get("agent"):
        a = report["agent"]
        print(
            f"Agent: 平均事件={a['avg_events']} 工具执行={a['avg_tool_exec']} "
            f"纠错率={a['correction_rate']}(有效={a['correction_success_rate']}) "
            f"avg_token={a['avg_tokens']} 角色={a['role_llm_calls']}"
        )
        print(f"  工具分布: {a['tool_calls']}")
    g = report["generation"]
    print(
        f"生成(RAGAS): faithfulness={g['avg_faithfulness']} relevancy={g['avg_answer_relevancy']} "
        f"correctness={g['avg_answer_correctness']} context_precision={g['avg_context_precision']} "
        f"(scored={g['scored_cases']})"
    )
    print(f"拒答: grounded_rate={report['out_of_kb']['grounded_rate']} ({report['out_of_kb']['cases']} 例)")
    print(f"报告: {report_path}")
    failed = [x for x in records if x["error"]]
    if failed:
        print(f"异常用例 {len(failed)} 个:")
        for x in failed:
            print(f"  {x['id']}: {x['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
