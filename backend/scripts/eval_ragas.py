"""modular RAG L3 语义评测脚本（RAGAS 标准指标评分）。

在 L1/L2 之上用 RAGAS 库对生成答案打标准分（Faithfulness / AnswerRelevancy），
可对比 L2 手写 judge 的评分差异。

用法（在 backend/ 目录下）：
    python scripts/eval_ragas.py --fake        # 离线冒烟：验证检索→生成→RAGAS→报告链路（分数无意义）
    python scripts/eval_ragas.py               # 真实评测：需配置 LLM_API_KEY 与 EMBEDDING_API_KEY
    python scripts/eval_ragas.py --report PATH # 指定报告输出路径（默认 eval/reports/latest_ragas.json）

输出：
    - 控制台：逐用例 RAGAS 评分 + 汇总
    - JSON 报告：全量逐用例答案与评分，供失败样本回流分析。
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 允许在 backend 任意相对路径下执行：把 backend/ 加入 sys.path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import ragas_eval  # noqa: E402


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="modular RAG L3 语义评测（RAGAS）")
    parser.add_argument("--fake", action="store_true", help="离线冒烟模式（Fake 模型，分数无评测意义）")
    parser.add_argument("--top-k", type=int, default=3, help="最终保留命中数（默认 3）")
    parser.add_argument("--report", default="eval/reports/latest_ragas.json", help="报告输出路径")
    args = parser.parse_args()

    records, report = ragas_eval.run(top_k=args.top_k, fake=args.fake)
    ragas_eval.save_report(report, args.report)

    print("=== modular RAG L3 语义评测（RAGAS）===")
    print(f"模式: {report['meta']['mode']} | top_k={args.top_k} | "
          f"语料 {report['meta']['corpus_size']} 条 | 用例 {report['meta']['eval_cases']} 条")
    print(f"平均忠实度 Faithfulness: {_fmt(report['avg_faithfulness'])}")
    print(f"平均相关性 AnswerRelevancy: {_fmt(report['avg_answer_relevancy'])}")
    print(f"计分用例: {report['scored_cases']}")
    print()
    print("=== 逐用例 ===")
    print(f"{'ID':<6} {'分支':<8}  {'忠实':>6}  {'相关':>6}   {'命中':>4}  {'可答':>4}  {'耗时':>7}  查询")
    for r in records:
        if r.get("skipped"):
            continue
        err = r.get("error")
        score_row = f"{_fmt(r['faithfulness']):>6}  {_fmt(r['answer_relevancy']):>6}" if not err else f"{'ERR':>6}  {'':>6}"
        print(f"{r['id']:<6} {r['branch']:<8}  {score_row}   "
              f"{r['retrieved_count']:>4}  {_fmt(r['answerable']):>4}  {r['elapsed_ms']:>7.1f}  {r['query']}")
        if err:
            print(f"      ⚠ {err}")
    print(f"\n报告已写入: {args.report}")
    if args.fake:
        print("提示: 去掉 --fake 并配置 LLM_API_KEY / EMBEDDING_API_KEY 可运行真实评测。")


if __name__ == "__main__":
    main()
