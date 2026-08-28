"""L2 语义评测脚本：真实 LLM 完整链路（检索→生成→judge 评分）的忠实度/相关性报告。

用法（在 backend/ 目录下）：
    python scripts/eval_semantic.py                  # 真实 LLM 评测（需配置 LLM_API_KEY）
    python scripts/eval_semantic.py --fake           # 离线冒烟：验证链路可跑通（分数无评测意义）
    python scripts/eval_semantic.py --report PATH    # 指定报告输出路径（默认 eval/reports/semantic_latest.json）

输出：
    - 控制台：逐用例的生成答案 + faithfulness/answer_relevance/grounded，含库外负向用例判定；
    - JSON 报告：全量逐用例明细 + 汇总（供失败样本回流分析）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许在 backend 任意相对路径下执行：把 backend/ 加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 终端默认 GBK：强制 UTF-8 输出，避免 ✓/中文 打印报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval import semantic  # noqa: E402

# 分支中文名
_BRANCH_NAME = {
    "no_retrieval": "不检索",
    "simple": "单点事实",
    "rewrite": "改写/指代",
    "decompose": "分解/对比",
    "multihop": "多跳/流程",
    "out_of_kb": "库外问题",
}


def _fmt_score(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="modular RAG L2 语义评测（真实 LLM 生成 + judge 评分）")
    parser.add_argument("--fake", action="store_true", help="Fake 模型冒烟：离线验证链路，分数无评测意义")
    parser.add_argument("--top-k", type=int, default=3, help="最终保留命中数（默认 3）")
    parser.add_argument("--report", default="eval/reports/semantic_latest.json", help="报告输出路径")
    args = parser.parse_args()

    records, report = semantic.run(top_k=args.top_k, fake=args.fake)
    semantic.save_report(report, args.report)

    print("=== modular RAG L2 语义评测报告 ===")
    print(f"模式: {report['meta']['mode']} | top_k={args.top_k} | 语料 {report['meta']['corpus_size']} 条 | "
          f"用例 {report['meta']['eval_cases']} 条")
    print(f"平均忠实度 faithfulness: {_fmt_score(report['avg_faithfulness'])}")
    print(f"平均相关性 answer_relevance: {_fmt_score(report['avg_answer_relevance'])}")
    print(f"库外负向不编造率 grounded(正确拒绝/说明不足): {_fmt_score(report['out_of_kb_grounded_rate'])}")
    print()
    print(f"{'用例':<6} {'分支':<8} {'忠实':>5} {'相关':>5} {'落地':>5}  答案 / 判定")
    print("-" * 100)
    for r in records:
        if r.get("skipped"):
            continue
        branch = _BRANCH_NAME.get(r["branch"], r["branch"])
        flag = "✓" if r["grounded"] else "✗ 编造!"
        ans = (r["answer"] or "").replace("\n", " ")
        if len(ans) > 60:
            ans = ans[:60] + "…"
        print(f"{r['id']:<6} {branch:<8} {_fmt_score(r['faithfulness']):>5} "
              f"{_fmt_score(r['answer_relevance']):>5} {flag:>5}  {ans}")
        if r.get("judge_reason"):
            print(f"       判因: {r['judge_reason']}")
        if r.get("error"):
            print(f"       错误: {r['error']}")
    print(f"\n报告已写入: {args.report}")
    if args.fake:
        print("提示: 去掉 --fake 用真实 LLM 评测（需配置 LLM_API_KEY）。")


if __name__ == "__main__":
    main()
