"""Agent 全量评测 CLI：L0/L1 离线断言 + L2 LLM-judge 答案评分 + 企业级补充维度。

用法（在 backend/ 目录下）：
    python scripts/eval_agent.py                    # 真实 LLM 评分（仅需 LLM_API_KEY，agent_judge 场景，不复用 RAGAS）
    python scripts/eval_agent.py --fake             # 离线冒烟，仅验证链路（分数无评测意义）
    python scripts/eval_agent.py --runs 3 --real    # 鲁棒性：真实 LLM 驱动行为层，每用例跑 3 次统计成功率（需 Key）
    python scripts/eval_agent.py --report PATH      # 指定报告输出路径（默认 eval/agent/reports/agent.json）

输出：
    - 控制台：按模式（react / plan_execute / reflection / multi_agent）汇总表，
      含通过率 + 答案质量均分（正确性/相关性/事实一致性）+ 成功率（--runs N 时）+
      成本（模型调用次数 / token）+ 延迟分布（min/p50/p95/max）；失败用例明细；
    - JSON 报告：全量逐用例行为断言 + 答案 + 工具证据 + 评分 + 成本 + 成功率，供失败样本回流分析。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许在 backend 任意相对路径下执行：把 backend/ 加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 评测为离线确定性回归：禁用运行记录落盘，避免评测跑写 data/telemetry
from app.config import settings as global_settings  # noqa: E402

global_settings.telemetry_enabled = False

from eval.agent import full  # noqa: E402

_MODE_ORDER = ["react", "plan_execute", "reflection", "multi_agent"]
_MODE_NAME = {
    "react": "ReAct 线性循环",
    "plan_execute": "计划-执行",
    "reflection": "生成-评审-修订",
    "multi_agent": "编排-委派",
}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _avg_ms(records: list[dict]) -> float:
    return round(sum(r["elapsed_ms"] for r in records) / len(records), 1) if records else 0.0


def _print_failures(records: list[dict]) -> None:
    print("\n=== 失败用例明细（需人工核查）===")
    shown = 0
    for r in records:
        if r["passed"]:
            continue
        shown += 1
        print(f"[{r['id']}] ({_MODE_NAME.get(r['mode'], r['mode'])}) {r['query']}")
        print(f"    执行={r['executed']} 必调={r['must_call']} 序列={r['sequence']}")
        if r["sequence_match"] is False:
            print("    轨迹序列失配")
        if r["args_failures"]:
            print(f"    参数缺失={r['args_failures']}")
        if r["forbidden_hits"]:
            print(f"    禁调命中={r['forbidden_hits']}")
        if r["arch_failures"]:
            print(f"    架构不变量违规={r['arch_failures']}")
        if r["error"]:
            print("    事件流出现 error")
        if shown >= 20:
            print("…（仅显示前 20 条）")
            break
    if not shown:
        print("（无）")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 全量评测 · 行为断言 + LLM-judge 答案质量")
    parser.add_argument("--fake", action="store_true", help="fake 模式：离线冒烟（仅验证链路，无评测意义）")
    parser.add_argument("--runs", type=int, default=1, help="每用例执行次数（>1 输出成功率，配合 --real 才有统计意义）")
    parser.add_argument("--real", action="store_true", help="真实 LLM（chat 场景）驱动行为层（需 LLM_API_KEY）")
    parser.add_argument("--report", default="eval/agent/reports/agent.json", help="报告输出路径")
    args = parser.parse_args()

    records, report = full.run(fake=args.fake, runs=args.runs, real=args.real)
    full.save_report(report, args.report)

    print("=== Agent 全量评测报告（L0/L1 行为断言 + L2 LLM-judge 答案质量）===")
    print(f"模式: {report['meta']['mode']} | 驱动: {report['meta']['driver']} | 每用例 {report['meta']['runs']} 次")
    print(f"用例: {report['meta']['eval_cases']} 条 | 通过率 {report['overall']['pass_rate']:.3f} "
          f"({report['overall']['passed']}/{report['overall']['cases']}) | "
          f"答案评分 {report['overall']['answer']['scored_cases']} 条")
    print(
        f"必调满足 {_fmt(report['overall']['must_call_rate'])} | "
        f"轨迹序列 {_fmt(report['overall']['sequence_match_rate'])} | "
        f"参数包含 {_fmt(report['overall']['args_ok_rate'])} | "
        f"禁调命中 {report['overall']['forbidden_hits']} | "
        f"架构违规 {report['overall']['arch_failures']} | "
        f"耗时 {report['overall']['avg_elapsed_ms']}ms"
    )
    ans = report["overall"]["answer"]
    print(
        f"答案质量 | 正确性 {_fmt(ans['answer_correctness'])} | "
        f"相关性 {_fmt(ans['answer_relevancy'])} | "
        f"事实一致性 {_fmt(ans['faithfulness'])}（{ans['faithful_cases']} 条有证据用例）"
    )
    cost = report["overall"]["cost"]
    lat = report["overall"]["latency_ms"]
    print(
        f"成本 | 平均模型调用 {_fmt(cost['avg_model_calls'])} 次/用例 | "
        f"总 token {cost['total_tokens']} | 平均 {cost['avg_tokens']} token/用例"
    )
    print(
        f"延迟 | min {lat['min']}ms | p50 {lat['p50']}ms | p95 {lat['p95']}ms | max {lat['max']}ms"
    )
    if report["meta"]["runs"] > 1:
        print(f"成功率 | {report['overall']['success_rate']:.3f}（通过次数/每用例运行次数，真实 LLM 驱动下才有统计意义）")
    print()
    header = (
        f"{'模式':<14} {'用例':>4} {'通过':>4} {'架构违规':>6} "
        f"{'正确性':>7} {'相关性':>7} {'事实一致':>7} {'成功率':>7} {'耗时':>8}"
    )
    print(header)
    print("-" * len(header))
    for mode in _MODE_ORDER:
        agg = report["overall"]["modes"].get(mode)
        if not agg:
            continue
        mode_records = [r for r in records if r["mode"] == mode]
        print(
            f"{_MODE_NAME[mode]:<14} {agg['cases']:>4} {agg['passed']:>4} {agg['arch_failures']:>6} "
            f"{_fmt(agg['avg_answer_correctness']):>7} {_fmt(agg['avg_answer_relevancy']):>7} "
            f"{_fmt(agg['avg_faithfulness']):>7} {_fmt(agg['avg_success_rate']):>7} "
            f"{_avg_ms(mode_records):>7}ms"
        )
    print("-" * len(header))
    print(f"总计     {report['overall']['cases']:>4} {report['overall']['passed']:>4}")

    _print_failures(records)
    print(f"\n报告已写入: {args.report}")


if __name__ == "__main__":
    main()
