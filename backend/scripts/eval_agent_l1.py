"""Agent L1 行为评测 CLI：跑全量任务集 → 任务层约束 + 架构层不变量 → 按模式汇总。

用法（在 backend/ 目录下）：
    python scripts/eval_agent_l1.py                    # 全部离线确定性，无需 Key
    python scripts/eval_agent_l1.py --runs 3 --real    # 鲁棒性：真实 LLM 驱动，每用例跑 3 次统计成功率（需 Key）
    python scripts/eval_agent_l1.py --report PATH      # 指定报告输出路径（默认 eval/agent/reports/agent_l1.json）

输出：
    - 控制台：按模式（react / plan_execute / reflection / multi_agent）汇总表
      + 成功率/成本/延迟（企业级补充维度）+ 失败用例明细；
    - JSON 报告：全量逐用例轨迹明细 + 汇总，供后续失败样本回流分析。
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

from eval.agent import runner  # noqa: E402

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
    parser = argparse.ArgumentParser(description="Agent L1 行为评测 · 任务层约束 + 架构层不变量")
    parser.add_argument("--runs", type=int, default=1, help="每用例执行次数（>1 输出成功率，配合 --real 才有统计意义）")
    parser.add_argument("--real", action="store_true", help="真实 LLM（chat 场景）驱动行为层（需 LLM_API_KEY）")
    parser.add_argument("--report", default="eval/agent/reports/agent_l1.json", help="报告输出路径")
    args = parser.parse_args()

    records, report = runner.run(runs=args.runs, real=args.real)
    runner.save_report(report, args.report)

    print("=== Agent L1 行为评测报告（任务/约束分离）===")
    print(f"模式: {report['meta']['mode']} | 驱动: {report['meta']['driver']} | 每用例 {report['meta']['runs']} 次")
    print(f"用例: {report['meta']['eval_cases']} 条 | 通过率 {report['overall']['pass_rate']:.3f} "
          f"({report['overall']['passed']}/{report['overall']['cases']})")
    print(
        f"必调满足 {_fmt(report['overall']['must_call_rate'])} | "
        f"轨迹序列 {_fmt(report['overall']['sequence_match_rate'])} | "
        f"参数包含 {_fmt(report['overall']['args_ok_rate'])} | "
        f"禁调命中 {report['overall']['forbidden_hits']} | "
        f"架构违规 {report['overall']['arch_failures']} | "
        f"审批闸门 {report['overall']['approval_cases']} 例 | "
        f"轮数上限 {report['overall']['loop_guard_cases']} 例 | "
        f"耗时 {report['overall']['avg_elapsed_ms']}ms"
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
    header = f"{'模式':<14} {'用例':>4}   {'通过':>4}   {'架构违规':>6}   {'耗时':>8}"
    print(header)
    print("-" * len(header))
    for mode in _MODE_ORDER:
        agg = report["overall"]["modes"].get(mode)
        if not agg:
            continue
        mode_records = [r for r in records if r["mode"] == mode]
        print(
            f"{_MODE_NAME[mode]:<14} {agg['cases']:>4}   {agg['passed']:>4}   "
            f"{agg['arch_failures']:>6}   {_avg_ms(mode_records):>7}ms"
        )
    print("-" * len(header))
    print(f"总计     {report['overall']['cases']:>4}   {report['overall']['passed']:>4}")

    _print_failures(records)
    print(f"\n报告已写入: {args.report}")


if __name__ == "__main__":
    main()
