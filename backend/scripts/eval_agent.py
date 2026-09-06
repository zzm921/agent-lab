"""Agent 全量评测 · P1 确定性断言层 CLI：跑全量评测集 → 轨迹断言指标 → 分支报告。

用法（在 backend/ 目录下）：
    python scripts/eval_agent.py                 # 全部离线确定性，无需 Key
    python scripts/eval_agent.py --report PATH   # 指定报告输出路径（默认 eval/reports/agent_latest.json）

输出：
    - 控制台：按分支汇总表 + 失败用例明细（序列失配 / 参数缺失 / 禁调命中）
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

from eval import agent_runner  # noqa: E402

_BRANCH_ORDER = ["single_tool", "two_step", "mixed_tools", "forbidden_command", "loop_guard"]
_BRANCH_NAME = {
    "single_tool": "单工具直答",
    "two_step": "多步工具链",
    "mixed_tools": "异源工具编排",
    "forbidden_command": "禁调审批闸门",
    "loop_guard": "轮数上限收口",
}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _print_failures(records: list[dict]) -> None:
    print("\n=== 失败用例明细（需人工核查）===")
    shown = 0
    for r in records:
        if r["passed"]:
            continue
        shown += 1
        print(f"[{r['id']}] ({_BRANCH_NAME.get(r['branch'], r['branch'])}) {r['query']}")
        print(f"    执行={r['executed']} 金标={r['gold']} 序列匹配={r['sequence_match']}")
        if r["args_failures"]:
            print(f"    参数缺失={r['args_failures']}")
        if r["forbidden_hits"]:
            print(f"    禁调命中={r['forbidden_hits']}")
        if r["error"]:
            print("    事件流出现 error")
        if shown >= 20:
            print("…（仅显示前 20 条）")
            break
    if not shown:
        print("（无）")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 全量评测 · P1 确定性断言层")
    parser.add_argument("--report", default="eval/reports/agent_latest.json", help="报告输出路径")
    args = parser.parse_args()

    records, report = agent_runner.run()
    agent_runner.save_report(report, args.report)

    print("=== Agent P1 确定性断言评测报告 ===")
    print(f"模式: {report['meta']['mode']}")
    print(f"用例: {report['meta']['eval_cases']} 条 | 通过率 {report['overall']['pass_rate']:.3f} "
          f"({report['overall']['passed']}/{report['overall']['cases']})")
    print(
        f"序列匹配 {_fmt(report['overall']['sequence_match_rate'])} | "
        f"参数包含 {_fmt(report['overall']['args_ok_rate'])} | "
        f"禁调命中 {report['overall']['forbidden_hits']} | "
        f"审批闸门 {report['overall']['approval_cases']} 例 | "
        f"轮数上限 {report['overall']['loop_guard_cases']} 例 | "
        f"耗时 {report['overall']['avg_elapsed_ms']}ms"
    )
    print()
    header = f"{'分支':<14} {'用例':>4}   {'通过':>4}   {'序列':>5}   {'参数':>5}   {'耗时':>8}"
    print(header)
    print("-" * len(header))
    for branch in _BRANCH_ORDER:
        agg = report["overall"]["branches"].get(branch)
        if not agg:
            continue
        br_records = [r for r in records if r["branch"] == branch]
        print(
            f"{_BRANCH_NAME[branch]:<14} {agg['cases']:>4}   {agg['passed']:>4}   "
            f"{_fmt(_mean_seq(br_records)):>5}   {_fmt(_mean_args(br_records)):>5}   "
            f"{_avg_ms(br_records):>7}ms"
        )
    print("-" * len(header))
    print(f"总计     {report['overall']['cases']:>4}   {report['overall']['passed']:>4}")

    _print_failures(records)
    print(f"\n报告已写入: {args.report}")


def _mean_seq(records: list[dict]) -> float | None:
    vals = [r["sequence_match"] for r in records if r["sequence_match"] is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _mean_args(records: list[dict]) -> float | None:
    vals = [r["args_ok"] for r in records if r["args_ok"] is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _avg_ms(records: list[dict]) -> float:
    return round(sum(r["elapsed_ms"] for r in records) / len(records), 1) if records else 0.0


if __name__ == "__main__":
    main()
