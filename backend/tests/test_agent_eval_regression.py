"""Agent 全量评测 · P1 确定性断言回归门禁（沿用 eval_modular 的硬阈值模式）。

把守「Agent 编排链」的确定性基线：给定脚本化决策，真实 Harness 必须
按期望序列执行工具、传递正确参数、零禁调命中、高危工具必触发审批、
轮数上限按时收口——防止后续改动悄悄破坏工具循环 / 事件流 / 护栏。

说明：
- 全部离线确定性：FakeChatModel 脚本化决策 + 真实 create_agent 循环，无 LLM 调用、不依赖 Key；
- 评测集 5 条，覆盖五个分支（单工具 / 多步链 / 异源编排 / 禁调审批 / 轮数上限）；
- 阈值基于当前基线留出余量，仅约束「不回归」。
"""
from __future__ import annotations

from eval import agent_runner


def _run() -> tuple[list[dict], dict]:
    """跑全量 Agent P1 确定性评测，返回 (逐用例记录, 汇总报告)。"""
    return agent_runner.run()


def test_all_cases_pass() -> None:
    """全部用例必须通过（序列 / 参数 / 禁调 / 审批 / 轮数收口逐项达标）。"""
    _, report = _run()
    assert report["overall"]["pass_rate"] == 1.0


def test_tool_sequence_exact_match() -> None:
    """工具执行序列必须与金标序列完全一致（顺序、工具名逐一相等）。"""
    _, report = _run()
    assert report["overall"]["sequence_match_rate"] == 1.0


def test_args_contain_expected() -> None:
    """每个金标步骤的关键参数必须出现在实际调用参数中（键存在 + 值子串包含）。"""
    _, report = _run()
    assert report["overall"]["args_ok_rate"] == 1.0


def test_forbidden_tools_zero_calls() -> None:
    """禁调工具整轮零调用（forbidden_command 用例不得被自动执行）。"""
    _, report = _run()
    assert report["overall"]["forbidden_hits"] == 0


def test_approval_gate_for_high_risk_tool() -> None:
    """高危工具（run_command）被调用时必须触发审批请求，且不得执行。"""
    records, _ = _run()
    cmd = [r for r in records if r["branch"] == "forbidden_command"]
    assert cmd, "评测集缺少 forbidden_command 分支用例"
    assert cmd[0]["approval_requested"] is True
    assert cmd[0]["executed"] == []


def test_loop_guard_enforced() -> None:
    """模型连续出牌超 max_steps 时，harness 必须按上限收口并给出「上限」提示。"""
    records, _ = _run()
    guard = [r for r in records if r["branch"] == "loop_guard"]
    assert guard, "评测集缺少 loop_guard 分支用例"
    assert guard[0]["done_limit"] is True
    assert len(guard[0]["executed"]) == len(guard[0]["gold"])


def test_branch_coverage() -> None:
    """评测集分支覆盖完备性：五个分支至少各 1 例，用例总数 ≥ 5。"""
    _, report = _run()
    branches = set(report["overall"]["branches"])
    assert {"single_tool", "two_step", "mixed_tools", "forbidden_command", "loop_guard"} <= branches
    assert report["meta"]["eval_cases"] >= 5


def test_eval_stays_offline_fast() -> None:
    """P1 确定性评测必须保持毫秒级平均耗时：若出现秒级，多半混入了真实 LLM 调用。"""
    _, report = _run()
    assert report["overall"]["avg_elapsed_ms"] < 2000.0
