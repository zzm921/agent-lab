"""Agent 全量评测 · 任务层约束 + 架构层不变量回归门禁（沿用 eval_rag_l1 的硬阈值模式）。

把守「Agent 编排链」的确定性基线：给定脚本化决策，真实 Harness 必须满足——
- 任务层（agent 无关）：must_call 必调满足、禁调零命中、参数包含、审批闸门、轮数上限；
- 架构层（agent 特定）：plan_execute 计划协议、reflection 评审协议、multi_agent 委派协议。

说明：
- 全部离线确定性：FakeChatModel 脚本化决策 + 真实 create_agent / StateGraph 循环，
  无 LLM 调用、不依赖 Key；
- 任务集 8 条 × 4 种架构（react / plan_execute / reflection / multi_agent）；
- 阈值基于当前基线留出余量，仅约束「不回归」。
"""
from __future__ import annotations

from eval.agent import runner

_MODES = {"react", "plan_execute", "reflection", "multi_agent"}


def _run() -> tuple[list[dict], dict]:
    """跑全量 Agent 评测（任务/约束分离），返回 (逐用例记录, 汇总报告)。"""
    return runner.run()


def test_all_cases_pass() -> None:
    """全部用例必须通过（任务层 + 架构层逐项达标）。"""
    _, report = _run()
    assert report["overall"]["pass_rate"] == 1.0


def test_mode_coverage() -> None:
    """四种架构至少各 1 例，用例总数 ≥ 8。"""
    _, report = _run()
    assert _MODES <= set(report["overall"]["modes"])
    assert report["meta"]["eval_cases"] >= 8


def test_must_call_satisfied() -> None:
    """每个用例的必调工具必须全部出现在实际执行序列中。"""
    _, report = _run()
    assert report["overall"]["must_call_rate"] == 1.0


def test_react_trajectory_exact() -> None:
    """react 用例可选精确轨迹：执行序列与 sequence 完全一致（顺序、工具名逐一相等）。"""
    records, _ = _run()
    react = [r for r in records if r["mode"] == "react" and r["sequence"] is not None]
    assert react, "react 用例缺少 sequence 精确轨迹约束"
    assert all(r["sequence_match"] for r in react)


def test_args_contain_expected() -> None:
    """关键参数必须出现在实际调用参数中（键存在 + 值子串包含）。"""
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
    assert len(guard[0]["executed"]) == len(guard[0]["must_call"]) or guard[0]["executed"]


def test_plan_execute_arch_spec() -> None:
    """架构不变量：plan_execute 必须产出 plan created（步骤数 ∈ [2,5]）且 plan done 收尾。"""
    records, _ = _run()
    cases = [r for r in records if r["mode"] == "plan_execute"]
    assert cases, "评测集缺少 plan_execute 模式用例"
    assert all(not r["arch_failures"] for r in cases)


def test_reflection_arch_spec() -> None:
    """架构不变量：reflection 必须产出 reflect(draft) 且评审以 PASS 通过。"""
    records, _ = _run()
    cases = [r for r in records if r["mode"] == "reflection"]
    assert cases, "评测集缺少 reflection 模式用例"
    assert all(not r["arch_failures"] for r in cases)


def test_multi_agent_arch_spec() -> None:
    """架构不变量：multi_agent 必须委派通用 worker subagent。"""
    records, _ = _run()
    cases = [r for r in records if r["mode"] == "multi_agent"]
    assert cases, "评测集缺少 multi_agent 模式用例"
    assert all(not r["arch_failures"] for r in cases)
    assert all("calculator" in r["executed"] for r in cases), "编排者委派的 worker 未真实执行工具"


def test_eval_stays_offline_fast() -> None:
    """评测必须保持毫秒级平均耗时：若出现秒级，多半混入了真实 LLM 调用。"""
    _, report = _run()
    assert report["overall"]["avg_elapsed_ms"] < 2000.0


def test_robustness_cost_latency_fields() -> None:
    """企业级补充维度：成功率 / 成本 / 延迟分布。

    - fake 确定性下 runs=2 两次结果必同 → success_rate ∈ {0, 1}，pass_count ∈ {0, 2}；
    - 逐用例带 runs / pass_count / success_rate，overall 带 cost 与 latency_ms；
    - fake 无 usage_metadata → token 恒为 0，但模型调用次数 > 0（agent 循环真实调用）；
    - 默认 runs=1 时语义不变（回归不破坏）。
    """
    records, report = runner.run(runs=2)

    assert all(r["runs"] == 2 for r in records)
    assert all(r["pass_count"] in (0, 2) for r in records)
    assert all(r["success_rate"] in (0.0, 1.0) for r in records)
    assert report["overall"]["success_rate"] is not None

    cost = report["overall"]["cost"]
    assert cost["avg_model_calls"] is not None and cost["avg_model_calls"] > 0
    assert cost["total_tokens"] == 0  # fake 无 usage_metadata

    lat = report["overall"]["latency_ms"]
    assert lat["min"] <= lat["p50"] <= lat["p95"] <= lat["max"]

    # 默认单次运行语义不变
    _, single = runner.run()
    assert single["overall"]["success_rate"] is not None
    assert all(r["runs"] == 1 for r in single["cases"])


def test_answer_layer_smoke_and_golden() -> None:
    """L2 答案层离线冒烟：链路可跑通，金标完整，答案/证据提取符合预期。

    - 所有产出最终答案的用例必须带 reference 金标（LLM-judge 对照标准）；
    - 答案从事件流最终 message 提取（计算用例答案须含关键事实）；
    - 工具证据从 tool_end 提取（计算用例证据须含工具名）；
    - 审批 / 轮数上限用例无最终答案，不进入评分。
    """
    from eval.agent import full

    records, report = full.run(fake=True)
    by_id = {r["id"]: r for r in records}

    scored = [r for r in records if r["answer_scored"]]
    assert len(scored) >= 6, "评测集缺少可评分的答案用例"
    assert all(r["reference"] for r in scored), "金标 reference 缺失"

    assert "25" in by_id["agent-001"]["answer"], "react 计算用例答案未从事件流提取到关键事实"
    assert any("calculator" in e for e in by_id["agent-001"]["evidence"]), "工具证据未从 tool_end 提取"

    assert not by_id["agent-004"]["answer_scored"], "审批用例不应进入答案评分"
    assert not by_id["agent-005"]["answer_scored"], "轮数上限用例不应进入答案评分"

    assert report["overall"]["answer"]["scored_cases"] == len(scored)
    assert report["overall"]["answer"]["answer_correctness"] is None  # fake 占位文本解析必败，分数无评测意义
