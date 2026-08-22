"""四种推理模式测试：使用 FakeChatModel 脚本驱动确定性执行。"""
from langchain_core.messages import AIMessage

from app.agents.runner import AgentRunner
from app.llm.fake_model import FakeChatModel
from tests.conftest import ai_with_tool, collect_stream


async def _runner_with(registry, sessions, settings, script):
    llm = FakeChatModel()
    llm.script = script
    return AgentRunner(settings, llm, registry, sessions)


async def test_react_direct_answer(settings, registry, sessions):
    runner = await _runner_with(registry, sessions, settings, [AIMessage(content="直接回答：42")])
    events = await collect_stream(runner, mode="react")
    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert "message" in types
    assert "done" in types
    assert not any(e["type"] == "tool_start" for e in events)


async def test_react_tool_loop(settings, registry, sessions):
    script = [
        ai_with_tool("需要计算", args={"expression": "1+1"}),
        AIMessage(content="计算结果是 2"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="react", enabled=["calculator"])
    types = [e["type"] for e in events]
    assert "tool_start" in types
    assert "tool_end" in types
    assert "message" in types
    tool_end = next(e for e in events if e["type"] == "tool_end")
    assert tool_end["success"] is True
    assert "2" in tool_end["result"]


async def test_plan_execute(settings, registry, sessions):
    script = [
        AIMessage(content="步骤一\n步骤二"),
        AIMessage(content="已执行步骤一"),
        AIMessage(content="已执行步骤二"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="plan_execute")
    plan_events = [e for e in events if e["type"] == "plan"]
    assert plan_events and plan_events[0]["status"] == "created"
    assert plan_events[-1]["status"] == "done"
    assert plan_events[0]["steps"] == ["步骤一", "步骤二"]
    assert any(e["type"] == "message" for e in events)


async def test_plan_execute_replan(settings, registry, sessions):
    # 第 1 步调用计算器触发除零失败 → 模型放弃该步 → should_replan 进入 replanner 生成新计划 → 继续执行
    script = [
        AIMessage(content="步骤一\n步骤二"),
        ai_with_tool("尝试计算", args={"expression": "1/0"}),
        AIMessage(content="该步失败，无法继续"),
        AIMessage(content="新步骤一\n新步骤二"),
        AIMessage(content="执行新步骤一"),
        AIMessage(content="执行新步骤二"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="plan_execute", enabled=["calculator"])
    plan_events = [e for e in events if e["type"] == "plan"]
    assert len([e for e in plan_events if e["status"] == "created"]) >= 2  # 初次计划 + 重规划
    assert plan_events[-1]["status"] == "done"
    assert any(e["type"] == "tool_end" and not e["success"] for e in events)


async def test_stop_cancels_running_execution(settings, registry, sessions):
    # 在首个工具事件处调用 stop：后台任务被取消，产出「已停止执行」done，且不会继续生成最终答案
    script = [
        ai_with_tool("计算 1", args={"expression": "1+1"}, cid="call_1"),
        ai_with_tool("计算 2", args={"expression": "2+2"}, cid="call_2"),
        ai_with_tool("计算 3", args={"expression": "3+3"}, cid="call_3"),
        AIMessage(content="最终结果 6"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    session_id = "s_stop"
    seen = []
    async for ev in runner.stream(session_id, "测试任务", "react", ["calculator"], "standard", "never"):
        seen.append(ev)
        if ev.get("type") == "tool_start":
            runner.stop(session_id)
    assert runner.harness._tasks.get(session_id) is None or runner.harness._tasks[session_id].done()
    done = next((e for e in seen if e.get("type") == "done"), None)
    assert done is not None and "停止" in done["summary"]
    full = "".join(e.get("delta", "") for e in seen if e["type"] == "message")
    assert "最终结果" not in full  # 已停止，未输出最终回答


async def test_reflection_revise_loop(settings, registry, sessions):
    script = [
        AIMessage(content="草稿答案"),
        AIMessage(content="需要补充细节"),
        AIMessage(content="修订后的完整答案"),
        AIMessage(content="无"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="reflection")
    types = [e["type"] for e in events]
    assert "reflect" in types
    assert "revise" in types
    reflect_events = [e for e in events if e["type"] == "reflect"]
    assert reflect_events[0]["stage"] == "draft"
    assert reflect_events[-1]["critique"] == "无"
    assert any(e["type"] == "message" for e in events)


async def test_multi_agent(settings, registry, sessions):
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool("派发计算", name="compute", args={"task": "计算 1+1"}),
        AIMessage(content="计算结论：完成计算"),
        ai_with_tool("派发分析", name="analyze", args={"task": "分析可行性"}),
        AIMessage(content="分析结论：任务可行"),
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    types = [e["type"] for e in events]
    assert "agent_event" in types
    agent_events = [e for e in events if e["type"] == "agent_event"]
    assert any(e["worker"] == "orchestrator" and e["status"] == "dispatch" for e in agent_events)
    assert any(e["worker"] == "compute" and e["status"] == "done" for e in agent_events)
    assert any(e["worker"] == "analyze" and e["status"] == "done" for e in agent_events)
    assert "message" in types
    assert "done" in types


async def test_tool_call_limit(settings, registry, sessions):
    # tool_max_calls=2：前 2 次工具调用正常执行，第 3 次被护栏拒绝（短路，不执行工具）
    script = [
        ai_with_tool("计算 1", args={"expression": "1+1"}, cid="c1"),
        ai_with_tool("计算 2", args={"expression": "2+2"}, cid="c2"),
        ai_with_tool("计算 3", args={"expression": "3+3"}, cid="c3"),
        AIMessage(content="结束"),
    ]
    settings.tool_max_calls = 2
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="react", enabled=["calculator"])
    assert len([e for e in events if e["type"] == "tool_end" and e["success"]]) == 2
    assert any(e["type"] == "tool_end" and not e["success"] and "上限" in e["result"] for e in events)


async def test_circuit_breaker(settings, registry, sessions):
    # 同一工具连续失败达到阈值（2）后，后续调用被熔断短路（不再执行工具）
    script = [
        ai_with_tool("除零 1", args={"expression": "1/0"}, cid="c1"),
        ai_with_tool("除零 2", args={"expression": "1/0"}, cid="c2"),
        ai_with_tool("除零 3", args={"expression": "1/0"}, cid="c3"),
        AIMessage(content="结束"),
    ]
    settings.circuit_fail_threshold = 2
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="react", enabled=["calculator"])
    assert any(e["type"] == "tool_end" and not e["success"] and "熔断" in e["result"] for e in events)


async def test_fault_injection_error(settings, registry, sessions):
    # 故障注入 error：工具处理节点钩子直接返回模拟报错（不执行工具、不触发审批）
    script = [
        ai_with_tool("计算", args={"expression": "1+1"}, cid="c1"),
        AIMessage(content="结束"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    runner.harness.set_fault("calculator", "error")
    events = await collect_stream(runner, mode="react", enabled=["calculator"], approval_policy="always")
    assert any(e["type"] == "tool_end" and not e["success"] and "故障注入" in e["result"] for e in events)
    assert not any(e["type"] == "approval_request" for e in events)  # 注入短路，不触发审批


async def test_fault_injection_triggers_circuit(settings, registry, sessions):
    # 连续故障注入（超时）达到阈值后触发熔断：后续调用被短路为「已触发熔断」
    script = [
        ai_with_tool("超时 1", args={"expression": "1+1"}, cid="c1"),
        ai_with_tool("超时 2", args={"expression": "1+1"}, cid="c2"),
        ai_with_tool("超时 3", args={"expression": "1+1"}, cid="c3"),
        AIMessage(content="结束"),
    ]
    settings.circuit_fail_threshold = 2
    settings.tool_retry_base_delay = 0.001
    settings.tool_retry_max_delay = 0.002
    runner = await _runner_with(registry, sessions, settings, script)
    runner.harness.set_fault("calculator", "timeout")
    events = await collect_stream(runner, mode="react", enabled=["calculator"])
    faulted = [e for e in events if e["type"] == "tool_end" and "故障注入" in e["result"]]
    assert len(faulted) == 2  # 前两次模拟超时（每次内部透明重试后仍失败）
    assert any(e["type"] == "tool_end" and "熔断" in e["result"] for e in events)  # 第三次被熔断


async def test_fault_transient_type_triggers_direct_retry(settings, registry, sessions):
    # 瞬时故障注入类型（如 http_500）：工具层透明重试，发 tool_retry 事件；
    # 重试耗尽后返回结构化错误给模型（Agent 层思考后重试），且不触发审批
    script = [
        ai_with_tool("触发 500", args={"expression": "1+1"}, cid="c1"),
        AIMessage(content="结束"),
    ]
    settings.tool_retry_base_delay = 0.001
    settings.tool_retry_max_delay = 0.002
    runner = await _runner_with(registry, sessions, settings, script)
    runner.harness.set_fault("calculator", "http_500")
    events = await collect_stream(runner, mode="react", enabled=["calculator"], approval_policy="always")
    retry_events = [e for e in events if e["type"] == "tool_retry"]
    assert len(retry_events) == 2  # tool_retry_max=3：原始 1 次 + 直接重试 2 次
    assert retry_events[0]["tool"] == "calculator"
    assert retry_events[0]["max"] == 3
    end = next(e for e in events if e["type"] == "tool_end" and not e["success"])
    assert "故障注入" in end["result"]
    assert "瞬时错误" in end["result"]  # 结构化错误给模型
    assert not any(e["type"] == "approval_request" for e in events)  # 故障注入短路审批


async def test_fault_permanent_type_goes_to_model(settings, registry, sessions):
    # 永久故障注入类型（如 http_400）：不直接重试（无 tool_retry 事件），错误直接返回给模型思考后重试
    script = [
        ai_with_tool("触发 400", args={"expression": "1+1"}, cid="c1"),
        AIMessage(content="结束"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    runner.harness.set_fault("calculator", "http_400")
    events = await collect_stream(runner, mode="react", enabled=["calculator"])
    assert not any(e["type"] == "tool_retry" for e in events)
    end = next(e for e in events if e["type"] == "tool_end" and not e["success"])
    assert "故障注入" in end["result"]
    assert "参数校验失败" in end["result"]


async def test_circuit_allows_retry_with_different_args(settings, registry, sessions):
    # 熔断按「工具+参数」计：相同参数连续失败达阈值熔断该参数调用，
    # 但模型换参数重试仍可正常执行（不会直接告诉用户工具不可用）
    script = [
        ai_with_tool("除零 1", args={"expression": "1/0"}, cid="c1"),
        ai_with_tool("除零 2", args={"expression": "1/0"}, cid="c2"),
        ai_with_tool("除零 3", args={"expression": "1/0"}, cid="c3"),  # 相同参数 → 熔断短路
        ai_with_tool("正确计算", args={"expression": "1+1"}, cid="c4"),  # 换参数 → 放行执行
        AIMessage(content="结束"),
    ]
    settings.circuit_fail_threshold = 2
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="react", enabled=["calculator"])
    assert sum(1 for e in events if e["type"] == "tool_end" and "熔断" in e["result"]) == 1  # 仅相同参数被熔断
    assert sum(1 for e in events if e["type"] == "tool_end" and e["success"]) == 1  # 换参数后成功执行


async def test_system_prompt_includes_tool_retry_hint(settings, registry, sessions):
    # 首轮 system prompt 应包含「工具失败可重试（换参数）」的规范，引导模型失败后重试而非直接说不可用
    runner = await _runner_with(registry, sessions, settings, [])
    graph = runner._build_graph("react", [], lambda d: None)
    config = {"configurable": {"thread_id": "s1", "approval_policy": "never"}}
    inputs = await runner._make_inputs(graph, config, "你好", "standard")
    system = next(m for m in inputs["messages"] if m.type == "system")
    assert "工具使用规范" in system.content
    assert "重试" in system.content
    assert "不要直接告诉用户工具不可用" in system.content


async def test_unknown_mode(settings, registry, sessions):
    llm = FakeChatModel()
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="nope")
    assert any(e["type"] == "error" for e in events)
