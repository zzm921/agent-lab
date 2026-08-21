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


async def test_unknown_mode(settings, registry, sessions):
    llm = FakeChatModel()
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="nope")
    assert any(e["type"] == "error" for e in events)
