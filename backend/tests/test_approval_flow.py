"""HITL 审批流：中断 → 批准/拒绝/修改 → 恢复执行。"""
from langchain_core.messages import AIMessage

from app.agents.harness import AgentHarness
from app.llm.fake_model import FakeChatModel
from tests.conftest import ai_with_tool, collect_stream


async def _approval_harness(settings, registry, sessions, script):
    llm = FakeChatModel()
    llm.script = script
    return AgentHarness(settings, llm, registry, sessions)


async def test_approve_flow(settings, registry, sessions):
    script = [ai_with_tool("需要计算", args={"expression": "1+1"}), AIMessage(content="最终结果 2")]
    harness = await _approval_harness(settings, registry, sessions, script)
    events = await collect_stream(harness, approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")
    assert request["tool_calls"][0]["name"] == "calculator"
    assert "done" not in [e["type"] for e in events]  # 暂停，未完成

    resumed = []
    async for ev in harness.resume(request["approval_id"], "approve", {}):
        resumed.append(ev)
    types = [e["type"] for e in resumed]
    assert "tool_start" in types
    assert "tool_end" in types
    assert "done" in types


async def test_reject_flow(settings, registry, sessions):
    script = [ai_with_tool("需要计算", args={"expression": "1+1"}), AIMessage(content="好的，不调用工具")]
    harness = await _approval_harness(settings, registry, sessions, script)
    events = await collect_stream(harness, approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in harness.resume(request["approval_id"], "reject", {}):
        resumed.append(ev)
    tool_end = next(e for e in resumed if e["type"] == "tool_end")
    assert tool_end["success"] is False
    assert "拒绝" in tool_end["result"]


async def test_modify_flow(settings, registry, sessions):
    script = [ai_with_tool("需要计算", args={"expression": "1+1"}), AIMessage(content="最终结果 2")]
    harness = await _approval_harness(settings, registry, sessions, script)
    events = await collect_stream(harness, approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in harness.resume(request["approval_id"], "modify", {"call_1": {"expression": "2+3"}}):
        resumed.append(ev)
    tool_start = next(e for e in resumed if e["type"] == "tool_start")
    assert tool_start["args"] == {"expression": "2+3"}


async def test_modify_flow_by_name_fallback(settings, registry, sessions):
    """工具调用未带 id 时前端按名称回填参数，后端应通过名称兜底匹配并生效。"""
    script = [ai_with_tool("需要计算", args={"expression": "1+1"}), AIMessage(content="最终结果 2")]
    harness = await _approval_harness(settings, registry, sessions, script)
    events = await collect_stream(harness, approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in harness.resume(request["approval_id"], "modify", {"calculator": {"expression": "9+9"}}):
        resumed.append(ev)
    tool_start = next(e for e in resumed if e["type"] == "tool_start")
    assert tool_start["args"] == {"expression": "9+9"}


async def test_tool_count_accumulates_across_approvals(settings, registry, sessions):
    """同一轮内多次审批时，done 的工具调用数应累计，而非只统计最后一次 resume 之后。"""
    script = [
        ai_with_tool("需要计算1", args={"expression": "1+1"}, cid="call_1"),
        ai_with_tool("需要计算2", args={"expression": "2+2"}, cid="call_2"),
        AIMessage(content="最终结果 5"),
    ]
    harness = await _approval_harness(settings, registry, sessions, script)
    events = await collect_stream(harness, approval_policy="always", enabled=["calculator"])

    req1 = next(e for e in events if e["type"] == "approval_request")
    resumed1 = []
    async for ev in harness.resume(req1["approval_id"], "approve", {}):
        resumed1.append(ev)
    assert "done" not in [e["type"] for e in resumed1]  # 第二个工具仍需审批

    req2 = next(e for e in resumed1 if e["type"] == "approval_request")
    resumed2 = []
    async for ev in harness.resume(req2["approval_id"], "approve", {}):
        resumed2.append(ev)
    done = next(e for e in resumed2 if e["type"] == "done")
    assert done["stats"]["tool_calls"] == 2


async def test_plan_execute_approve_flow(settings, registry, sessions):
    """plan_execute（StateGraph 版）工具审批：中断 → 批准 → 恢复执行 → 完成。"""
    script = [
        AIMessage(content="步骤一"),
        ai_with_tool("需要计算", args={"expression": "1+1"}),
        AIMessage(content="最终结果 2"),
    ]
    harness = await _approval_harness(settings, registry, sessions, script)
    events = await collect_stream(harness, mode="plan_execute", approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")
    assert request["tool_calls"][0]["name"] == "calculator"
    assert "done" not in [e["type"] for e in events]  # 暂停，未完成

    resumed = []
    async for ev in harness.resume(request["approval_id"], "approve", {}):
        resumed.append(ev)
    types = [e["type"] for e in resumed]
    assert "tool_start" in types
    assert "tool_end" in types
    assert "done" in types


async def test_plan_execute_modify_flow(settings, registry, sessions):
    """plan_execute（StateGraph 版，旧 make_tools_node）：修改后的参数必须传入工具。"""
    script = [
        AIMessage(content="步骤一"),
        ai_with_tool("需要计算", args={"expression": "1+1"}),
        AIMessage(content="最终结果 2"),
    ]
    harness = await _approval_harness(settings, registry, sessions, script)
    events = await collect_stream(harness, mode="plan_execute", approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in harness.resume(request["approval_id"], "modify", {"call_1": {"expression": "2+3"}}):
        resumed.append(ev)
    tool_start = next(e for e in resumed if e["type"] == "tool_start")
    assert tool_start["args"] == {"expression": "2+3"}


async def test_plan_execute_modify_flow_by_name_fallback(settings, registry, sessions):
    """plan_execute 的 make_tools_node 按名称兜底匹配修改参数。"""
    script = [
        AIMessage(content="步骤一"),
        ai_with_tool("需要计算", args={"expression": "1+1"}),
        AIMessage(content="最终结果 2"),
    ]
    harness = await _approval_harness(settings, registry, sessions, script)
    events = await collect_stream(harness, mode="plan_execute", approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in harness.resume(request["approval_id"], "modify", {"calculator": {"expression": "4+4"}}):
        resumed.append(ev)
    tool_start = next(e for e in resumed if e["type"] == "tool_start")
    assert tool_start["args"] == {"expression": "4+4"}


async def test_resume_unknown_approval(settings, registry, sessions):
    harness = AgentHarness(settings, FakeChatModel(), registry, sessions)
    events = []
    async for ev in harness.resume("not-exist", "approve", {}):
        events.append(ev)
    assert events[0]["type"] == "error"
