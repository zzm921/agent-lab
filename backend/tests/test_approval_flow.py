"""HITL 审批流：中断 → 批准/拒绝/修改 → 恢复执行。"""
from langchain_core.messages import AIMessage

from app.agents.runner import AgentRunner
from app.llm.fake_model import FakeChatModel
from tests.conftest import ai_with_tool, collect_stream


async def _approval_runner(settings, registry, sessions, script):
    llm = FakeChatModel()
    llm.script = script
    return AgentRunner(settings, llm, registry, sessions)


async def test_approve_flow(settings, registry, sessions):
    script = [ai_with_tool("需要计算", args={"expression": "1+1"}), AIMessage(content="最终结果 2")]
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")
    assert request["tool_calls"][0]["name"] == "calculator"
    assert "done" not in [e["type"] for e in events]  # 暂停，未完成

    resumed = []
    async for ev in runner.resume(request["approval_id"], "approve", {}):
        resumed.append(ev)
    types = [e["type"] for e in resumed]
    assert "tool_start" in types
    assert "tool_end" in types
    assert "done" in types


async def test_reject_flow(settings, registry, sessions):
    script = [ai_with_tool("需要计算", args={"expression": "1+1"}), AIMessage(content="好的，不调用工具")]
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in runner.resume(request["approval_id"], "reject", {}):
        resumed.append(ev)
    tool_end = next(e for e in resumed if e["type"] == "tool_end")
    assert tool_end["success"] is False
    assert "拒绝" in tool_end["result"]


async def test_modify_flow(settings, registry, sessions):
    script = [ai_with_tool("需要计算", args={"expression": "1+1"}), AIMessage(content="最终结果 2")]
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in runner.resume(request["approval_id"], "modify", {"call_1": {"expression": "2+3"}}):
        resumed.append(ev)
    tool_start = next(e for e in resumed if e["type"] == "tool_start")
    assert tool_start["args"] == {"expression": "2+3"}


async def test_modify_flow_by_name_fallback(settings, registry, sessions):
    """工具调用未带 id 时前端按名称回填参数，后端应通过名称兜底匹配并生效。"""
    script = [ai_with_tool("需要计算", args={"expression": "1+1"}), AIMessage(content="最终结果 2")]
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in runner.resume(request["approval_id"], "modify", {"calculator": {"expression": "9+9"}}):
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
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, approval_policy="always", enabled=["calculator"])

    req1 = next(e for e in events if e["type"] == "approval_request")
    resumed1 = []
    async for ev in runner.resume(req1["approval_id"], "approve", {}):
        resumed1.append(ev)
    assert "done" not in [e["type"] for e in resumed1]  # 第二个工具仍需审批

    req2 = next(e for e in resumed1 if e["type"] == "approval_request")
    resumed2 = []
    async for ev in runner.resume(req2["approval_id"], "approve", {}):
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
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, mode="plan_execute", approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")
    assert request["tool_calls"][0]["name"] == "calculator"
    assert "done" not in [e["type"] for e in events]  # 暂停，未完成

    resumed = []
    async for ev in runner.resume(request["approval_id"], "approve", {}):
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
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, mode="plan_execute", approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in runner.resume(request["approval_id"], "modify", {"call_1": {"expression": "2+3"}}):
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
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, mode="plan_execute", approval_policy="always", enabled=["calculator"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in runner.resume(request["approval_id"], "modify", {"calculator": {"expression": "4+4"}}):
        resumed.append(ev)
    tool_start = next(e for e in resumed if e["type"] == "tool_start")
    assert tool_start["args"] == {"expression": "4+4"}


async def test_resume_unknown_approval(settings, registry, sessions):
    runner = AgentRunner(settings, FakeChatModel(), registry, sessions)
    events = []
    async for ev in runner.resume("not-exist", "approve", {}):
        events.append(ev)
    assert events[0]["type"] == "error"


async def test_command_tool_forced_hitl_even_when_never(settings, registry, sessions):
    """run_command 无论审批策略如何都强制 HITL：approval_policy=never 也必须先审批。"""
    script = [
        ai_with_tool("执行命令", name="run_command", args={"command": "echo hi"}),
        AIMessage(content="完成"),
    ]
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, enabled=["run_command"])  # approval_policy 默认 never
    request = next(e for e in events if e["type"] == "approval_request")
    assert request["tool_calls"][0]["name"] == "run_command"
    assert "done" not in [e["type"] for e in events]  # 暂停等待人工审批

    resumed = []
    async for ev in runner.resume(request["approval_id"], "approve", {}):
        resumed.append(ev)
    types = [e["type"] for e in resumed]
    assert "tool_start" in types
    assert "tool_end" in types
    assert "done" in types


async def test_command_tool_reject_flow(settings, registry, sessions):
    """run_command 被人工拒绝后不执行，Agent 收到拒绝结果继续。"""
    script = [
        ai_with_tool("执行命令", name="run_command", args={"command": "echo hi"}),
        AIMessage(content="好的，不执行"),
    ]
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, enabled=["run_command"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in runner.resume(request["approval_id"], "reject", {}):
        resumed.append(ev)
    tool_end = next(e for e in resumed if e["type"] == "tool_end")
    assert tool_end["success"] is False
    assert "拒绝" in tool_end["result"]


async def test_plan_execute_command_forced_hitl(settings, registry, sessions):
    """plan_execute（make_tools_node）对 run_command 同样强制 HITL。"""
    script = [
        AIMessage(content="步骤一"),
        ai_with_tool("执行命令", name="run_command", args={"command": "echo hi"}),
        AIMessage(content="完成"),
    ]
    runner = await _approval_runner(settings, registry, sessions, script)
    events = await collect_stream(runner, mode="plan_execute", enabled=["run_command"])
    request = next(e for e in events if e["type"] == "approval_request")
    assert request["tool_calls"][0]["name"] == "run_command"

    resumed = []
    async for ev in runner.resume(request["approval_id"], "approve", {}):
        resumed.append(ev)
    assert "done" in [e["type"] for e in resumed]


async def _two_run_command_script(final_content: str):
    """一步内两个 run_command 工具调用（同一模型响应，均强制 HITL）。"""
    return [
        AIMessage(
            content="执行两条命令",
            tool_calls=[
                {"name": "run_command", "args": {"command": "echo hi"}, "id": "call_1", "type": "tool_call"},
                {"name": "run_command", "args": {"command": "echo bye"}, "id": "call_2", "type": "tool_call"},
            ],
        ),
        AIMessage(content=final_content),
    ]


async def test_react_multi_forced_tools_batch_approval(settings, registry, sessions):
    """react 一步内多个需审批工具（两条 run_command）合并为一次批量审批，避免多 interrupt 恢复报错。"""
    settings.sandbox_backend = "local"  # 不依赖 OpenSandbox 服务端
    runner = await _approval_runner(settings, registry, sessions, await _two_run_command_script("完成"))
    events = await collect_stream(runner, enabled=["run_command"])
    request = next(e for e in events if e["type"] == "approval_request")
    assert len(request["tool_calls"]) == 2  # 一次审批覆盖两个工具调用

    resumed = []
    async for ev in runner.resume(request["approval_id"], "approve", {}):
        resumed.append(ev)
    types = [e["type"] for e in resumed]
    assert types.count("tool_start") == 2
    assert types.count("tool_end") == 2
    assert "done" in types


async def test_react_multi_forced_tools_batch_reject(settings, registry, sessions):
    """批量拒绝：两条 run_command 均不执行，Agent 收到拒绝结果继续。"""
    settings.sandbox_backend = "local"
    runner = await _approval_runner(settings, registry, sessions, await _two_run_command_script("好的，不执行"))
    events = await collect_stream(runner, enabled=["run_command"])
    request = next(e for e in events if e["type"] == "approval_request")

    resumed = []
    async for ev in runner.resume(request["approval_id"], "reject", {}):
        resumed.append(ev)
    tool_ends = [e for e in resumed if e["type"] == "tool_end"]
    assert len(tool_ends) == 2
    assert all(e["success"] is False for e in tool_ends)
    assert "done" in [e["type"] for e in resumed]
