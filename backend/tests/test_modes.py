"""四种推理模式测试：使用 FakeChatModel 脚本驱动确定性执行。"""
import json

from langchain_core.messages import AIMessage

from app.agents.modes.multi_agent import _task_key, _worker_tool
from app.agents.modes.plan_execute import _parse_plan_output
from app.agents.runner import AgentRunner
from app.llm.fake_model import FakeChatModel
from tests.conftest import ai_with_tool, collect_stream, make_settings


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
        AIMessage(content='{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}, {"desc": "步骤二", "deps": []}]}'),
        AIMessage(content="已执行步骤一"),
        AIMessage(content="已执行步骤二"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="plan_execute")
    plan_events = [e for e in events if e["type"] == "plan"]
    assert plan_events and plan_events[0]["status"] == "created"
    assert plan_events[-1]["status"] == "done"
    first = plan_events[0]["items"]
    assert [t["desc"] for t in first] == ["步骤一", "步骤二"]
    assert all(t["status"] == "pending" for t in first)
    # 逐项推进：running 事件中当前项置 running（状态机 pending→running→done），前序项已 done
    running = [e for e in plan_events if e["status"] == "running"]
    assert running and running[0]["items"][0]["status"] == "running"
    assert any(t["status"] == "done" for t in running[-1]["items"])
    assert all(t["status"] == "done" for t in plan_events[-1]["items"])
    assert any(e["type"] == "message" for e in events)


async def test_plan_execute_replan(settings, registry, sessions):
    # 第 1 步成功、第 2 步调用计算器触发除零失败 → 该步标 failed → replanner 重写剩余待办
    # （保留已完成的 t1）→ 继续执行直到全部 done
    script = [
        AIMessage(content='{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}, {"desc": "步骤二", "deps": []}]}'),
        AIMessage(content="完成步骤一"),
        ai_with_tool("尝试计算", args={"expression": "1/0"}),
        AIMessage(content="该步失败，无法继续"),
        AIMessage(content='{"direct": false, "tasks": [{"desc": "新步骤二", "deps": []}]}'),
        AIMessage(content="执行新步骤二"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="plan_execute", enabled=["calculator"])
    plan_events = [e for e in events if e["type"] == "plan"]
    assert len([e for e in plan_events if e["status"] == "created"]) >= 2  # 初次计划 + 重规划
    assert plan_events[-1]["status"] == "done"
    assert any(e["type"] == "tool_end" and not e["success"] for e in events)
    # 失败发生后：该失败事件里 t1 保持 done（已完成项不被推翻）
    failed_ev = next(e for e in plan_events if any(t["status"] == "failed" for t in e["items"]))
    assert any(t["id"] == "t1" and t["status"] == "done" for t in failed_ev["items"])
    # 重规划后保留已完成项，最终全部 done
    assert any(t["id"] == "t1" and t["status"] == "done" for t in plan_events[-1]["items"])
    assert all(t["status"] == "done" for t in plan_events[-1]["items"])


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
    # 评审以流式自由文本输出：不通过（无 PASS 标记）→ 修订 → 通过（【PASS】）→ 结束
    script = [
        AIMessage(content="草稿答案"),
        AIMessage(content="需要补充细节"),
        AIMessage(content="修订后的完整答案"),
        AIMessage(content="【PASS】答案已足够好"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="reflection")
    types = [e["type"] for e in events]
    assert "reflect" in types
    assert "revise" in types
    assert "critique" in types  # 评审文本流式下发
    reflect_events = [e for e in events if e["type"] == "reflect"]
    assert reflect_events[0]["stage"] == "draft"
    assert any(e["type"] == "message" for e in events)
    critique_full = "".join(e.get("delta", "") for e in events if e["type"] == "critique")
    assert "需要补充细节" in critique_full and "【PASS】" in critique_full


async def test_reflection_text_critique_fallback(settings, registry, sessions):
    # 评审以自由文本流式输出：无 PASS 标记不通过 → 修订 → 【PASS】通过结束
    script = [
        AIMessage(content="草稿答案"),
        AIMessage(content="需要补充细节"),
        AIMessage(content="修订后的完整答案"),
        AIMessage(content="【PASS】无需修改"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="reflection")
    assert any(e["type"] == "revise" for e in events)
    assert any(e["type"] == "done" for e in events)
    critique_full = "".join(e.get("delta", "") for e in events if e["type"] == "critique")
    assert "无需修改" in critique_full


async def test_reflection_max_iter_bound(settings, registry, sessions):
    # 评审始终不通过 → 由 max_iter 兜底强制结束（防止死循环）
    settings.max_iterations = 2
    script = [
        AIMessage(content="草稿"),
        AIMessage(content="仍需修改"),
        AIMessage(content="修订一"),
        AIMessage(content="仍需修改"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="reflection")
    critique_full = "".join(e.get("delta", "") for e in events if e["type"] == "critique")
    assert critique_full.count("仍需修改") == 2  # 恰好评审 max_iter 次后强制结束
    assert any(e["type"] == "done" for e in events)


async def test_reflection_tool_in_draft_phase(settings, registry, sessions):
    # 草稿阶段模型先调用工具（计算器），拿到结果后再产出草稿 → 评审通过结束
    script = [
        ai_with_tool("需要计算", args={"expression": "1+1"}),
        AIMessage(content="草稿答案：计算结果是 2"),
        AIMessage(content="无"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="reflection", enabled=["calculator"])
    types = [e["type"] for e in events]
    assert "tool_start" in types and "tool_end" in types
    tool_end = next(e for e in events if e["type"] == "tool_end")
    assert tool_end["success"] is True and "2" in tool_end["result"]
    reflect_events = [e for e in events if e["type"] == "reflect"]
    assert reflect_events[0]["stage"] == "draft"
    full = "".join(e.get("delta", "") for e in events if e["type"] == "message")
    assert "计算结果是 2" in full
    assert any(e["type"] == "done" for e in events)


async def test_reflection_tool_in_revise_phase(settings, registry, sessions):
    # 评审未通过 → 修订阶段模型调用工具补充信息后产出修订稿 → 再评审通过结束
    script = [
        AIMessage(content="草稿答案"),
        AIMessage(content="需要补充计算"),
        ai_with_tool("需要计算", args={"expression": "2+2"}),
        AIMessage(content="修订稿答案：结果是 4"),
        AIMessage(content="无"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="reflection", enabled=["calculator"])
    types = [e["type"] for e in events]
    assert "tool_start" in types and "tool_end" in types
    reflect_events = [e for e in events if e["type"] == "reflect"]
    assert reflect_events[0]["stage"] == "draft"
    assert len([e for e in events if e["type"] == "revise"]) >= 1
    revise_full = "".join(e.get("delta", "") for e in events if e["type"] == "revise")
    assert "结果是 4" in revise_full
    assert any(e["type"] == "done" for e in events)


async def test_reflection_step_limit(settings, registry, sessions):
    # 草稿阶段模型反复请求工具（从不产出草稿）→ 达到轮数上限（max_steps）强制结束
    settings.max_steps = 3
    script = [
        ai_with_tool("计算 1", args={"expression": "1+1"}, cid="c1"),
        ai_with_tool("计算 2", args={"expression": "2+2"}, cid="c2"),
        ai_with_tool("计算 3", args={"expression": "3+3"}, cid="c3"),
        ai_with_tool("计算 4", args={"expression": "4+4"}, cid="c4"),
        AIMessage(content="草稿"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="reflection", enabled=["calculator"])
    done = next(e for e in events if e["type"] == "done")
    assert "轮数上限" in done["summary"]
    assert len([e for e in events if e["type"] == "tool_end" and e["success"]]) == 3  # 前 3 轮执行，第 4 轮被拦截


async def test_plan_execute_step_limit(settings, registry, sessions):
    # 单步内模型反复请求工具（从不完成该步）→ 达到轮数上限（max_steps）强制结束
    settings.max_steps = 3
    script = [
        AIMessage(content='{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}]}'),
        ai_with_tool("计算 1", args={"expression": "1+1"}, cid="c1"),
        ai_with_tool("计算 2", args={"expression": "2+2"}, cid="c2"),
        ai_with_tool("计算 3", args={"expression": "3+3"}, cid="c3"),
        ai_with_tool("计算 4", args={"expression": "4+4"}, cid="c4"),
        AIMessage(content="已执行步骤一"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="plan_execute", enabled=["calculator"])
    done = next(e for e in events if e["type"] == "done")
    assert "轮数上限" in done["summary"]
    assert len([e for e in events if e["type"] == "tool_end" and e["success"]]) == 3


async def test_plan_execute_no_plan(settings, registry, sessions):
    # 简单问候无需计划：规划器输出 {"direct": true} → 不生成 todo、不发 plan 事件，直接流式回复
    script = [
        AIMessage(content='{"direct": true}'),
        AIMessage(content="你好，有什么可以帮您？"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="plan_execute")
    assert not any(e["type"] == "plan" for e in events), "简单问候不应生成执行计划"
    full = "".join(e.get("delta", "") for e in events if e["type"] == "message")
    assert "你好" in full
    assert any(e["type"] == "done" for e in events)


async def test_react_step_limit(settings, registry, sessions):
    # ReAct 模式模型反复请求工具（从不给出最终答案）→ 达到轮数上限（max_steps）强制结束
    settings.max_steps = 3
    script = [
        ai_with_tool("计算 1", args={"expression": "1+1"}, cid="c1"),
        ai_with_tool("计算 2", args={"expression": "2+2"}, cid="c2"),
        ai_with_tool("计算 3", args={"expression": "3+3"}, cid="c3"),
        ai_with_tool("计算 4", args={"expression": "4+4"}, cid="c4"),
        AIMessage(content="最终答案"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="react", enabled=["calculator"])
    done = next(e for e in events if e["type"] == "done")
    assert "轮数上限" in done["summary"]
    assert len([e for e in events if e["type"] == "tool_end" and e["success"]]) == 3
    full = "".join(e.get("delta", "") for e in events if e["type"] == "message")
    assert "最终答案" not in full  # 未走到最终答案已被拦截


async def test_multi_agent(settings, registry, sessions):
    # 任务单协议：编排者以 tasks 数组分派 worker subagent，结果按 task id 归位（role 为角色标注）
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool(
            "派发计算",
            name="worker",
            args={"tasks": [{"id": "t1", "role": "compute", "task": "计算 1+1"}]},
        ),
        AIMessage(content="计算结论：完成计算"),
        ai_with_tool(
            "派发分析",
            name="worker",
            args={"tasks": [{"id": "t2", "role": "analyze", "task": "分析可行性"}]},
        ),
        AIMessage(content="分析结论：任务可行"),
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    types = [e["type"] for e in events]
    assert "agent_event" in types
    agent_events = [e for e in events if e["type"] == "agent_event"]
    # 按 task id 归位：每个任务的 dispatch/done 成对且携带 task_id
    assert any(e["worker"] == "worker" and e["status"] == "done" and e.get("task_id") == "t1" for e in agent_events)
    assert any(e["worker"] == "worker" and e["status"] == "done" and e.get("task_id") == "t2" for e in agent_events)
    # 任务单 → todo 视图：plan 事件 items 带 assignee（执行角色）
    plan_events = [e for e in events if e["type"] == "plan"]
    assert plan_events and plan_events[0]["status"] == "created"
    assert any(t.get("assignee") == "compute" for t in plan_events[0]["items"])
    assert "message" in types
    assert "done" in types


async def test_multi_agent_task_tickets(settings, registry, sessions):
    # 一次分派多个任务单：worker 并行执行，结果按 id 归位到对应 task
    # （Fake 剧本共享模型下并行任务消费剧本输出的顺序不确定，断言放宽为集合匹配）
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool(
            "派发两个计算",
            name="worker",
            args={
                "tasks": [
                    {"id": "t1", "role": "compute", "task": "计算 1+1"},
                    {"id": "t2", "role": "compute", "task": "计算 2+2"},
                ]
            },
        ),
        AIMessage(content="结果1：2"),
        AIMessage(content="结果2：4"),
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    agent_events = [e for e in events if e["type"] == "agent_event"]
    done_by_id = {
        e.get("task_id"): e.get("result")
        for e in agent_events
        if e["status"] == "done" and e.get("task_id") in ("t1", "t2")
    }
    assert set(done_by_id) == {"t1", "t2"}, "两个任务都应按 id 归位"
    assert done_by_id["t1"] != done_by_id["t2"], "各自拿到独立结果"
    assert {"结果1：2", "结果2：4"} == set(done_by_id.values()), "结果覆盖剧本的两个输出"
    plan_events = [e for e in events if e["type"] == "plan"]
    assert len(plan_events[0]["items"]) == 2  # 两个任务单都进入 todo 视图
    assert plan_events[-1]["status"] == "done"


async def test_multi_agent_task_limit(settings, registry, sessions):
    # 单轮任务单超过 MAX_TASKS(8) 被截断：只派发前 8 个（防拆解爆炸）
    tasks = [{"id": f"t{i}", "role": "compute", "task": f"任务 {i}"} for i in range(1, 11)]
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool("派发超量任务", name="worker", args={"tasks": tasks}),
        *[AIMessage(content=f"完成 {i}") for i in range(1, 9)],
        AIMessage(content="汇总完成"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    plan_events = [e for e in events if e["type"] == "plan"]
    assert len(plan_events[0]["items"]) == 8  # 截断为 MAX_TASKS
    assert not any(t["id"] in ("t9", "t10") for t in plan_events[0]["items"])
    agent_events = [e for e in events if e["type"] == "agent_event"]
    assert len([e for e in agent_events if e["status"] == "dispatch" and e.get("task_id")]) == 8


async def test_multi_agent_worker_running_stream(settings, registry, sessions):
    # worker 执行过程流式转发：dispatch 后出现 agent_event(status=running, stage, delta, task_id)，
    # worker 输出逐段增量下发（前端按任务就地累积），done 事件携带完整结果
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool(
            "派发计算",
            name="worker",
            args={"tasks": [{"id": "t1", "role": "compute", "task": "计算 1+1"}]},
        ),
        AIMessage(content="计算结论：完成计算"),
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    running = [e for e in events if e["type"] == "agent_event" and e.get("status") == "running"]
    msg_deltas = [e for e in running if e.get("stage") == "message" and e.get("task_id") == "t1"]
    assert msg_deltas and all(e.get("delta") for e in msg_deltas)
    assert "".join(e.get("delta", "") for e in msg_deltas) == "计算结论：完成计算"
    done = [e for e in events if e["type"] == "agent_event" and e.get("status") == "done" and e.get("task_id") == "t1"]
    assert done and done[0]["result"] == "计算结论：完成计算"


async def test_multi_agent_worker_tool_scope(settings, registry, sessions):
    # worker 内部工具调用仍保留在流中（telemetry/评测 must_call 统计），但打上 scope=worker + task_id，
    # 前端据此隐藏 subagent 内部工具卡片
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool(
            "派发计算",
            name="worker",
            args={"tasks": [{"id": "t1", "role": "compute", "task": "计算 2+2"}]},
        ),
        ai_with_tool("计算", name="calculator", args={"expression": "2+2"}),  # worker 内部工具调用
        AIMessage(content="结果是 4"),
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    calc_start = [e for e in events if e["type"] == "tool_start" and e.get("tool") == "calculator"]
    assert calc_start, "worker 内部工具事件应保留在流中"
    assert all(e.get("scope") == "worker" and e.get("task_id") == "t1" for e in calc_start)
    calc_end = [e for e in events if e["type"] == "tool_end" and e.get("tool") == "calculator"]
    assert calc_end and all(e.get("scope") == "worker" for e in calc_end)


async def test_multi_agent_worker_failure_status(settings, registry, sessions):
    # worker 执行失败（轮数超限/协议外返回）：agent_event 标记 failed 并带错误文本，
    # 前端子代理卡片显示失败状态（工具卡片隐藏后失败信号不丢）
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool(
            "派发计算",
            name="worker",
            args={"tasks": [{"id": "t1", "role": "compute", "task": "计算 2+2"}]},
        ),
        # worker 5 次模型调用全返回工具调用 → 第 6 次触发轮数上限异常，worker 失败
        *[ai_with_tool(f"计算{i}", name="calculator", args={"expression": "1+1"}) for i in range(6)],
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    failed = [e for e in events if e["type"] == "agent_event" and e.get("status") == "failed" and e.get("task_id") == "t1"]
    assert failed, "worker 失败时应发射 failed 状态的 agent_event"
    assert failed[0].get("result"), "failed 事件应携带错误文本"
    # worker 工具执行本身也标记失败（tool_end success=false）
    tool_ends = [e for e in events if e["type"] == "tool_end" and e.get("tool") == "worker"]
    assert tool_ends and not tool_ends[-1].get("success")


async def test_multi_agent_concurrency_limit(settings):
    # 并发护栏：同一时刻并行执行的子代理数不超过 multi_agent_max_concurrent（防爆炸），
    # 超限任务排队等待；任务单内仍顺序执行保证结果归位确定性
    import asyncio

    from app.agents.modes.multi_agent import _worker_tool

    active = 0
    peak = 0

    class FakeWorker:
        async def ainvoke(self, inputs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return {"messages": [AIMessage(content="完成")]}

    tool = _worker_tool(FakeWorker(), "worker", "desc", asyncio.Semaphore(2))
    # 模拟编排者同轮并行发起 4 个 worker 工具调用（每个 1 个任务）；
    # 任务文本互不相同，避免命中任务去重缓存（execute-once）导致 worker 不执行
    await asyncio.gather(*[tool.arun({"tasks": [{"id": f"t{i}", "task": f"任务 {i}"}]}) for i in range(4)])
    assert peak == 2, f"并发上限应生效：峰值并发 {peak}，期望 2"


async def test_multi_agent_task_dedup(settings, registry, sessions):
    # execute-once 任务去重：worker 已完成的任务（相同描述）再次分派 →
    # 不重新执行，agent_event 返回"已由 worker Worker 执行"的引用结果，
    # 且不再产生 t2 的流式执行事件（worker 未被再次调用）
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool(
            "派发分析",
            name="worker",
            args={"tasks": [{"id": "t1", "role": "analyze", "task": "分析可行性"}]},
        ),
        AIMessage(content="分析结论：可行"),
        ai_with_tool(
            "派发计算",
            name="worker",
            args={"tasks": [{"id": "t2", "role": "compute", "task": "分析可行性"}]},
        ),
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    done_t2 = [
        e for e in events if e["type"] == "agent_event" and e.get("task_id") == "t2" and e.get("status") == "done"
    ]
    assert done_t2, "t2 分派应返回 done"
    assert "由 worker" in done_t2[0]["result"], "重复任务应返回引用结果而非重新执行"
    # t2 未真正执行：无 t2 的 running（thinking/message）事件
    assert not any(
        e["type"] == "agent_event" and e.get("task_id") == "t2" and e.get("status") == "running"
        for e in events
    )


async def test_multi_agent_task_dedup_failure_not_cached(settings, registry, sessions):
    # 失败任务不入去重缓存：worker 失败（轮数超限）后，编排者再派相同描述任务 → 重新执行
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool(
            "派发计算",
            name="worker",
            args={"tasks": [{"id": "t1", "role": "compute", "task": "计算 2+2"}]},
        ),
        # worker1 5 次模型调用全返回工具调用 → 第 6 次触发轮数上限异常，t1 失败
        *[ai_with_tool(f"计算{i}", name="calculator", args={"expression": "1+1"}) for i in range(6)],
        ai_with_tool(
            "再次派发",
            name="worker",
            args={"tasks": [{"id": "t2", "role": "compute", "task": "计算 2+2"}]},
        ),
        AIMessage(content="完成 t2"),
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    assert any(e["type"] == "agent_event" and e.get("task_id") == "t1" and e.get("status") == "failed" for e in events)
    done_t2 = [
        e for e in events if e["type"] == "agent_event" and e.get("status") == "done" and e.get("task_id") == "t2"
    ]
    assert done_t2 and done_t2[0]["result"] == "完成 t2", "失败任务不缓存，再次派发应重新执行而非引用"
    # t2 真正执行：存在 t2 的流式执行过程（worker 被调用）
    assert any(
        e["type"] == "agent_event" and e.get("task_id") == "t2" and e.get("status") == "running"
        for e in events
    )


async def test_multi_agent_approval_resume_no_duplicate_dispatch(settings, registry, sessions):
    # 回归：HITL 审批恢复时 LangGraph 会重跑被中断 superstep 的 middleware 链，
    # 若分派事件在审批 interrupt 前 emit，恢复后会重复发射（前端重复显示 dispatch）。
    # 修复后 dispatch/plan 延后到审批通过后 emit：每个 task 恰好一次 dispatch、一次 done。
    script = [
        ai_with_tool(
            "派发计算",
            name="worker",
            args={"tasks": [{"id": "t1", "role": "compute", "task": "计算 1+1"}]},
            cid="c1",
        ),
        AIMessage(content="计算结论：完成计算"),
        AIMessage(content="最终汇总答案"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    first = await collect_stream(
        runner, mode="multi_agent", enabled=["calculator"], approval_policy="always"
    )
    request = next(e for e in first if e["type"] == "approval_request")
    assert request["tool_calls"][0]["name"] == "worker"
    # 审批前不产生分派/计划事件（延后到审批通过后 emit，避免恢复重跑重复）
    assert not any(e["type"] == "agent_event" and e.get("status") == "dispatch" for e in first)
    assert not any(e["type"] == "plan" for e in first)

    resumed = []
    async for ev in runner.resume(request["approval_id"], "approve", {}):
        resumed.append(ev)
    merged = first + resumed
    dispatches = [e for e in merged if e["type"] == "agent_event" and e.get("status") == "dispatch"]
    assert len(dispatches) == 1, f"dispatch 应恰好一次，实际 {len(dispatches)} 次"
    assert dispatches[0]["task_id"] == "t1"
    done_events = [e for e in merged if e["type"] == "agent_event" and e.get("status") == "done"]
    assert len(done_events) == 1 and done_events[0]["task_id"] == "t1"
    plan_created = [e for e in merged if e["type"] == "plan" and e["status"] == "created"]
    assert len(plan_created) == 1, "plan created 应恰好一次"
    assert any(e["type"] == "done" for e in merged)


async def test_plan_execute_todo_running(settings, registry, sessions):
    # 状态机流转：executor 进入时当前项置 running（pending→running→done），
    # 完成后置 done，最终 plan done
    script = [
        AIMessage(content='{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}, {"desc": "步骤二", "deps": []}]}'),
        AIMessage(content="已执行步骤一"),
        AIMessage(content="已执行步骤二"),
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="plan_execute")
    plan_events = [e for e in events if e["type"] == "plan"]
    running = [e for e in plan_events if e["status"] == "running"]
    assert running
    # 首个 running 事件：第一项已置 running（工具循环期间保持 running，而非直接 done）
    assert any(t["status"] == "running" for t in running[0]["items"])
    assert plan_events[-1]["status"] == "done"
    assert all(t["status"] == "done" for t in plan_events[-1]["items"])


async def test_plan_execute_task_limit(settings, registry, sessions):
    # 规划器输出超过 8 项被截断（护栏，防拆解爆炸）
    tasks = ",".join(f'{{"desc": "步骤{i}", "deps": []}}' for i in range(1, 13))
    script = [
        AIMessage(content=f'{{"direct": false, "tasks": [{tasks}]}}'),
        *[AIMessage(content=f"完成步骤{i}") for i in range(1, 6)],  # max_steps=5，只执行前 5 步
    ]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="plan_execute")
    plan_events = [e for e in events if e["type"] == "plan"]
    assert plan_events and len(plan_events[0]["items"]) == 8  # 12 项被截断为 8


# ---- 规划产物 schema 化：JSON 主路径 + 文本回退 ----
def test_plan_json_parse():
    # 正常 JSON：id 由后端分配，deps 支持数字序号与 tN 两种写法
    direct, items = _parse_plan_output(
        '{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}, {"desc": "步骤二", "deps": [1]}, {"desc": "步骤三", "deps": ["t1"]}]}'
    )
    assert direct is False
    assert [t["id"] for t in items] == ["t1", "t2", "t3"]
    assert [t["desc"] for t in items] == ["步骤一", "步骤二", "步骤三"]
    assert items[0]["deps"] == [] and items[1]["deps"] == ["t1"] and items[2]["deps"] == ["t1"]
    assert all(t["status"] == "pending" for t in items)


def test_plan_json_direct():
    direct, items = _parse_plan_output('{"direct": true}')
    assert direct is True and items == []


def test_plan_json_markdown_wrapped():
    # markdown 代码块包裹：剥包裹后仍按 JSON 主路径解析
    text = '```json\n{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}]}\n```'
    direct, items = _parse_plan_output(text)
    assert direct is False and len(items) == 1 and items[0]["desc"] == "步骤一"


def test_plan_json_fallback_text():
    # 非 JSON 输出 → 回退文本行解析（原协议）
    direct, items = _parse_plan_output("步骤一\n步骤二")
    assert direct is False
    assert [t["desc"] for t in items] == ["步骤一", "步骤二"]


def test_plan_json_invalid_deps_fallback():
    # 依赖引用不存在的任务 → 回退文本解析（整行作为单个任务，不静默产出坏依赖）
    text = '{"direct": false, "tasks": [{"desc": "步骤一", "deps": ["t9"]}]}'
    direct, items = _parse_plan_output(text)
    assert direct is False
    assert len(items) == 1  # 回退路径把整行当一条文本任务
    assert items[0]["deps"] == []


def test_plan_json_truncated_bracket():
    # 模型输出被 token 截断缺失闭合括号（tasks 数组缺 "]" / 整体缺 "]}"）→ 补全解析成功
    text = '{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}, {"desc": "步骤二", "deps": ["t1"]}'
    direct, items = _parse_plan_output(text)
    assert direct is False
    assert [t["desc"] for t in items] == ["步骤一", "步骤二"]
    assert items[1]["deps"] == ["t1"]


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
    # 瞬时故障注入类型（如 http_500）：工具执行前仍需 HITL 审批（审批永不跳过）；
    # 批准后进入工具层透明重试（发 tool_retry 事件）；重试耗尽后返回结构化错误给模型（Agent 层思考后重试）
    script = [
        ai_with_tool("触发 500", args={"expression": "1+1"}, cid="c1"),
        AIMessage(content="结束"),
    ]
    settings.tool_retry_base_delay = 0.001
    settings.tool_retry_max_delay = 0.002
    runner = await _runner_with(registry, sessions, settings, script)
    runner.harness.set_fault("calculator", "http_500")
    events = await collect_stream(runner, mode="react", enabled=["calculator"], approval_policy="always")
    request = next(e for e in events if e["type"] == "approval_request")  # 故障工具也先审批
    assert request["tool_calls"][0]["name"] == "calculator"
    assert not any(e["type"] == "tool_retry" for e in events)  # 批准前未执行工具

    resumed = []
    async for ev in runner.resume(request["approval_id"], "approve", {}):
        resumed.append(ev)
    retry_events = [e for e in resumed if e["type"] == "tool_retry"]
    assert retry_events, "批准后应进入工具层透明重试"
    assert retry_events[0]["tool"] == "calculator"
    assert retry_events[0]["max"] == 3
    end = next(e for e in resumed if e["type"] == "tool_end" and not e["success"])
    assert "故障注入" in end["result"]
    assert "瞬时错误" in end["result"]  # 结构化错误给模型


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


async def test_rag_enabled_auto_retrieves(settings, registry, sessions):
    """RAG 作为独立检索阶段：启用 rag 后自动召回并产 retrieve 事件，不产生工具调用。"""
    registry.rag_manager.ingest_all(["LangGraph 基于 StateGraph 构建有状态、多步骤的 AI Agent。"])
    script = [AIMessage(content="（基于知识库回答）")]
    runner = await _runner_with(registry, sessions, settings, script)
    events = await collect_stream(runner, mode="react", enabled=["rag"], rag_scheme="naive", rag_enabled=True)
    retrieve = next((e for e in events if e["type"] == "retrieve"), None)
    assert retrieve is not None
    assert retrieve["scheme"] == "naive"
    assert retrieve["query"] == "测试任务"
    assert retrieve["hits"] and "LangGraph" in retrieve["hits"][0]["text"]
    # RAG 不再进入工具集：无工具调用事件
    assert not any(e["type"] == "tool_start" for e in events)
    assert any(e["type"] == "message" for e in events)


async def test_rag_advanced_rewrite_before_retrieve(settings, registry, sessions):
    """advanced：页面上先输出 Query 重写结果（rewrite 事件），再输出知识库检索命中（retrieve 事件）。"""
    registry.rag_manager.ingest_all(["公司报销制度要求出差结束后尽快提交发票。"])
    runner = await _runner_with(registry, sessions, settings, [AIMessage(content="ok")])
    events = await collect_stream(runner, mode="react", enabled=["rag"], rag_scheme="advanced", rag_enabled=True)
    rewrite = next((e for e in events if e["type"] == "rewrite"), None)
    retrieve = next((e for e in events if e["type"] == "retrieve"), None)
    assert rewrite is not None and rewrite["rewrites"], "advanced 应输出 Query 重写结果"
    assert retrieve is not None
    assert events.index(rewrite) < events.index(retrieve), "重写结果应输出在知识库检索之前"


async def test_rag_enabled_default_scheme_fallback(settings, registry, sessions):
    """未指定 rag_scheme 时回退默认方案（naive），仍自动检索。"""
    registry.rag_manager.ingest_all(["ReAct 模式由 思考-行动-观察 循环组成。"])
    runner = await _runner_with(registry, sessions, settings, [AIMessage(content="ok")])
    events = await collect_stream(runner, mode="react", enabled=["rag"], rag_enabled=True)
    retrieve = next((e for e in events if e["type"] == "retrieve"), None)
    assert retrieve is not None and retrieve["scheme"] == "naive"


async def test_rag_disabled_no_retrieval(settings, registry, sessions):
    """知识库检索被前端开关关闭（rag_enabled=False）时不产 retrieve 事件、不注入上下文。"""
    registry.rag_manager.ingest_all(["LangGraph 基于 StateGraph 构建有状态、多步骤的 AI Agent。"])
    runner = await _runner_with(registry, sessions, settings, [AIMessage(content="ok")])
    events = await collect_stream(runner, mode="react", enabled=["rag"], rag_scheme="naive", rag_enabled=False)
    assert not any(e["type"] == "retrieve" for e in events)


def test_augment_query_injects_context():
    """检索命中注入用户消息；无命中时原样返回。"""
    ctx = {"name": "朴素 RAG", "hits": [{"score": 0.9, "text": "LangGraph 基于 StateGraph 构建。"}]}
    out = AgentRunner._augment_query("LangGraph 是什么", ctx)
    assert "【知识库检索结果（朴素 RAG）】" in out
    assert "LangGraph 基于 StateGraph" in out
    assert "请优先基于以上检索内容回答" in out
    assert AgentRunner._augment_query("x", {"name": "naive", "hits": []}) == "x"
    assert AgentRunner._augment_query("x", None) == "x"


# ---- multi_agent 分派治理：批内拓扑排序 + 前置结果注入 + 完成度校验 + in-flight/重复 id/环 ----
class _Recorder:
    """记录每次 worker 输入，返回可预测结果（供断言注入内容）。"""

    def __init__(self):
        self.calls: list[str] = []

    async def ainvoke(self, inputs):
        text = inputs["messages"][0].content
        self.calls.append(text)
        head = text.splitlines()[0][:4] if text.splitlines() else ""
        return {"messages": [AIMessage(content=f"完成:{head}")]}


async def test_multi_agent_worker_deps_topo_and_injection():
    # 批内依赖：t2 依赖 t1、t3 依赖 t2 → 按拓扑顺序执行，前置结果自动注入任务文本
    rec = _Recorder()
    tool = _worker_tool(rec, "worker", "desc")
    out = json.loads(
        await tool.arun(
            {
                "tasks": [
                    {"id": "t1", "task": "任务A"},
                    {"id": "t2", "task": "任务B", "deps": ["t1"]},
                    {"id": "t3", "task": "任务C", "deps": ["t2"]},
                ]
            }
        )
    )
    assert rec.calls[0].startswith("任务A"), "依赖任务应先执行（t1 先于 t2/t3）"
    assert "前置任务 t1 的结果" in rec.calls[1] and "完成:任务A" in rec.calls[1]
    assert "前置任务 t2 的结果" in rec.calls[2] and "完成:任务B" in rec.calls[2]
    assert all(not r.get("error") for r in out)
    assert {r["id"]: r["result"] for r in out}.keys() == {"t1", "t2", "t3"}


async def test_multi_agent_worker_deps_unsatisfied():
    # 依赖前置校验：deps 引用未完成/不存在的任务 → 返回结构化 error，worker 不执行
    rec = _Recorder()
    tool = _worker_tool(rec, "worker", "desc")
    out = json.loads(await tool.arun({"tasks": [{"id": "t2", "task": "任务B", "deps": ["t1"]}]}))
    assert not rec.calls, "依赖未满足的任务不应执行"
    assert out[0]["error"] and "依赖任务" in out[0]["error"]


async def test_multi_agent_worker_deps_cross_batch():
    # 跨批依赖：上一批已完成的 t1，本批 t2 deps=["t1"] → 完成度校验通过并注入前置结果
    rec = _Recorder()
    tool = _worker_tool(rec, "worker", "desc")
    await tool.arun({"tasks": [{"id": "t1", "task": "任务A"}]})
    rec.calls.clear()
    out = json.loads(await tool.arun({"tasks": [{"id": "t2", "task": "任务B", "deps": ["t1"]}]}))
    assert not out[0].get("error"), "跨批已完成依赖应校验通过"
    assert "前置任务 t1 的结果" in rec.calls[0] and "完成:任务A" in rec.calls[0]


async def test_multi_agent_worker_cycle_error():
    # 依赖环：t1↔t2 相互依赖 → 均返回结构化 error，worker 不执行
    rec = _Recorder()
    tool = _worker_tool(rec, "worker", "desc")
    out = json.loads(
        await tool.arun(
            {"tasks": [{"id": "t1", "task": "A", "deps": ["t2"]}, {"id": "t2", "task": "B", "deps": ["t1"]}]}
        )
    )
    assert not rec.calls
    assert len(out) == 2 and all("环" in r.get("error", "") for r in out)


async def test_multi_agent_worker_self_loop_error():
    # 自环：任务依赖自身 → 结构化 error，不执行
    rec = _Recorder()
    tool = _worker_tool(rec, "worker", "desc")
    out = json.loads(await tool.arun({"tasks": [{"id": "t1", "task": "A", "deps": ["t1"]}]}))
    assert not rec.calls
    assert len(out) == 1 and "环" in out[0].get("error", "")


async def test_multi_agent_worker_dup_id():
    # 同一批次内 id 重复：仅首个执行，其余返回结构化 error
    rec = _Recorder()
    tool = _worker_tool(rec, "worker", "desc")
    out = json.loads(
        await tool.arun({"tasks": [{"id": "t1", "task": "任务A"}, {"id": "t1", "task": "任务B"}]})
    )
    errors = [r for r in out if r.get("error")]
    assert len(errors) == 1 and "重复" in errors[0]["error"]
    assert len(rec.calls) == 1, "重复 id 任务只执行首个"


async def test_multi_agent_worker_inflight_dedup():
    # in-flight 去重：任务正在执行中（running 集合命中）→ 返回结构化 error，不执行
    gov = {"cache": {}, "running": set(), "done_by_id": {}}
    rec = _Recorder()
    tool = _worker_tool(rec, "worker", "desc", gov=gov)
    gov["running"].add(_task_key("任务A"))
    out = json.loads(await tool.arun({"tasks": [{"id": "t1", "task": "任务A"}]}))
    assert not rec.calls, "正在执行中的任务不应重复执行"
    assert "正在执行中" in out[0]["error"]


async def test_multi_agent_worker_error_status(settings, registry, sessions):
    # 分派治理失败（依赖未满足）经中间件归位为 failed 状态并携带错误文本
    llm = FakeChatModel()
    llm.script = [
        ai_with_tool(
            "派发依赖任务",
            name="worker",
            args={"tasks": [{"id": "t1", "role": "compute", "task": "任务B", "deps": ["tX"]}]},
        ),
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    failed = [
        e for e in events if e["type"] == "agent_event" and e.get("task_id") == "t1" and e.get("status") == "failed"
    ]
    assert failed and "依赖任务" in failed[0]["result"]
    # 计划视图同样标记失败
    plan_events = [e for e in events if e["type"] == "plan"]
    assert plan_events[-1]["items"][0]["status"] == "failed"
