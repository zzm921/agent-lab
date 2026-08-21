"""plan-and-execute 模式：LangGraph StateGraph 原生编排（planner → executor ⇄ tools → replanner）。

- planner：把任务拆解为有序子步骤，发射 plan created；
- executor：对当前步骤做一次流式模型调用（thinking/message 事件），产出 tool_calls 则路由
  到 tools，否则本步完成并推进 current_step，发射 plan running/done；
- tools：复用 make_tools_node（工具事件 + HITL 审批 + 异常兜底，任一步失败写 step_failed）；
- replanner：按已完成步骤与失败重新生成计划（plan created），覆盖旧计划；
- should_replan：executor 出口条件边——有工具调用去 tools；步骤失败且重规划未超限去 replanner；
  全部步骤完成则 end，否则 continue 到下一步。
"""
from __future__ import annotations

import re
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.middleware.events_mw import stream_model_call
from app.tools.runner import make_tools_node

_PLAN_PROMPT = "你是任务规划器。把下面的用户任务拆解为 2-5 个有序子步骤，每行一个步骤，不要编号，不要解释。"
_REPLAN_PROMPT = "你是任务规划器。请根据已完成步骤与遇到的失败，重新制定剩余子步骤，每行一个步骤，不要编号，不要解释。"


class PlanState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]
    current_step: int
    past_steps: list[str]
    replans: int
    step_failed: bool


def _parse_steps(text: str) -> list[str]:
    """把规划器输出解析为步骤列表：去掉行首符号/编号与空白。"""
    lines = []
    for ln in (text or "").splitlines():
        ln = ln.strip().lstrip("-•*·")
        ln = re.sub(r"^\d+[.、)]?\s*", "", ln)
        if ln:
            lines.append(ln)
    return lines


def _latest_task(state) -> str:
    """取最近一条用户消息作为当前任务。"""
    for m in reversed(state.get("messages") or []):
        if getattr(m, "type", None) == "human":
            return m.content
    return ""


def _step_hint(state) -> str:
    """构造当前步骤的执行提示，拼入本次模型调用的 system prompt。"""
    plan = state.get("plan") or []
    if not plan:
        return ""
    idx = min(state.get("current_step") or 0, max(len(plan) - 1, 0))
    return f"当前执行计划第 {idx + 1}/{len(plan)} 步：{plan[idx]}"


def _step_failure(state) -> bool:
    """当前步骤的失败标记：新步骤（末条非 ToolMessage）重置为 False，中途（刚执行完工具）保留。"""
    msgs = state.get("messages") or []
    if msgs and getattr(msgs[-1], "type", None) == "tool":
        return bool(state.get("step_failed"))
    return False


def build_plan_execute_agent(llm, tools, emit, settings, checkpointer=None, harness=None):
    """构建 plan-and-execute 代理：planner → executor ⇄ tools → replanner。"""
    tool_list = list(tools)
    tools_node = make_tools_node(tool_list, emit, harness=harness)
    max_replans = max(1, settings.max_iterations // 2)

    async def planner(state):
        task = _latest_task(state)
        text = (await llm.ainvoke([SystemMessage(_PLAN_PROMPT), HumanMessage(task)])).content
        steps = _parse_steps(text)
        emit({"type": "plan", "steps": steps, "current_step": 0, "status": "created"})
        return {"plan": steps, "current_step": 0, "past_steps": [], "replans": 0, "step_failed": False}

    async def executor(state):
        msgs = list(state.get("messages") or [])
        base = ""
        if msgs and getattr(msgs[0], "type", None) == "system":
            base = str(msgs[0].content)
            msgs = msgs[1:]
        system_prompt = (base + "\n" + _step_hint(state)).strip()
        msg = await stream_model_call(llm, msgs, emit, tools=tool_list, system_prompt=system_prompt)
        failed = _step_failure(state)
        if getattr(msg, "tool_calls", None):
            return {"messages": [msg], "step_failed": failed}
        plan = state.get("plan") or []
        old_idx = state.get("current_step") or 0
        idx = old_idx + 1
        done = idx >= len(plan)
        past = list(state.get("past_steps") or [])
        if old_idx < len(plan):
            past.append(f"已完成第 {old_idx + 1} 步：{plan[old_idx]}")
        emit({"type": "plan", "steps": plan, "current_step": idx, "status": "done" if done else "running"})
        return {"messages": [msg], "current_step": idx, "past_steps": past, "step_failed": failed}

    async def replanner(state):
        task = _latest_task(state)
        progress = "\n".join(state.get("past_steps") or [])
        context = f"原任务：{task}\n已完成步骤：\n{progress or '（无）'}"
        text = (await llm.ainvoke([SystemMessage(_REPLAN_PROMPT), HumanMessage(context)])).content
        steps = _parse_steps(text)
        emit({"type": "plan", "steps": steps, "current_step": 0, "status": "created"})
        return {"plan": steps, "current_step": 0, "replans": (state.get("replans") or 0) + 1}

    def should_replan(state) -> str:
        msgs = state.get("messages") or []
        if msgs and getattr(msgs[-1], "tool_calls", None):
            return "tools"
        plan = state.get("plan") or []
        idx = state.get("current_step") or 0
        if not plan or idx >= len(plan):
            return "end"
        if state.get("step_failed") and (state.get("replans") or 0) < max_replans:
            return "replan"
        return "continue"

    builder = StateGraph(PlanState)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("tools", tools_node)
    builder.add_node("replanner", replanner)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges(
        "executor",
        should_replan,
        {
            "tools": "tools",
            "continue": "executor",
            "replan": "replanner",
            "end": END,
        },
    )
    builder.add_edge("tools", "executor")
    builder.add_edge("replanner", "executor")
    return builder.compile(checkpointer=checkpointer)
