"""plan-and-execute 模式：LangGraph StateGraph 原生编排（planner → executor ⇄ tools → replanner）。

- planner：把任务拆解为有序子步骤，发射 plan created；
- executor：对当前步骤做一次流式模型调用（thinking/message 事件），产出 tool_calls 则路由
  到 tools，否则本步完成并推进 current_step，发射 plan running/done；
- tools：复用 make_tools_node（工具事件 + HITL 审批 + 异常兜底，任一步失败写 step_failed）；
- replanner：按已完成步骤与失败重新生成计划（plan created），覆盖旧计划；
- should_replan：executor 出口条件边——有工具调用去 tools；步骤失败且重规划未超限去 replanner；
  全部步骤完成则 end，否则 continue 到下一步；
- 轮数上限（max_steps）：executor 累计模型调用/工具回合数（steps），超过即置 stopped=max_steps
  不再调用模型并结束，防「单步内反复请求工具」的死循环。
"""
from __future__ import annotations

import re
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.middleware.events_mw import resolve_guards, stream_model_call
from app.tools.ask_user import is_ask_reply_msg
from app.tools.runner import make_tools_node

_PLAN_PROMPT = (
    "你是任务规划器。把下面的用户任务拆解为 2-5 个有序子步骤，每行一个步骤，不要编号，不要解释。"
    "若任务缺少执行必需的关键信息（如出发地、预算、偏好、时间、同行人员等），"
    "应把「向用户确认这些关键信息」列为第一步，后续步骤再基于已确认的信息继续拆解；"
    "不要拆解依赖缺失信息的步骤，不要假设或编造用户未提供的信息。"
)
_REPLAN_PROMPT = "你是任务规划器。请根据已完成步骤与遇到的失败，重新制定剩余子步骤，每行一个步骤，不要编号，不要解释。"
# executor 每步执行引导：不确定事实优先工具核实；仅用户私有信息缺失时追问澄清；严禁编造。
_EXECUTE_PROMPT = (
    "执行本步骤前先判断所需信息是否已具备。若存在可通过工具核实的事实性信息"
    "（如目的地景点、花期、天气、交通票价、开放时间等），应主动调用 web_search 等工具核实后再给出建议；"
    "若缺少的是只有用户能提供的信息（如出发地、预算、偏好、同行人员等），"
    "应通过一次 ask_user 调用（questions 列表一次性列出所有缺失项，不要逐个提问多次往返）"
    "礼貌地向用户澄清、引导其补充，不要编造或臆测未确认的事实。"
    "若澄清回复为「跳过未回答」或信息仍不完整，直接基于已有信息给出建议"
    "并如实标注假设与所缺信息，不要重复追问同一问题。"
)


class PlanState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]
    current_step: int
    past_steps: list[str]
    replans: int
    step_failed: bool
    steps: int       # 累计模型调用/工具回合数（轮数上限判定）
    stopped: str     # 结束原因："max_steps" 表示达到轮数上限


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


def build_plan_execute_agent(planner_llm, executor_llm, tools, emit, settings, checkpointer=None, harness=None):
    """构建 plan-and-execute 代理：planner → executor ⇄ tools → replanner。

    planner_llm：任务规划/重规划（低随机、输出精炼）；executor_llm：分步执行（流式）。
    """
    tool_list = list(tools)
    tools_node = make_tools_node(tool_list, emit, harness=harness)
    max_replans = max(1, settings.max_iterations // 2)
    max_steps = max(1, settings.max_steps)

    async def planner(state):
        task = _latest_task(state)
        text = (await planner_llm.ainvoke([SystemMessage(_PLAN_PROMPT), HumanMessage(task)])).content
        steps = _parse_steps(text)
        emit({"type": "plan", "steps": steps, "current_step": 0, "status": "created"})
        return {"plan": steps, "current_step": 0, "past_steps": [], "replans": 0, "step_failed": False}

    async def executor(state):
        msgs = list(state.get("messages") or [])
        # 新回合（最近一条为 user）重置轮数计数，避免多轮会话累计导致过早停止
        fresh = bool(msgs) and getattr(msgs[-1], "type", None) == "human"
        # 澄清回复后的接续调用（最近一条是 ask_user 回复）不计入轮数：提问/回答不消耗执行预算
        ask_round = (not fresh) and bool(msgs) and is_ask_reply_msg(msgs[-1])
        steps = (0 if fresh else (state.get("steps") or 0)) + (0 if ask_round else 1)
        if steps > max_steps:
            return {"steps": steps, "stopped": "max_steps"}
        base = ""
        if msgs and getattr(msgs[0], "type", None) == "system":
            base = str(msgs[0].content)
            msgs = msgs[1:]
        system_prompt = (base + "\n" + _step_hint(state) + "\n" + _EXECUTE_PROMPT).strip()
        msg = await stream_model_call(executor_llm, msgs, emit, tools=tool_list, system_prompt=system_prompt, guards=resolve_guards(settings))
        failed = _step_failure(state)
        if getattr(msg, "tool_calls", None):
            return {"messages": [msg], "step_failed": failed, "steps": steps}
        plan = state.get("plan") or []
        old_idx = state.get("current_step") or 0
        idx = old_idx + 1
        done = idx >= len(plan)
        past = list(state.get("past_steps") or [])
        if old_idx < len(plan):
            past.append(f"已完成第 {old_idx + 1} 步：{plan[old_idx]}")
        emit({"type": "plan", "steps": plan, "current_step": idx, "status": "done" if done else "running"})
        return {"messages": [msg], "current_step": idx, "past_steps": past, "step_failed": failed, "steps": steps}

    async def replanner(state):
        task = _latest_task(state)
        progress = "\n".join(state.get("past_steps") or [])
        context = f"原任务：{task}\n已完成步骤：\n{progress or '（无）'}"
        text = (await planner_llm.ainvoke([SystemMessage(_REPLAN_PROMPT), HumanMessage(context)])).content
        steps = _parse_steps(text)
        emit({"type": "plan", "steps": steps, "current_step": 0, "status": "created"})
        return {"plan": steps, "current_step": 0, "replans": (state.get("replans") or 0) + 1}

    def should_replan(state) -> str:
        # 达到轮数上限 → 直接结束（不再执行工具/继续规划）
        if state.get("stopped") == "max_steps":
            return "end"
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
