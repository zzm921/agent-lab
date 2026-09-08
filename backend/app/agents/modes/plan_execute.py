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
from app.core.events import emit_text
from app.tools.ask_user import is_ask_reply_msg
from app.tools.runner import make_tools_node

_PLAN_PROMPT = (
    "你是任务规划器。先判断用户任务是否需要拆解为多个子任务逐步执行。\n"
    "需要计划的场景：多步骤任务、需要逐步调研/计算的任务、包含多个子问题的复杂任务。\n"
    "无需计划的场景：简单问候、一句话即可回答的简单问答。\n"
    "如果无需计划，第一行只输出 DIRECT，不要输出任何计划内容。\n"
    "如果需要计划，把用户任务拆解为 2-5 个有序子任务，每行一个子任务，不要编号，不要解释。\n"
    "拆解粒度要适度：优先合并同类工作，避免拆得过细；每个子任务应独立可执行、可验证，"
    "宁可少而整，不要多而碎。\n"
    "每行格式：子任务描述；若该子任务依赖前面的子任务，在行首写 [行号]（多个依赖用逗号分隔）。\n"
    "若任务缺少执行必需的关键信息（仅用户能提供的私有信息），"
    "应把「向用户确认这些关键信息」列为第一步，后续步骤再基于已确认的信息继续拆解；"
    "不要拆解依赖缺失信息的步骤，不要假设或编造用户未提供的信息。"
)
_REPLAN_PROMPT = (
    "你是任务规划器。请根据已完成步骤与遇到的失败，重新制定剩余子任务，每行一个子任务，"
    "格式同前（行首 [行号] 标注对剩余列表内前置行的依赖）。只列出尚未完成的子任务，不要重复已完成的工作。"
)
# executor 每步执行引导（框架化，不绑定具体任务形态）：
# 事实类信息→工具核实；私有信息→一次性澄清；按是否最终步骤约束输出；严禁编造。
_EXECUTE_PROMPT = (
    "执行本步骤前，先判断完成本步骤所需的信息是否已具备，按以下框架处理："
    "1. 事实类信息（可通过外部工具核实的内容）：主动调用相应工具核实后再输出结论，不得依赖模型记忆臆测；"
    "2. 私有信息（仅用户掌握、工具无法获取的内容）：通过一次 ask_user 调用（questions 一次性列出全部缺失项，"
    "不要逐项往返提问）礼貌澄清；若用户跳过或信息仍不完整，则基于已有信息输出结论，"
    "并如实标注假设与缺口，不要重复追问同一问题；"
    "3. 输出约束：严格按系统提示标明的是否为最终步骤控制输出——非最终步骤只输出本步简短结论"
    "（关键数据/决定，2-4 行），不得输出完整最终答案，不得重复此前已输出的内容；"
    "最终步骤才输出完整的最终答案。"
    "全程不得编造或臆测未核实的事实。"
)


class PlanState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    todo: list[dict]   # 带状态机的子任务清单：[{id, desc, deps, status}]，status ∈ pending/dispatched/running/done/failed
    past_steps: list[str]
    replans: int
    step_failed: bool
    steps: int       # 累计模型调用/工具回合数（轮数上限判定）
    stopped: str     # 结束原因："max_steps" 表示达到轮数上限


def _parse_todo(text: str, start: int = 0) -> list[dict]:
    """把规划器输出解析为 todo 列表：每行一个子任务，行首可选 [行号]（或 [tN]）标注依赖。

    id 自动分配为 t1/t2/…；start 为重规划时保留已完成项后的编号偏移。
    """
    items = []
    for ln in (text or "").splitlines():
        ln = ln.strip().lstrip("-•*·")
        if not ln:
            continue
        deps = []
        m = re.match(r"^\[([0-9tT][0-9,，tT]*)\]\s*(.*)$", ln)
        if m:
            refs = [x.strip() for x in re.split(r"[,，]", m.group(1)) if x.strip()]
            for r in refs:
                if re.match(r"^t\d+$", r, re.IGNORECASE):
                    deps.append(r.lower())
                else:
                    deps.append(f"t{start + int(r)}")
            ln = m.group(2).strip()
        ln = re.sub(r"^\d+[.、)]?\s*", "", ln)
        if not ln:
            continue
        items.append({"id": f"t{start + len(items) + 1}", "desc": ln, "deps": deps, "status": "pending"})
    return items


def _is_direct(text: str) -> bool:
    """判断规划器是否判定「无需计划」（输出 DIRECT 标记）：此时不生成 todo、不发 plan 事件，直接回复。"""
    t = (text or "").strip()
    return t.upper().startswith("DIRECT") or t.startswith("无需计划") or t.startswith("不需要计划")


def _reason_text(chunk) -> str:
    """提取模型思考内容（DashScope reasoning_content），取数路径与 events_mw 保持一致。"""
    extra = getattr(chunk, "additional_kwargs", None) or {}
    reasoning = extra.get("reasoning_content")
    if reasoning:
        return reasoning if isinstance(reasoning, str) else str(reasoning)
    reasoning = getattr(chunk, "reasoning_content", None)
    return reasoning if reasoning else ""


async def _plan_stream(llm, system_text: str, user_text: str, emit) -> str:
    """规划器流式调用：逐 token 生成并实时下发 thinking 事件（思考过程），
    规划文本合并返回（不注入 message 事件，避免与顶部 todo 固定区重复展示）。"""
    chunks = []
    async for chunk in llm.bind().astream([SystemMessage(system_text), HumanMessage(user_text)]):
        chunks.append(chunk)
        reason = _reason_text(chunk)
        if reason:
            emit_text(emit, "thinking", reason)
    if not chunks:
        raise RuntimeError("模型流式调用未返回任何内容")
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    return str(getattr(merged, "content", "") or "")


def _next_task(state) -> dict | None:
    """取下一个可执行子任务：优先「依赖均已 done 的 pending 项」，否则回退第一个 pending（防死锁）。"""
    todo = state.get("todo") or []
    done_ids = {t["id"] for t in todo if t["status"] == "done"}
    for t in todo:
        if t["status"] == "pending" and all(d in done_ids for d in (t.get("deps") or [])):
            return t
    for t in todo:
        if t["status"] == "pending":
            return t
    return None


def _emit_todo(emit, todo: list[dict], status: str) -> None:
    """发射 plan 事件：携带逐项状态，current_step 指向下一个未完成项位置。"""
    idx = next((i for i, t in enumerate(todo) if t["status"] != "done"), len(todo))
    emit({"type": "plan", "items": todo, "current_step": idx, "status": status})


def _latest_task(state) -> str:
    """取最近一条用户消息作为当前任务。"""
    for m in reversed(state.get("messages") or []):
        if getattr(m, "type", None) == "human":
            return m.content
    return ""


def _step_hint(state) -> str:
    """构造当前子任务的执行提示，拼入本次模型调用的 system prompt。

    区分最终步骤与非最终步骤：中间步骤只输出简短结论，完整方案留到最后一步整合，
    避免模型每步都重复输出完整方案（多次循环计算）。
    """
    todo = state.get("todo") or []
    cur = _next_task(state)
    if not cur:
        return ""
    idx = next((i for i, t in enumerate(todo) if t["id"] == cur["id"]), 0)
    is_final = idx + 1 >= len(todo)
    hint = f"当前执行任务 t{idx + 1}（{idx + 1}/{len(todo)}）：{cur['desc']}"
    if is_final:
        return hint + "\n这是最后一步：请整合此前所有步骤的结论，输出完整、最终的方案。"
    return hint + "\n注意：这不是最终步骤，请只输出本步骤的简短结论（关键数据/决定，2-4 行），完整方案留待最后一步整合，不要重复此前已输出的内容。"


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
        text = await _plan_stream(planner_llm, _PLAN_PROMPT, task, emit)
        if _is_direct(text):
            # 无需计划：不生成 todo、不发 plan 事件，由 executor 直接流式回复
            return {"todo": [], "past_steps": [], "replans": 0, "step_failed": False}
        todo = _parse_todo(text)
        _emit_todo(emit, todo, "created")
        return {"todo": todo, "past_steps": [], "replans": 0, "step_failed": False}

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
        # 本子任务完成：推进 todo 状态（成功置 done，失败置 failed），再发射更新事件
        todo = [dict(t) for t in (state.get("todo") or [])]
        cur = _next_task(state)
        past = list(state.get("past_steps") or [])
        if cur is not None:
            for t in todo:
                if t["id"] == cur["id"]:
                    t["status"] = "failed" if failed else "done"
                    break
            past.append(f"第 {cur['id']} 步{'失败' if failed else '完成'}：{cur['desc']}")
        all_done = bool(todo) and all(t["status"] == "done" for t in todo)
        if todo:  # 无需计划（todo 为空）时不发射 plan 事件，直接回复收尾
            _emit_todo(emit, todo, "done" if all_done else "running")
        return {"messages": [msg], "todo": todo, "past_steps": past, "step_failed": failed, "steps": steps}

    async def replanner(state):
        task = _latest_task(state)
        todo = list(state.get("todo") or [])
        done_items = [dict(t) for t in todo if t["status"] == "done"]
        remain = [t["desc"] for t in todo if t["status"] != "done"]
        progress = "\n".join(state.get("past_steps") or [])
        parts = [f"原任务：{task}", f"已完成步骤：\n{progress or '（无）'}"]
        if remain:
            parts.append("剩余待办（可重写/合并/拆分）：\n" + "\n".join(remain))
        text = await _plan_stream(planner_llm, _REPLAN_PROMPT, "\n".join(parts), emit)
        new_items = _parse_todo(text, start=len(done_items))
        new_todo = done_items + new_items
        _emit_todo(emit, new_todo, "created")
        return {"todo": new_todo, "replans": (state.get("replans") or 0) + 1}

    def should_replan(state) -> str:
        # 达到轮数上限 → 直接结束（不再执行工具/继续规划）
        if state.get("stopped") == "max_steps":
            return "end"
        msgs = state.get("messages") or []
        if msgs and getattr(msgs[-1], "tool_calls", None):
            return "tools"
        todo = state.get("todo") or []
        has_pending = any(t["status"] in ("pending", "dispatched", "running") for t in todo)
        has_failed = any(t["status"] == "failed" for t in todo)
        # 失败项存在且允许重规划 → 进 replanner 重写剩余待办（保留已完成项）
        if has_failed and (state.get("replans") or 0) < max_replans:
            return "replan"
        if not has_pending:
            return "end"
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
