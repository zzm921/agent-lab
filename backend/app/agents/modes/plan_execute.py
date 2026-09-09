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

import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.middleware.events_mw import resolve_guards, stream_model_call
from app.tools.ask_user import is_ask_reply_msg
from app.tools.runner import make_tools_node

_PLAN_PROMPT = (
    "你是任务规划器。先判断用户任务是否需要拆解为多个子任务逐步执行，再按结构化 schema 输出。\n"
    "需要计划的场景：多步骤任务、需要逐步调研/计算的任务、包含多个子问题的复杂任务。\n"
    "无需计划的场景：简单问候、一句话即可回答的简单问答（direct=true，不输出 tasks）。\n"
    "约束：\n"
    "1. tasks 为 2-5 项，按执行顺序排列；\n"
    "2. desc 是独立可执行、可验证的最小单元；优先合并同类工作，避免拆得过细，宁可少而整，不要多而碎；\n"
    "3. deps 表示对前面任务的依赖：依赖第 1 项写 [1] 或 [\"t1\"]，无依赖写 []，只写实际存在的任务序号；\n"
    "4. 若任务缺少执行必需的关键信息（仅用户能提供的私有信息），把「向用户确认这些关键信息」列为第一个子任务，"
    "后续步骤再基于已确认的信息继续拆解；不要拆解依赖缺失信息的步骤，不要假设或编造用户未提供的信息。"
)
_REPLAN_PROMPT = (
    "你是任务规划器。根据已完成步骤与遇到的失败，重新制定剩余子任务，按结构化 schema 输出。\n"
    "只列出尚未完成的子任务，不要重复已完成的工作；deps 的序号基于本列表内的位置"
    "（依赖第 1 项写 [1] 或 [\"t1\"]，只写实际存在的任务序号）。"
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


class PlanTask(TypedDict, total=False):
    """结构化规划子任务：desc 必填，deps 可选（默认 []）。"""
    desc: str
    deps: list[int | str]


class PlanOutput(TypedDict, total=False):
    """规划器结构化输出：direct=true 表示无需计划；tasks 为子任务清单。"""
    direct: bool
    tasks: list[PlanTask]


class PlanState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    todo: list[dict]   # 带状态机的子任务清单：[{id, desc, deps, status}]，status ∈ pending/dispatched/running/done/failed
    past_steps: list[str]
    replans: int
    step_failed: bool
    steps: int       # 累计模型调用/工具回合数（轮数上限判定）
    stopped: str     # 结束原因："max_steps" 表示达到轮数上限


def _parse_todo_text(text: str, start: int = 0, max_items: int = 8) -> list[dict]:
    """把规划器文本输出解析为 todo 列表（回退路径）：每行一个子任务，行首可选 [行号]（或 [tN]）标注依赖。

    id 自动分配为 t1/t2/…；start 为重规划时保留已完成项后的编号偏移。
    max_items 为单轮拆解数量上限（护栏，防拆解爆炸）：超限项直接丢弃。
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
    return items[:max_items]


def _extract_json(text: str) -> dict | None:
    """宽容提取规划器输出中的 JSON：剥 ```json 代码块包裹与前后杂质，失败返回 None。

    模型偶发截断闭合括号（tasks 数组或整体缺 "]" / "}]"）：失败时按后缀补全再试。
    """
    t = (text or "").strip()
    lines = t.splitlines()
    if lines and lines[0].strip().startswith("```"):
        t = "\n".join(lines[1:])
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3].rstrip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(t[start : end + 1])
        if isinstance(payload, dict):
            return payload
    except (ValueError, TypeError):
        pass
    # 截断补全：先关 tasks 数组（"]"）再关外层对象（"}"）——按缺哪个补哪个逐级尝试：
    # "]}": 末尾为任务对象 "}"（数组与外层对象都未闭合）；"}"：数组已闭合仅缺外层对象
    for suffix in ("]}", "}"):
        try:
            payload = json.loads(t[start : end + 1] + suffix)
            if isinstance(payload, dict):
                return payload
        except (ValueError, TypeError):
            continue
    return None


def _coerce_deps(deps, start: int) -> list[str] | None:
    """把 JSON 的 deps 字段（[1, "t2"]）规范化为后端 id 列表（["t1", "t2"]）；非法返回 None。"""
    if not isinstance(deps, list):
        return None
    out = []
    for d in deps:
        if isinstance(d, bool) or not isinstance(d, (int, str)):
            return None
        s = str(d).strip().lower()
        if re.match(r"^t\d+$", s):
            out.append(s)
        elif re.match(r"^\d+$", s):
            out.append(f"t{start + int(s)}")
        else:
            return None
    return out


def _parse_plan_json(payload: dict, start: int = 0) -> list[dict] | None:
    """JSON schema → todo 列表；字段缺失/类型错误/依赖引用非法（含自环）时返回 None（触发回退）。

    主路径：id 由后端分配（t{start+i+1}），LLM 不提供 id，避免重复/乱序。
    """
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return None
    items = []
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            return None
        desc = str(t.get("desc") or "").strip()
        if not desc:
            return None
        deps = _coerce_deps(t.get("deps") or [], start)
        if deps is None:
            return None
        tid = f"t{start + i + 1}"
        if tid in deps:  # 自环
            return None
        items.append({"id": tid, "desc": desc, "deps": deps, "status": "pending"})
    valid_ids = {it["id"] for it in items}
    for it in items:
        if any(d not in valid_ids for d in it["deps"]):  # 引用不存在的任务
            return None
    return items


def _parse_plan_output(raw: dict | str, start: int = 0, max_items: int = 8) -> tuple[bool, list[dict]]:
    """解析规划器输出：返回 (direct, todo items)。

    - raw 为 dict（with_structured_output 结构化产出）→ 直接校验，结构非法视为空计划；
    - raw 为 str（回退文本）→ 宽容提取 JSON，缺失/非法再回退文本行解析（原协议）；
    - direct=True：无需计划（调用方不生成 todo、不发 plan 事件）。
    """
    if isinstance(raw, dict):
        if raw.get("direct") is True:
            return True, []
        items = _parse_plan_json(raw, start)
        return False, (items or [])[:max_items]
    payload = _extract_json(raw)
    if payload is not None:
        if payload.get("direct") is True:
            return True, []
        items = _parse_plan_json(payload, start)
        if items is not None:
            return False, items[:max_items]
    return _is_direct(raw), _parse_todo_text(raw, start, max_items)


def _is_direct(text: str) -> bool:
    """判断规划器是否判定「无需计划」（输出 DIRECT 标记）：此时不生成 todo、不发 plan 事件，直接回复。"""
    t = (text or "").strip()
    return t.upper().startswith("DIRECT") or t.startswith("无需计划") or t.startswith("不需要计划")


async def _plan_once(llm, system_text: str, user_text: str) -> dict | str:
    """规划器一次性调用：优先结构化输出，失败回退自由文本。

    - 主路径：with_structured_output(json_mode) 注入 schema 约束模型输出，返回 dict（PlanOutput）；
    - 回退路径：结构化解析失败（OutputParserException 等）时走普通 ainvoke 返回文本，由
      _parse_plan_output 走宽容 JSON 提取 + 文本行解析兜底；
    - 企业级实践：规划是短小的结构化决策产物（2-5 项 todo），一次 ainvoke 返回
      比逐 token 流式更确定、更省成本；前端在规划期间显示「规划中」占位卡片，
      规划完成后由调用方一次性 emit plan created（携带完整 todo 清单）。
    """
    msgs = [SystemMessage(system_text), HumanMessage(user_text)]
    try:
        resp = await llm.with_structured_output(PlanOutput, method="json_mode").ainvoke(msgs)
        if isinstance(resp, dict):
            return resp
        return str(getattr(resp, "content", "") or "")
    except Exception:
        fallback = await llm.ainvoke(msgs)
        return str(getattr(fallback, "content", "") or "")


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
    """发射 plan 事件：携带逐项状态，current_step 指向下一个未完成项位置。

    注意：事件 payload 必须快照（深拷贝 items），因为调用方后续会继续原地修改 todo
    dict（置 running/done），若发射引用则队列中已入队的事件会被后续修改污染。
    """
    idx = next((i for i, t in enumerate(todo) if t["status"] != "done"), len(todo))
    emit({"type": "plan", "items": [dict(t) for t in todo], "current_step": idx, "status": status})


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
    cur = next((t for t in todo if t["status"] == "running"), None) or _next_task(state)
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
        text = await _plan_once(planner_llm, _PLAN_PROMPT, task)
        direct, todo = _parse_plan_output(text)
        if direct:
            # 无需计划：不生成 todo、不发 plan 事件，由 executor 直接流式回复
            return {"todo": [], "past_steps": [], "replans": 0, "step_failed": False}
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
        # 进入执行：当前 pending 项先置 running（状态机 pending→running→done/failed）。
        # 工具循环期间保持 running；再次进入（工具结果返回）时 running 项优先承接，不重复发射。
        todo = [dict(t) for t in (state.get("todo") or [])]
        cur = next((t for t in todo if t["status"] == "running"), None) or _next_task(state)
        if cur is not None and cur["status"] == "pending":
            for t in todo:
                if t["id"] == cur["id"]:
                    t["status"] = "running"
                    break
            if todo:
                _emit_todo(emit, todo, "running")
        msg = await stream_model_call(executor_llm, msgs, emit, tools=tool_list, system_prompt=system_prompt, guards=resolve_guards(settings))
        failed = _step_failure(state)
        if getattr(msg, "tool_calls", None):
            return {"messages": [msg], "todo": todo, "step_failed": failed, "steps": steps}
        # 本子任务完成：推进 todo 状态（成功置 done，失败置 failed），再发射更新事件
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
        text = await _plan_once(planner_llm, _REPLAN_PROMPT, "\n".join(parts))
        _, new_items = _parse_plan_output(text, start=len(done_items))
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
