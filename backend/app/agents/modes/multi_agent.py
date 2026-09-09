"""multi-agent 模式：Orchestrator 通过工具调用通用 worker subagent → 汇总（create_agent 实现）。

- 单个通用 subagent（create_agent，带 calculator 工具）经 convert_runnable_to_tool 包装为编排者工具；
- 任务单（TaskTicket）协议：worker 工具入参为 {"tasks": [{id, role, task, context, deps}]}，
  role 为角色标注（compute=数值计算 / analyze=逻辑分析 / 自定义角色），执行者统一为 subagent；
  一次可派多个子任务；任务单内按依赖深度分层并行执行（同层任务并发、层间串行保证前置注入），
  同一轮多个工具调用并行执行；跨批依赖任务由编排者 decide 循环下一轮再派；
- 编排者 create_agent 自带工具路由与循环，MultiAgentMiddleware 发射分派/完成事件并按 task id 归位，
  StreamEventsMiddleware 负责 thinking/message 与工具 HITL；
- 中间件顺序：StreamEventsMiddleware（HITL 审批 interrupt）在前、MultiAgentMiddleware（分派事件）在后，
  避免 interrupt 恢复时 middleware 链重跑导致 dispatch/plan 重复发射；
- 编排者与 subagent 都挂 ModelCallLimitMiddleware（轮数上限）：单轮模型调用超过 max_steps
  即抛 ModelCallLimitExceededError，运行器转为 done，防死循环；
- subagent 不持有 checkpointer（WorkerEventsMiddleware 不做 HITL 中断），审批收敛到编排者层；
- 并发护栏：settings.multi_agent_max_concurrent 限定同一时刻并行执行的子任务数，
  _worker_tool 内以 asyncio.Semaphore 控制，超限任务排队等待（防一次多工具调用并行爆炸）；
- worker 执行过程可观测：_worker_tool 用 contextvars 记录当前 task_id/worker，
  WorkerEventsMiddleware.awrap_model_call 据此把 worker 思考/输出流式转发为 agent_event 事件。
- 分派治理（_worker_tool 内 gov 共享状态，同一实例内所有任务共用）：
  1) 批内按 deps 拓扑排序执行，前置结果自动注入当前任务文本；
  2) 跨批依赖须已成功完成（done_by_id 记录），未满足返回结构化 error 条目；
  3) 正在执行中的任务防重复派发（running 集合，含并发工具调用场景）；
  4) 已完成任务按规范化文本去重复用（cache，execute-once）；
  治理失败（依赖未满足/重复派发/重复 id/依赖环）均以 {"id", "role", "error"} 条目返回，
  编排者据 error 内容修正后重新分派，不抛异常中断整批。
"""
import asyncio
import hashlib
import json
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import convert_runnable_to_tool

from app.agents.middleware.events_mw import (
    StreamEventsMiddleware,
    WorkerEventsMiddleware,
    _current_task_id,
    _current_worker,
    _worker_run_count,
)
from app.agents.middleware.multi_agent_mw import MultiAgentMiddleware


class TaskTicket(TypedDict, total=False):
    """任务单（TaskTicket）协议：单个子任务的分派规格。"""

    id: str
    role: str
    task: str
    context: str
    deps: list[str]

_WORKER_PROMPT = (
    "你是通用 Worker（subagent），负责执行编排者派发的单个子任务。"
    "需要数值计算时调用计算工具，否则直接进行逻辑分析并给出结论。"
    "若工具调用失败或返回错误，先修正参数或换一种方式重试，不要直接说工具不可用。"
    "只完成派给你的单个子任务，不做全盘汇总或最终总结，汇总由编排者负责。"
)
_ORCHESTRATOR_PROMPT = (
    "你是编排者。根据任务调用 worker（通用 subagent）分派子任务，"
    "最后整合它们的执行结果给出完整的最终答案。若某 Worker 返回错误，可调整任务措辞后重新分派。\n"
    "分派规范（任务单）：\n"
    "1. 一次调用可派多个子任务：tools 入参为 tasks 数组，每项含 id / role / task / context"
    "（role 为角色标注：compute=数值计算、analyze=逻辑分析，也可用研究员/开发者等自定义角色，"
    "用于提示 worker 的执行侧重；context 为可选的参考上下文，无则省略）；\n"
    "2. 依赖任务（前置任务的产出是它的输入）通过 deps 字段列出前置任务 id：可引用本轮批次内"
    "的任务或此前已派发完成的任务；Worker 会按依赖顺序自动执行并注入前置结果，deps 顺序与批次"
    "内位置无关。若前置任务不存在或未完成，Worker 会返回 error，此时必须先完成前置任务再派；\n"
    "3. 每个子任务必须是目标角色可独立完成的最小单元；\n"
    "3.1 一次分派：同一轮中相互独立的子任务尽量在一次工具调用中一次性派完（tasks 数组一次传多个），"
    "不要逐任务多次调用；仅当后置任务依赖前置任务结果时才分多轮派发；\n"
    "4. 任务去重：同一子任务只执行一次。若某任务（描述相同的任务）已由任一 Worker 成功完成，"
    "不得再分派给任何 Worker（含换角色），直接引用其结果；Worker 返回错误的任务不算完成，"
    "可调整措辞后重新分派。\n"
    "5. 信息不足时，通过 ask_user 统一向用户澄清，不要编造或假设缺失信息；\n"
    "6. Worker 返回 error 字段时，按错误内容修正后重新分派：依赖未满足→先派发前置任务；"
    "任务正在执行中→不要重复派发，稍后直接引用其结果；同一批次内任务 id 重复→改用唯一 id。\n"
    "汇总规范（synthesize + decide）：\n"
    "7. 最终答案由你（编排者）整合所有 Worker 的结果后直接输出。总结与汇总是你的职责，"
    "不得把总结/汇总类任务派给任何 Worker；\n"
    "8. 汇总时必须区分「Worker 已产出的结果」与「你的推断」，不要编造 Worker 没给的数据；\n"
    "9. 全部关键子任务已覆盖 → 直接输出完整最终答案；仍有缺口且该缺口可通过某 Worker 获得"
    "→ 再分派一轮补齐（tasks 中注明依赖的 task id），不要停止；缺口是用户才能提供的私有信息"
    "→ 用 ask_user 澄清后继续。"
)


def _task_key(task_text: str) -> str:
    """任务去重键：规范化（折叠空白）后取 SHA-1。

    同一任务文本（含拼入的参考上下文）无论 role 标注如何都映射到同一键，
    供 worker 共享去重表执行 execute-once。
    """
    norm = " ".join(task_text.split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _topo_order(batch: list[dict]) -> tuple[list[dict], list[dict]]:
    """对任务单批次做依赖拓扑排序，返回 (按依赖顺序的任务序列, 错误条目)。

    - id 唯一性：同一批次内任务 id 重复时，仅首个保留执行，其余记为错误条目；
    - 依赖环（含自环）：环中任务全部记为错误条目，不执行；
    - 跨批次/未完成的依赖不在此判定（调用方按完成记录校验后注入结果）。
    """
    from collections import deque

    first_ids: dict[str, dict] = {}
    dup_ids: set[str] = set()
    for t in batch:
        tid = t.get("id")
        if tid is None:
            continue
        if tid in first_ids:
            dup_ids.add(tid)
        else:
            first_ids[tid] = t

    errors: list[dict] = []
    for t in batch:
        tid = t.get("id")
        if tid is not None and tid in dup_ids and t is not first_ids[tid]:
            errors.append(
                {
                    "id": tid,
                    "error": f"同一批次内任务 id 重复：{tid}，请为每个子任务分配唯一 id（保留首个，其余忽略）",
                }
            )
    dup_skip = {e["id"] for e in errors}

    # 节点图：无 id 的任务不能作为依赖目标，按叶子处理（自动分配内部节点名）
    nodes: dict[str, dict] = {}
    auto = 0
    for i, t in enumerate(batch):
        tid = t.get("id")
        if tid is None:
            auto += 1
            tid = f"__auto_{auto}__"
        # 重复 id 只保留首个执行，非首个跳过（已在 errors 记录错误条目）
        if t.get("id") in dup_skip and t is not first_ids.get(t.get("id")):
            continue
        nodes[tid] = t

    indeg = {tid: 0 for tid in nodes}
    children = {tid: [] for tid in nodes}
    for tid, t in nodes.items():
        for d in (t.get("deps") or []):
            if d in nodes:
                children[d].append(tid)
                indeg[tid] += 1

    queue = deque(tid for tid, n in indeg.items() if n == 0)
    ordered: list[str] = []
    while queue:
        tid = queue.popleft()
        ordered.append(tid)
        for c in children[tid]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)

    for tid, n in indeg.items():
        if n > 0:
            t = nodes[tid]
            errors.append(
                {
                    "id": t.get("id"),
                    "error": f"依赖关系存在环（含自环）：{t.get('deps')}，无法安排执行顺序，请调整依赖",
                }
            )
    # 返回节点附带内部 _key（无 id 任务的自动节点名 / 原任务 id），供调用方做依赖分层与结果归位
    out = []
    for tid in ordered:
        node = dict(nodes[tid])
        node["_key"] = tid
        out.append(node)
    return out, errors


def _worker_tool(worker, name, description, semaphore=None, gov=None):
    """把子代理包装为任务单工具：入参 {"tasks": [{id, role, task, context, deps}]}。

    按依赖深度对任务单分层并行执行子任务（context 与已满足的前置结果拼入 task 文本），
    同一层内相互独立的任务并发执行（受 semaphore 并发护栏限制），层间串行保证前置注入，
    结果按 id 归位为 JSON 数组返回给编排者。

    semaphore 为并发护栏：每个子任务执行前获取令牌，同一时刻最多 semaphore 个任务并行，
    超限任务排队等待；semaphore 为 None 时不限制（不启用护栏）。

    gov 为分派治理共享状态（worker 单一实例内所有任务共用同一 dict，默认自建）：
    - gov["cache"]：任务去重表（execute-once），键为 _task_key，值为 {"worker", "result"}。
      命中（描述相同的任务已由任一 Worker 成功完成）时不重新执行，直接返回引用结果；
      仅成功结果入缓存，Worker 返回错误的任务不算完成，可被重新派发；
    - gov["running"]：正在执行中的任务键集合，防并发重复派发（同一轮多个工具调用并行时
      同描述任务只执行一次，重复派发返回结构化 error）；
    - gov["done_by_id"]：已完成任务 id → {"result", "key"}，供跨批次 deps 完成度校验与前置
      结果注入；依赖未满足的任务返回结构化 error（{"id", "role", "error"}），不中断整批。

    执行前用 contextvars 记录当前 task_id/worker，供 WorkerEventsMiddleware 流式转发
    worker 思考/输出为带 task_id 的 agent_event（前端逐步展示子代理执行过程）。
    """
    gov = gov if gov is not None else {"cache": {}, "running": set(), "done_by_id": {}}

    async def _run(tasks: dict) -> str:
        """执行任务单：tasks 为完整入参字典（{"tasks": [TaskTicket, ...]}）。

        按依赖深度对任务单分层：同一层内任务相互独立，asyncio.gather 并行执行
        （受 semaphore 并发护栏限制），层间按拓扑顺序串行（保证前置结果注入）。
        """
        batch = tasks.get("tasks") or []
        ordered, order_errors = _topo_order(batch)
        results = list(order_errors)
        executed: dict[str, str] = {}  # 本批次已完成任务 id → 结果（供批内依赖注入）

        # 依赖分层：depth = 1 + max(批内前置依赖的 depth)；同一 depth 的任务相互独立可并行。
        # deps 引用的跨批任务（已由 gov["done_by_id"] 记录）不在批内，不参与分层。
        depth: dict[str, int] = {}
        for t in ordered:
            d = 0
            for dep in (t.get("deps") or []):
                if dep in depth:
                    d = max(d, depth[dep] + 1)
            depth[t["_key"]] = d
        levels: dict[int, list[dict]] = {}
        for t in ordered:
            levels.setdefault(depth[t["_key"]], []).append(t)

        for level_idx in sorted(levels):
            level = levels[level_idx]
            pending: list[tuple[dict, str]] = []  # (任务, 拼好依赖注入的任务文本)
            for t in level:
                tid = t.get("id")
                role = t.get("role", name)
                task_text = t.get("task", "")
                if t.get("context"):
                    task_text = f"{task_text}\n参考上下文：{t['context']}"
                # 依赖完成度校验：deps 中的 id 必须在本批已完成或历史已完成，否则返回结构化 error
                unsat = [d for d in (t.get("deps") or []) if d not in executed and d not in gov["done_by_id"]]
                if unsat:
                    results.append(
                        {
                            "id": tid,
                            "role": role,
                            "error": f"依赖任务 {unsat} 尚未完成（不存在或未派发过），请先完成前置任务后再分派",
                        }
                    )
                    continue
                # 前置结果注入：批内已完成 / 历史已完成的前置任务结果拼入当前任务文本
                injections = []
                for d in (t.get("deps") or []):
                    src = executed.get(d) or (gov["done_by_id"].get(d) or {}).get("result")
                    if src is not None:
                        injections.append(f"前置任务 {d} 的结果：\n{src}")
                if injections:
                    task_text = f"{task_text}\n" + "\n".join(injections)
                key = _task_key(task_text)
                if key in gov["running"]:
                    # 正在执行中：防并发重复派发（同一轮多个 worker 工具调用并行可能派同描述任务）
                    results.append(
                        {
                            "id": tid,
                            "role": role,
                            "error": "该任务正在执行中（重复分派），请勿重复派发，稍后直接引用其结果",
                        }
                    )
                    continue
                cached = gov["cache"].get(key)
                if cached is not None:
                    # 去重命中：同描述任务已成功完成，不再执行，返回引用结果给编排者复用
                    cached_text = (
                        f"该任务已完成（由 {cached['worker']} Worker 执行），"
                        f"直接使用以下结果，无需重复执行：\n{cached['result']}"
                    )
                    results.append({"id": tid, "role": cached["worker"], "result": cached_text})
                    if tid is not None:
                        executed[tid] = cached["result"]
                    continue
                pending.append((t, task_text))

            if not pending:
                continue

            async def _exec_one(ticket, task_text):
                tid = ticket.get("id")
                token_id = _current_task_id.set(tid)
                token_worker = _current_worker.set(name)
                token_count = _worker_run_count.set(0)  # 每个任务独立计数（worker 轮数上限）
                try:
                    return await worker.ainvoke({"messages": [HumanMessage(task_text)]})
                finally:
                    _worker_run_count.reset(token_count)
                    _current_task_id.reset(token_id)
                    _current_worker.reset(token_worker)

            # 同层并行去重表：key → Future，首个任务完成后 set_result，后续同描述任务等待引用
            key_futures: dict[str, asyncio.Future] = {}

            async def _exec_with_gov(ticket, task_text):
                """并行执行单个任务并维护分派治理共享状态（执行结果入缓存/完成记录）。"""
                key = _task_key(task_text)
                fut = key_futures.get(key)
                if fut is not None:
                    # 同层并行去重：另一任务正在执行同描述任务，等待其完成后引用结果
                    res_text = await fut
                    return ticket, res_text
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                key_futures[key] = fut
                gov["running"].add(key)
                try:
                    if semaphore is not None:
                        async with semaphore:
                            result = await _exec_one(ticket, task_text)
                    else:
                        result = await _exec_one(ticket, task_text)
                    last = result["messages"][-1]
                    res_text = str(getattr(last, "content", "") or "")
                    # 仅成功结果入缓存与完成记录（失败任务由 worker 抛异常向上传播，不写入）
                    gov["cache"][key] = {"worker": name, "result": res_text}
                    tid = ticket.get("id")
                    if tid is not None:
                        gov["done_by_id"][tid] = {"result": res_text, "key": key}
                    fut.set_result(res_text)
                    return ticket, res_text
                finally:
                    gov["running"].discard(key)

            done = await asyncio.gather(*(_exec_with_gov(t, text) for t, text in pending))
            for ticket, res_text in done:
                tid = ticket.get("id")
                role = ticket.get("role", name)
                if tid is not None:
                    executed[tid] = res_text
                results.append({"id": tid, "role": role, "result": res_text})
        return json.dumps(results, ensure_ascii=False)

    return convert_runnable_to_tool(
        RunnableLambda(_run),
        name=name,
        description=description,
        arg_types={"tasks": list[TaskTicket]},
    )


def build_multi_agent_agent(llm, tools, emit, settings, checkpointer=None, harness=None):
    """构建 multi-agent 代理：编排者路由调用通用 subagent（worker）执行任务单并汇总。"""
    worker_tools = [t for t in tools if t.name == "calculator"]
    step_limit = max(1, settings.max_steps)
    # 并发护栏：同一时刻最多 multi_agent_max_concurrent 个子任务并行执行
    concurrency = max(1, getattr(settings, "multi_agent_max_concurrent", 2))
    semaphore = asyncio.Semaphore(concurrency)

    # 单一通用 subagent：数值计算调 calculator，其余直接分析；不做汇总（汇总归编排者）
    worker_agent = create_agent(
        model=llm,
        tools=worker_tools,
        system_prompt=_WORKER_PROMPT,
        middleware=[
            WorkerEventsMiddleware(emit, harness=harness, run_limit=step_limit),
            ModelCallLimitMiddleware(run_limit=step_limit, exit_behavior="error"),
        ],
        checkpointer=None,
    )

    # 分派治理共享状态（同一实例内所有任务共用）：去重缓存 + in-flight 集合 + 完成记录
    gov: dict = {"cache": {}, "running": set(), "done_by_id": {}}
    worker_tool = _worker_tool(
        worker_agent,
        "worker",
        "负责执行编排者分派的子任务：数值计算调用计算工具，逻辑分析直接给结论。"
        "入参为 tasks 任务单数组（每项含 id/role/task/context/deps），一次可派多个子任务。",
        semaphore,
        gov,
    )

    return create_agent(
        model=llm,
        tools=[worker_tool],
        system_prompt=_ORCHESTRATOR_PROMPT,
        middleware=[
            # 顺序关键：StreamEventsMiddleware 在内层之前执行 HITL 审批 interrupt。
            # LangGraph interrupt 恢复时被中断 superstep 的 middleware 链会从头重跑，
            # 若 MultiAgentMiddleware 在外层（先 emit dispatch/plan 再审批），
            # 恢复后 dispatch/plan 会重复发射（前端重复显示）。
            # 故审批（interrupt）在前、分派事件在后：dispatch/plan 在审批通过后只 emit 一次。
            StreamEventsMiddleware(emit, harness=harness),
            MultiAgentMiddleware(emit, settings=settings),
            ModelCallLimitMiddleware(run_limit=step_limit, exit_behavior="error"),
        ],
        checkpointer=checkpointer,
    )
