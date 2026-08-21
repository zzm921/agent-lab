"""Agent 统一运行器：构建模式图、SSE 流式运行、HITL 中断/恢复。"""
import asyncio
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from app.agents.modes.multi_agent import build_multi_agent_agent
from app.agents.modes.plan_execute import build_plan_execute_agent
from app.agents.modes.react import build_react_agent
from app.agents.modes.reflection import build_reflection_agent
from app.agents.tools_builder import build_tools

STRATEGY_PROMPTS = {
    "standard": "你是专业的 AI 助手，请直接、准确地回答用户的问题。",
    "few_shot": "你是专业的 AI 助手。请参照示例格式组织回答：\n示例：用户问'计算 2+2'→'结果为 4，计算过程：2+2=4。'\n请按此清晰结构作答。",
    "cot": "你是专业的 AI 助手。回答前请逐步思考（chain-of-thought），先列出推理步骤再给出结论。",
}


class AgentHarness:
    """持有会话配置，支持 stream（含中断暂停）与 resume（批准/拒绝/修改）。

    说明：LangGraph 1.x 中 ainvoke 遇 interrupt 不抛异常，而是返回暂停状态；
    通过 aget_state(config).tasks 中的 Interrupt 判断是否需要审批；
    恢复时用同一 checkpointer 重建图，使事件流入本次连接的队列。
    """

    def __init__(self, settings, llm, registry, session_store):
        self.settings = settings
        self.llm = llm
        self.registry = registry
        self.sessions = session_store
        self._configs: dict[str, dict] = {}
        self._specs: dict[str, tuple] = {}
        self._approvals: dict[str, str] = {}
        # thread_id → [工具调用计数]：同一轮内跨 resume 累计，避免多次审批时少算
        self._tool_counts: dict[str, list] = {}
        # session_id → 正在运行的后台图任务：供 /api/stop 取消，及时停止执行以节省 token
        self._tasks: dict[str, asyncio.Task] = {}

    def _build_graph(self, mode, tools, emit):
        checkpointer = self.sessions.checkpointer
        if mode == "react":
            return build_react_agent(self.llm, tools, emit, self.settings, checkpointer)
        if mode == "plan_execute":
            return build_plan_execute_agent(self.llm, tools, emit, self.settings, checkpointer)
        if mode == "reflection":
            return build_reflection_agent(self.llm, tools, emit, self.settings, checkpointer)
        if mode == "multi_agent":
            return build_multi_agent_agent(self.llm, tools, emit, self.settings, checkpointer)
        raise ValueError(f"未知模式：{mode}")

    def _config(self, session_id, approval_policy, strategy):
        return {
            "configurable": {
                "thread_id": session_id,
                "approval_policy": approval_policy,
                "prompt_strategy": strategy,
            },
            "recursion_limit": self.settings.max_iterations * 10 + 50,
        }

    async def _make_inputs(self, graph, config, message, strategy):
        snap = await graph.aget_state(config)
        msgs = []
        if snap is not None and snap.values:
            msgs = list(snap.values.get("messages", []))
        if not msgs:
            msgs.append(SystemMessage(content=STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["standard"])))
        msgs.append(HumanMessage(content=message))
        return {"messages": msgs}

    @staticmethod
    def _new_channel():
        return asyncio.Queue(), [0]

    @staticmethod
    def _make_emit(queue, tool_count):
        def emit(data):
            if data.get("type") == "tool_start":
                tool_count[0] += 1
            queue.put_nowait(data)

        return emit

    async def stream(self, session_id, message, mode, enabled, prompt_strategy, approval_policy):
        """启动新一轮对话并产出 SSE 事件；遇 HITL 中断产出 approval_request 后暂停。"""
        queue, tool_count = self._new_channel()
        self._tool_counts[session_id] = tool_count  # 每轮重置；该轮后续 resume 复用累计
        emit = self._make_emit(queue, tool_count)

        tools = build_tools(self.registry, enabled, session_id, emit)
        try:
            graph = self._build_graph(mode, tools, emit)
        except Exception as exc:
            yield {"type": "error", "message": f"无法构建模式 {mode}", "detail": str(exc)}
            return
        config = self._config(session_id, approval_policy, prompt_strategy)
        self._configs[session_id] = config
        self._specs[session_id] = (mode, tools)
        yield {"type": "meta", "session_id": session_id, "mode": mode, "capabilities": enabled}
        inputs = await self._make_inputs(graph, config, message, prompt_strategy)
        async for ev in self._run_graph(session_id, graph, config, inputs, queue, tool_count):
            yield ev

    async def resume(self, approval_id, decision, modified_args):
        """通过审批号找到暂停的会话，恢复图执行并继续产出 SSE 事件。"""
        session_id = self._approvals.get(approval_id)
        spec = self._specs.get(session_id)
        config = self._configs.get(session_id)
        if session_id is None or spec is None or config is None:
            yield {"type": "error", "message": "审批会话不存在或已过期，请重新发送", "detail": ""}
            return

        queue = asyncio.Queue()
        tool_count = self._tool_counts.get(session_id) or [0]  # 复用该轮累计计数
        emit = self._make_emit(queue, tool_count)
        mode, tools = spec
        graph = self._build_graph(mode, tools, emit)
        command = Command(resume={"action": decision, "modified_args": modified_args or {}})
        async for ev in self._run_graph(session_id, graph, config, command, queue, tool_count):
            yield ev

    def stop(self, session_id: str) -> None:
        """取消指定会话正在运行的后台图任务（配合前端停止按钮，及时停止执行节省 token）。"""
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()

    async def _run_graph(self, session_id, graph, config, inputs, queue, tool_count):
        """以后台任务驱动图执行，同时实时排空事件队列，实现真正的流式下发。

        任务登记到 self._tasks，供 stop() 取消；客户端中断（SSE 连接关闭）时也在 finally 中取消。
        """
        sentinel = object()
        outcome: dict = {}

        async def _run():
            try:
                await graph.ainvoke(inputs, config)
                outcome["ok"] = True
            except asyncio.CancelledError:
                outcome["cancelled"] = True
            except Exception as exc:  # noqa: BLE001
                outcome["ok"] = False
                outcome["exc"] = exc
            finally:
                queue.put_nowait(sentinel)

        task = asyncio.create_task(_run())
        self._tasks[session_id] = task
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item
        finally:
            self._tasks.pop(session_id, None)
            # 客户端中断（SSE 连接关闭）时取消后台图任务，避免遗留运行
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if outcome.get("cancelled"):
            yield {"type": "done", "summary": "已停止执行", "stats": {"tool_calls": tool_count[0]}}
            return
        if not outcome.get("ok"):
            yield {"type": "error", "message": "Agent 运行失败", "detail": str(outcome.get("exc"))}
            return

        snap = await graph.aget_state(config)
        pending = []
        for run_task in snap.tasks or ():
            pending.extend(run_task.interrupts or ())
        if pending:
            intr = pending[0]
            payload = getattr(intr, "value", {}) or {}
            approval_id = uuid.uuid4().hex
            self._approvals[approval_id] = config["configurable"]["thread_id"]
            yield {
                "type": "approval_request",
                "approval_id": approval_id,
                "tool_calls": payload.get("tool_calls", []),
            }
            return
        yield {"type": "done", "summary": "本次任务处理完成", "stats": {"tool_calls": tool_count[0]}}
