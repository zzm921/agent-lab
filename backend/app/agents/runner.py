"""Agent 统一运行器：构建模式图、SSE 流式运行、HITL 中断/恢复。

职责：编排循环（构建模式图、事件队列驱动流式下发、resume 恢复执行）；
护栏相关（审批策略、资源上限、止损、统计）统一委托 AgentHarness（app.agents.harness）。
"""
from __future__ import annotations

import asyncio
import uuid

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from app.agents.harness import AgentHarness
from app.agents.modes.multi_agent import build_multi_agent_agent
from app.agents.modes.plan_execute import build_plan_execute_agent
from app.agents.modes.react import build_react_agent
from app.agents.modes.reflection import build_reflection_agent
from app.agents.tools_builder import build_tools
from app.llm.service import LLMService

STRATEGY_PROMPTS = {
    "standard": "你是专业的 AI 助手，请直接、准确地回答用户的问题。",
    "few_shot": "你是专业的 AI 助手。请参照示例格式组织回答：\n示例：用户问'计算 2+2'→'结果为 4，计算过程：2+2=4。'\n请按此清晰结构作答。",
    "cot": "你是专业的 AI 助手。回答前请逐步思考（chain-of-thought），先列出推理步骤再给出结论。",
}

# 工具使用规范：追加到首轮 system prompt，让模型在工具失败后优先换参数/换方式重试，
# 而不是直接告诉用户「工具不可用」（与熔断机制配合：相同参数重复失败才拦截，换参重试放行）。
TOOL_RETRY_HINT = (
    "工具使用规范：当工具调用失败或返回错误时，应修正参数或换一种调用方式重试（不要用相同参数反复重试），"
    "不要直接告诉用户工具不可用；仅在换参数后仍连续多次失败时，才可改用其它工具或如实向用户说明。"
)


class AgentRunner:
    """持有会话配置，支持 stream（含中断暂停）与 resume（批准/拒绝/修改）。

    说明：LangGraph 1.x 中 ainvoke 遇 interrupt 不抛异常，而是返回暂停状态；
    通过 aget_state(config).tasks 中的 Interrupt 判断是否需要审批；
    恢复时用同一 checkpointer 重建图，使事件流入本次连接的队列。
    """

    def __init__(self, settings, llm, registry, session_store):
        self.harness = AgentHarness(settings)  # 护栏层：审批策略/资源上限/止损/统计
        self.settings = settings
        self.llm = llm  # 单模型路径（测试注入 Fake 等）时所有场景共用
        self.llm_service = llm if isinstance(llm, LLMService) else None  # 按场景取模型的路径
        self.registry = registry
        self.sessions = session_store
        self._configs: dict[str, dict] = {}
        self._specs: dict[str, tuple] = {}

    def _scenario_llm(self, scenario: str):
        """取指定场景的模型：有 LLMService 走场景配置，否则回退单模型。"""
        if self.llm_service is not None:
            return self.llm_service.get(scenario)
        return self.llm

    def _build_graph(self, mode, tools, emit):
        checkpointer = self.sessions.checkpointer
        if mode == "react":
            return build_react_agent(self._scenario_llm("chat"), tools, emit, self.settings, checkpointer, self.harness)
        if mode == "plan_execute":
            return build_plan_execute_agent(self._scenario_llm("planner"), self._scenario_llm("chat"), tools, emit, self.settings, checkpointer, self.harness)
        if mode == "reflection":
            return build_reflection_agent(self._scenario_llm("chat"), self._scenario_llm("critic"), tools, emit, self.settings, checkpointer, self.harness)
        if mode == "multi_agent":
            return build_multi_agent_agent(self._scenario_llm("chat"), tools, emit, self.settings, checkpointer, self.harness)
        raise ValueError(f"未知模式：{mode}")

    def _config(self, session_id, approval_policy, strategy):
        return {
            "configurable": {
                "thread_id": session_id,
                "approval_policy": approval_policy,
                "prompt_strategy": strategy,
            },
            "recursion_limit": self.harness.recursion_limit(),
        }

    async def _make_inputs(self, graph, config, message, strategy, rag_context=None):
        snap = await graph.aget_state(config)
        msgs = []
        if snap is not None and snap.values:
            msgs = list(snap.values.get("messages", []))
        if not msgs:
            base = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["standard"])
            msgs.append(SystemMessage(content=f"{base}\n\n{TOOL_RETRY_HINT}"))
        msgs.append(HumanMessage(content=self._augment_query(message, rag_context)))
        return {"messages": msgs}

    @staticmethod
    def _augment_query(message: str, rag_context: dict | None) -> str:
        """把检索命中注入用户消息：RAG 是独立检索阶段，不依赖模型主动调用工具。

        注入到用户消息而非 system prompt，因为各推理模式内部各自构造 system prompt，
        只有用户消息能保证被所有模式的首次模型调用看到。
        """
        if not rag_context or not rag_context.get("hits"):
            return message
        blocks = "\n".join(f"[相关度 {h['score']}] {h['text']}" for h in rag_context["hits"])
        return (
            f"{message}\n\n"
            f"【知识库检索结果（{rag_context['name']}）】\n{blocks}\n"
            "请优先基于以上检索内容回答；若内容不足以回答，再结合自身知识补充。"
        )

    @staticmethod
    def _make_emit(queue, tool_count):
        def emit(data):
            if data.get("type") == "tool_start":
                tool_count[0] += 1
            queue.put_nowait(data)

        return emit

    async def stream(self, session_id, message, mode, enabled, prompt_strategy, approval_policy, rag_scheme=None):
        """启动新一轮对话并产出 SSE 事件；遇 HITL 中断产出 approval_request 后暂停。

        rag_scheme：本轮选定的 RAG 方案 id（当前仅 naive，后续扩展），在 meta 中回显，
        并用于前置检索——RAG 是独立检索阶段，不进入工具集。
        """
        queue = asyncio.Queue()
        # 工具调用计数统一存到护栏层：本轮重置，后续 resume 复用同一计数器累计
        tool_count = self.harness.new_tool_counter(session_id)
        emit = self._make_emit(queue, tool_count)
        tools = build_tools(self.registry, enabled, session_id, emit)
        try:
            graph = self._build_graph(mode, tools, emit)
        except Exception as exc:
            yield self.harness.error_event(f"无法构建模式 {mode}", str(exc))
            return
        config = self._config(session_id, approval_policy, prompt_strategy)
        self._configs[session_id] = config
        self._specs[session_id] = (mode, tools)
        yield {"type": "meta", "session_id": session_id, "mode": mode, "capabilities": enabled, "rag_scheme": rag_scheme}
        # RAG 前置检索：启用 rag 能力时按选定方案自动召回并注入上下文（不依赖模型调用工具）
        rag_context = None
        if self.registry.rag_manager is not None:
            scheme = self.registry.rag_manager.resolve(rag_scheme)
            result = scheme.retrieve_full(message, self.settings.rag_top_k)
            emit(
                {
                    "type": "retrieve",
                    "query": message,
                    "scheme": scheme.id,
                    "hits": result.hits,
                    "rewrites": result.rewrites,  # Query 重写变体（advanced 有值）
                    "reranked": result.reranked,  # 是否经过重排
                }
            )
            rag_context = {"name": scheme.name, "hits": result.hits}
        inputs = await self._make_inputs(graph, config, message, prompt_strategy, rag_context=rag_context)
        async for ev in self._run_graph(session_id, graph, config, inputs, queue, tool_count):
            yield ev

    async def resume(self, approval_id, decision, modified_args):
        """通过审批号找到暂停的会话，恢复图执行并继续产出 SSE 事件。"""
        session_id = self.harness.resolve_approval(approval_id)
        spec = self._specs.get(session_id)
        config = self._configs.get(session_id)
        if session_id is None or spec is None or config is None:
            yield self.harness.error_event("审批会话不存在或已过期，请重新发送")
            return

        queue = asyncio.Queue()
        tool_count = self.harness.tool_counter(session_id)  # 复用该轮累计计数
        emit = self._make_emit(queue, tool_count)
        mode, tools = spec
        graph = self._build_graph(mode, tools, emit)
        # 同一 superstep 可能存在多个 pending interrupt（如一步内多个需审批的工具调用）：
        # 需按 interrupt id 以 resume map 恢复（LangGraph 要求），单个时保持原样
        interrupt_ids = self.harness.approval_interrupt_ids(approval_id)
        decision_value = {"action": decision, "modified_args": modified_args or {}}
        if len(interrupt_ids) > 1:
            command = Command(resume={iid: decision_value for iid in interrupt_ids})
        else:
            command = Command(resume=decision_value)
        async for ev in self._run_graph(session_id, graph, config, command, queue, tool_count):
            yield ev

    def stop(self, session_id: str) -> None:
        """取消指定会话正在运行的后台图任务（委托护栏层止损）。"""
        self.harness.stop(session_id)

    async def _run_graph(self, session_id, graph, config, inputs, queue, tool_count):
        """以后台任务驱动图执行，同时实时排空事件队列，实现真正的流式下发。

        任务登记到护栏层（harness），供 stop() 取消；客户端中断（SSE 连接关闭）时也在 finally 中取消。
        """
        sentinel = object()
        outcome: dict = {}

        async def _run():
            try:
                await graph.ainvoke(inputs, config)
                outcome["ok"] = True
            except asyncio.CancelledError:
                outcome["cancelled"] = True
            except ModelCallLimitExceededError:
                # create_agent 模式（react/multi_agent）达到轮数上限：转为 done 而非 error
                outcome["limit"] = True
            except Exception as exc:  # noqa: BLE001
                outcome["ok"] = False
                outcome["exc"] = exc
            finally:
                queue.put_nowait(sentinel)

        task = asyncio.create_task(_run())
        self.harness.register_run(session_id, task)
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item
        finally:
            self.harness.release_run(session_id)
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
        if outcome.get("limit"):
            # create_agent 模式达到轮数上限：转为 done（前端可见提示），而非 error
            yield {
                "type": "done",
                "summary": f"已达到最大轮数上限（{self.settings.max_steps} 轮），已停止执行",
                "stats": {"tool_calls": tool_count[0]},
            }
            return
        if not outcome.get("ok"):
            yield self.harness.error_event("Agent 运行失败", str(outcome.get("exc")))
            return

        snap = await graph.aget_state(config)
        pending = []
        for run_task in snap.tasks or ():
            pending.extend(run_task.interrupts or ())
        if pending:
            # 同一 superstep 可能产生多个 pending interrupt（如一步内多个需审批的工具调用），
            # 合并为一次审批请求（前端弹窗已支持多工具调用批量审批），resume 时按 interrupt id 映射恢复
            tool_calls = []
            interrupt_ids = []
            for intr in pending:
                payload = getattr(intr, "value", {}) or {}
                tool_calls.extend(payload.get("tool_calls", []))
                interrupt_ids.append(intr.id)
            approval_id = uuid.uuid4().hex
            self.harness.register_approval(approval_id, config["configurable"]["thread_id"], interrupt_ids)
            yield {
                "type": "approval_request",
                "approval_id": approval_id,
                "tool_calls": tool_calls,
            }
            return
        # 手写 StateGraph 模式（reflection / plan_execute）达到轮数上限：节点已置 stopped，转为 done
        values = snap.values or {}
        if values.get("stopped") == "max_steps":
            yield {
                "type": "done",
                "summary": f"已达到最大轮数上限（{self.settings.max_steps} 轮），已停止执行",
                "stats": {"tool_calls": tool_count[0]},
            }
            return
        yield {"type": "done", "summary": "本次任务处理完成", "stats": {"tool_calls": tool_count[0]}}
