"""Agent 统一运行器：构建模式图、SSE 流式运行、HITL 中断/恢复。

职责：编排循环（构建模式图、事件队列驱动流式下发、resume 恢复执行）；
护栏相关（审批策略、资源上限、止损、统计）统一委托 AgentHarness（app.agents.harness）。
"""
from __future__ import annotations

import asyncio
import uuid

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from app.agents.harness import AgentHarness
from app.agents.modes.multi_agent import build_multi_agent_agent
from app.agents.modes.plan_execute import build_plan_execute_agent
from app.agents.modes.react import build_react_agent
from app.agents.modes.reflection import build_reflection_agent
from app.agents.tools_builder import build_tools
from app.llm.service import LLMService
from app.security import InputGuard, wrap_untrusted

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
        # 输入 Guardrail（security.md 第一层）：越狱/提示注入特征命中即礼貌拒绝
        self.input_guard = InputGuard(
            enabled=bool(
                getattr(self.settings, "security_enabled", True)
                and getattr(self.settings, "guard_input", True)
            )
        )
        self.llm = llm  # 单模型路径（测试注入 Fake 等）时所有场景共用
        self.llm_service = llm if isinstance(llm, LLMService) else None  # 按场景取模型的路径
        self.registry = registry
        self.sessions = session_store
        self._configs: dict[str, dict] = {}
        self._specs: dict[str, tuple] = {}
        # 跨轮 seed 复用：按会话记录上一轮最终检索命中（已含重排/压缩），
        # 供 modular 下一轮作为「候选证据」复用（经 _cross_turn_seed 分数/相关性闸门过滤）。
        self._last_hits: dict[str, list[dict]] = {}

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

    async def _make_inputs(self, graph, config, message, strategy, rag_context=None, insufficient=False, generation_mode=None):
        snap = await graph.aget_state(config)
        msgs = []
        if snap is not None and snap.values:
            msgs = list(snap.values.get("messages", []))
        if not msgs:
            base = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["standard"])
            msgs.append(SystemMessage(content=f"{base}\n\n{TOOL_RETRY_HINT}"))
        mark_untrusted = bool(
            getattr(self.settings, "security_enabled", True)
            and getattr(self.settings, "mark_untrusted", True)
        )
        msgs.append(
            HumanMessage(
                content=self._augment_query(message, rag_context, insufficient, generation_mode, mark_untrusted)
            )
        )
        return {"messages": msgs}

    @staticmethod
    def _build_sources(hits: list[dict]) -> list[str]:
        """从命中元数据生成来源清单：卷/章/节 → 文件 → 兜底「知识库」。"""
        sources = []
        for i, h in enumerate(hits, start=1):
            meta = h.get("metadata") or {}
            parts = [x for x in (meta.get("volume"), meta.get("chapter"), meta.get("section")) if x]
            if parts:
                label = " / ".join(parts)
            else:
                src = meta.get("source")
                label = src if src and src != "builtin" else "知识库"
            sources.append(f"[{i}] {label}")
        return sources

    @staticmethod
    def _augment_query(
        message: str,
        rag_context: dict | None,
        insufficient: bool = False,
        generation_mode: str | None = None,
        mark_untrusted: bool = True,
    ) -> str:
        """把检索命中注入用户消息：RAG 是独立检索阶段，不依赖模型主动调用工具。

        注入到用户消息而非 system prompt，因为各推理模式内部各自构造 system prompt，
        只有用户消息能保证被所有模式的首次模型调用看到。

        generation_mode：语义路由产出的生成策略（direct / citation / comparison）；
        无路由事件（naive / advanced）时默认 citation；检索结果不足（insufficient）时
        强制模型如实说明缺失信息并向用户追问澄清（指令优先级最高，不依赖自身知识编造）。

        mark_untrusted：为 True 时把检索命中包上「不可信外部数据」分隔符（security.md
        来源可信分级 / 提示注入防御）——知识库内容属外部数据，其中夹带的指令一律忽略。
        """
        if not rag_context or not rag_context.get("hits"):
            # 零命中：无「答案不足」信号时原样返回；但若答案充分性判定需澄清
            # （insufficient=True，如 out-of-kb/证据缺口），必须仍注入追问澄清指令——
            # 否则主 LLM 拿不到任何上下文，会凭空编造（如声称工具不可用、依赖自身
            # 知识作答），而非如实说明信息缺失。
            if insufficient:
                return (
                    f"{message}\n\n"
                    "【知识库检索结果】本轮未检索到与问题相关的知识库内容。"
                    "请如实告知用户当前检索未能获取足够信息，说明缺失的关键信息，"
                    "并礼貌地向用户追问补充依据（如具体文件、部门名称等）；"
                    "不要编造、不要依赖自身知识臆测内部数据，也不要声称工具不可用。"
                )
            return message
        hits = rag_context["hits"]
        name = rag_context["name"]
        if insufficient:
            blocks = "\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, start=1))
            data_block = wrap_untrusted(blocks, f"知识库·{name}") if mark_untrusted else f"【知识库检索结果（{name}）】\n{blocks}"
            return (
                f"{message}\n\n"
                f"{data_block}\n"
                + "请严格基于以上检索内容如实回答；若检索内容不足以回答用户问题，"
                "请明确说明缺失的关键信息，并礼貌地向用户追问补充，不要编造、"
                "不要依赖自身知识臆测内部人事数据。"
            )
        mode = generation_mode or "citation"
        if mode == "direct":
            blocks = "\n".join(h["text"] for h in hits)
            instruction = "请直接回答用户问题，无需标注引用来源。"
        elif mode == "comparison":
            blocks = "\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, start=1))
            instruction = (
                "请基于以上检索内容，用 Markdown 对比表格结构化回答：每一行一个对比维度，"
                "每一列一个对比对象；表格内关键结论在句末标注上方编号 [1]/[2] 的来源。"
                "表格之后用一段话总结差异，末尾附「引用来源」清单（编号 → 出处）。"
            )
        else:  # citation
            blocks = "\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, start=1))
            instruction = (
                "请优先基于以上检索内容回答；每个关键事实在句末标注上方编号 [1]/[2] 的来源。"
                "回答末尾附「引用来源」清单（编号 → 出处）。"
                "若检索内容不足以回答，再结合自身知识补充，并注明哪些属于推测。"
            )
        sources = "\n".join(AgentRunner._build_sources(hits)) if mode != "direct" else ""
        data_block = wrap_untrusted(blocks, f"知识库·{name}") if mark_untrusted else f"【知识库检索结果（{name}）】\n{blocks}"
        return (
            f"{message}\n\n"
            f"{data_block}\n"
            + (f"来源：\n{sources}\n" if sources else "")
            + instruction
        )

    @staticmethod
    def _make_emit(queue, tool_count):
        def emit(data):
            if data.get("type") == "tool_start":
                tool_count[0] += 1
            queue.put_nowait(data)

        return emit

    @staticmethod
    async def _recent_context(graph, config, limit: int = 6) -> str | None:
        """取最近会话上下文（用户/助手回合文本），供 RAG 指代消解使用；无历史时返回 None。"""
        try:
            snap = await graph.aget_state(config)
        except Exception:  # noqa: BLE001 — 取上下文失败不应阻断检索
            return None
        msgs = (snap.values.get("messages") or []) if snap is not None and snap.values else []
        turns = []
        for m in msgs[-limit:]:
            if isinstance(m, HumanMessage):
                turns.append(f"用户: {m.content}")
            elif isinstance(m, AIMessage):
                content = m.content if isinstance(m.content, str) else ""
                if not content:
                    continue
                turns.append(f"助手: {content}")
        return "\n".join(turns) if turns else None

    async def stream(self, session_id, message, mode, enabled, prompt_strategy, approval_policy, rag_scheme=None, rag_enabled=True):
        """启动新一轮对话并产出 SSE 事件；遇 HITL 中断产出 approval_request 后暂停。

        rag_scheme：本轮选定的 RAG 方案 id（当前仅 naive，后续扩展），在 meta 中回显，
        并用于前置检索——RAG 是独立检索阶段，不进入工具集。
        rag_enabled：本轮是否启用知识库检索（默认开启，由前端开关控制；总开关由
        settings.rag_enabled 控制，未开启时 registry.rag_manager 为 None，
        二者都满足才执行前置检索）。
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
        yield {"type": "meta", "session_id": session_id, "mode": mode, "capabilities": enabled, "rag_scheme": rag_scheme, "rag_enabled": rag_enabled}
        # 输入 Guardrail（security.md 第一层）：越狱/提示注入特征命中即礼貌拒绝，
        # 不进入 RAG 检索与图执行，避免浪费 token 与扩大风险面。
        verdict = self.input_guard.check(message)
        if verdict.rejected:
            # 直接下发拒绝文案（不经事件队列：本路径不会进入 _run_graph 排空队列，
            # 若用 emit_text 入队会导致「抱歉」文案永远不被消费）
            yield {"type": "message", "delta": self.input_guard.refusal}
            yield {"type": "guard_refused", "reason": verdict.reason, "matched": verdict.matched}
            yield {"type": "done", "summary": "已按安全策略拒绝", "stats": {"tool_calls": 0}}
            return
        # RAG 前置检索：启用 rag 能力时按选定方案自动召回并注入上下文（不依赖模型调用工具）。
        # 需总开关开启（rag_manager 非 None）且本轮请求开启（rag_enabled=True）才执行。
        # 方案经 astream 流式产出事件（rewrite→retrieve），runner 逐条直发前端保持解耦；
        # 最后一条 retrieve 事件里的 hits 用于上下文注入。
        rag_context = None
        insufficient = False
        generation_mode = None  # 语义路由产出的生成策略（citation/comparison/direct），注入主 LLM 时使用
        # 注入给主 LLM 的用户消息：默认原消息；modular 指代消解后用它产出的消解 query 替换，
        # 避免主 LLM 对「他/这…」二次解析（把指代词误指回上轮问题主语）导致答非所问。
        effective_message = message
        if self.registry.rag_manager is not None and rag_enabled:
            scheme = self.registry.rag_manager.resolve(rag_scheme)
            # 最近会话上下文（用户/助手回合）：RAG 是独立检索阶段，只有当前消息；
            # 传入上下文供 modular 前置「指代消解」把「他/这…」替换为具体实体。
            context = await self._recent_context(graph, config)
            # 跨轮 seed 复用：modular/agentic 方案支持（共享 _cross_turn_seed 闸门）；
            # 传入上一轮最终命中，由方案内按分数/相关性过滤后作候选证据
            # （省重复检索、不注入查询文本）。
            prev_hits = self._last_hits.get(session_id)
            stream_kwargs = {"context": context}
            if getattr(scheme, "id", None) in ("modular", "agentic") and prev_hits:
                stream_kwargs["seed_hits"] = prev_hits
            async for ev in scheme.astream(message, self.settings.rag_top_k, **stream_kwargs):
                yield ev
                if ev["type"] == "rewrite" and ev.get("reason") == "指代消解" and ev.get("rewrites"):
                    effective_message = ev["rewrites"][0]
                elif ev["type"] == "classify":
                    generation_mode = ev.get("generation_mode") or generation_mode
                elif ev["type"] == "retrieve":
                    # retrieve 事件携带实际用于检索的 query（modular 已含指代消解结果），
                    # 作为无 rewrite 事件场景的兜底（如未消解但有查询改写时保持原文）。
                    effective_message = ev.get("query") or effective_message
                    if ev.get("hits"):
                        rag_context = {"name": scheme.name, "hits": ev["hits"]}
                elif ev["type"] == "answerability":
                    # 最后一条 answerability 事件即最终验证结论：
                    # 检索结果不足以回答 → 强制模型追问澄清，不编造内部数据。
                    if ev.get("verdict", {}).get("answerable") is False:
                        insufficient = True
            # 更新跨轮 seed 缓存：本轮检索到命中则记录（供下一轮复用），否则清空
            if rag_context and rag_context.get("hits"):
                self._last_hits[session_id] = rag_context["hits"]
            else:
                self._last_hits.pop(session_id, None)
        inputs = await self._make_inputs(
            graph,
            config,
            effective_message,
            prompt_strategy,
            rag_context=rag_context,
            insufficient=insufficient,
            generation_mode=generation_mode,
        )
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
