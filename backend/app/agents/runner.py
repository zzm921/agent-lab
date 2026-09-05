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

from app.agents.context_manage import ContextManager
from app.agents.harness import AgentHarness
from app.agents.modes.multi_agent import build_multi_agent_agent
from app.agents.modes.plan_execute import build_plan_execute_agent
from app.agents.modes.react import build_react_agent
from app.agents.modes.reflection import build_reflection_agent
from app.agents.tools_builder import build_tools
from app.rag.task.graph import SessionLedger
from app.rag.task.rag_task_tool import make_knowledge_task_tool
from app.tools.rag_tool import rag_block_payload
from app.llm.service import LLMService
from app.memory.consolidate import maybe_consolidate
from app.memory.proactive import maybe_recall as proactive_recall
from app.security import InputGuard

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
        # 会话账本（P3）：跨任务累计内层触发/token，与任务账本叠加防级联超支
        self._task_ledgers: dict[str, SessionLedger] = {}
        # L2 主动语义召回：会话级「已见记忆 id 集合」——同一记忆不重复注入（跨轮去重）
        self._injected_memory_ids: dict[str, set[str]] = {}
        # 后台静默任务（轮末记忆提取等）：持句柄防 GC 取消，完成即移除
        self._bg_tasks: set[asyncio.Task] = set()
        # 上下文管理与压缩管线（snip/micro/auto-compact，大文件落盘在工具结果层挂钩）
        self.context_manager = ContextManager(settings)

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

    def _constant_memory_block(self, memory_enabled: bool = True, client_key: str = "default") -> tuple[str | None, int]:
        """常驻记忆块：从当前客户端的常驻库取高重要记忆（importance ≥ 阈值）top-k 组装为 system 块。

        返回 (注入块, 注入条数)；记忆未启用 / 无 embedding / 常驻库为空时返回 (None, 0)。
        仅在首轮（无历史）时注入；后续轮次沿用 checkpointer 内已有 system 消息，不重复注入。
        client_key 由请求层判定（设备指纹优先、IP 兜底），保证每个试用者只见自己的常驻记忆。
        """
        if not memory_enabled:
            return None, 0
        if not getattr(self.settings, "memory_enabled", True):
            return None, 0
        if not getattr(self.settings, "memory_constant_enabled", True):
            return None, 0
        if self.registry.embeddings is None:
            return None, 0
        try:
            constant = self.sessions.constant_memory(self.registry.embeddings, client_key)
            items = constant.constant_memories(
                self.settings.memory_constant_top_k,
                min_importance=self.settings.memory_constant_min_importance,
            )
        except Exception:  # noqa: BLE001 — 常驻注入失败绝不影响主链路
            return None, 0
        if not items:
            return None, 0
        lines = [
            f"- [{meta.get('kind', 'fact')}] {text}"
            for text, meta in items
        ]
        return (
            "## 用户记忆（来自历史会话，仅供参考）\n"
            "以下记忆可能过时或不准确，请作为背景参考；若与用户本次说明冲突，以本次说明为准：\n"
            + "\n".join(lines),
            len(items),
        )

    async def _proactive_memory_recall(self, session_id, client_key, query, emit):
        """L2 主动语义召回：selector 判断 → 会话库+常驻库合并召回 → 返回注入块。

        企业级四件套：触发判断（轻量 LLM，判否跳过）→ 合并召回 → 已见去重 → 预算封顶。
        注入到 user 消息（本轮生效）；命中即 touch 访问频率。任何异常吞掉不影响主链路。
        """
        if not getattr(self.settings, "memory_proactive_enabled", True):
            return None
        if self.registry.embeddings is None:
            return None
        try:
            store = self.sessions.long_memory(session_id, self.registry.embeddings)
            constant = self.sessions.constant_memory(self.registry.embeddings, client_key)
            injected = self._injected_memory_ids.setdefault(session_id, set())
            block, _, _ = await proactive_recall(
                self._scenario_llm("memory_selector"),
                store,
                constant,
                query,
                top_k=self.settings.memory_proactive_top_k,
                threshold=self.settings.memory_proactive_threshold,
                max_chars=self.settings.memory_proactive_max_chars,
                injected_ids=injected,
                emit=emit,
                selector_enabled=bool(getattr(self.settings, "memory_proactive_selector", True)),
            )
            return block
        except Exception:  # noqa: BLE001 — 主动召回失败绝不影响主链路
            return None

    def _seed_constant_ids(self, session_id, client_key, memory_enabled: bool = True):
        """把 L1 常驻注入的记忆 id 预置进会话已见集合，避免 L2 主动召回重复注入同一批常驻记忆。

        常驻记忆已在首轮注入 system（L1），L2 主动召回若再从常驻库召到同一批会重复注入
        （system 与 user 双份、浪费 token 且干扰模型）。首轮先 seed，proactive 的已见去重
        自然跳过这些 id。仅依赖 constant_memories 的返回 meta.id，任何异常静默吞掉。
        """
        if not memory_enabled or not getattr(self.settings, "memory_enabled", True):
            return
        if not getattr(self.settings, "memory_constant_enabled", True):
            return
        if self.registry.embeddings is None:
            return
        try:
            constant = self.sessions.constant_memory(self.registry.embeddings, client_key)
            items = constant.constant_memories(
                self.settings.memory_constant_top_k,
                min_importance=self.settings.memory_constant_min_importance,
            )
        except Exception:  # noqa: BLE001 — seed 失败不影响主链路
            return
        injected = self._injected_memory_ids.setdefault(session_id, set())
        for _text, meta in items:
            rid = meta.get("id")
            if rid:
                injected.add(rid)

    async def _make_inputs(self, graph, config, message, strategy, emit=None, rag_context=None, insufficient=False, generation_mode=None, keep_rounds=0, memory_enabled=True, client_key="default", memory_block=None):
        snap = await graph.aget_state(config)
        msgs = []
        if snap is not None and snap.values:
            msgs = list(snap.values.get("messages", []))
        # 上下文管理与压缩：对历史副本执行 snip → micro → auto 管线（四模式统一生效）。
        # 只作用于副本、不写回 checkpointer（原始历史保留可回滚）；在本轮 HumanMessage 追加前执行，
        # 本轮消息始终原文保留。关闭总开关则整条管线不生效，行为回退现状。
        # keep_rounds>0 时进入「每轮压缩」演示模式：保留最近 keep_rounds 轮原文，更早历史每轮都裁剪。
        if msgs and getattr(self.settings, "context_mgmt_enabled", True):
            msgs, ctx_events = await self.context_manager.build(
                msgs,
                llm=self._scenario_llm("chat"),
                session_id=config["configurable"]["thread_id"],
                keep_rounds=int(keep_rounds or 0),
            )
            if emit is not None:
                for ev in ctx_events:
                    emit({"type": "context", **ev})
        if not msgs:
            base = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["standard"])
            content = f"{base}\n\n{TOOL_RETRY_HINT}"
            block, count = self._constant_memory_block(memory_enabled, client_key)
            if block:
                content = f"{content}\n\n{block}"
                if emit is not None:
                    emit({"type": "memory_constant", "count": count})
            msgs.append(SystemMessage(content=content))
        msgs.append(HumanMessage(content=self._augment_query(message, rag_context, insufficient, generation_mode, memory_block)))
        return {"messages": msgs}

    @staticmethod
    def _augment_query(
        message: str,
        rag_context: dict | None,
        insufficient: bool = False,
        generation_mode: str | None = None,
        memory_block: str | None = None,
    ) -> str:
        """把检索命中注入用户消息：RAG 是独立检索阶段，不依赖模型主动调用工具。

        注入到用户消息而非 system prompt，因为各推理模式内部各自构造 system prompt，
        只有用户消息能保证被所有模式的首次模型调用看到。

        generation_mode：语义路由产出的生成策略（direct / citation / comparison）；
        naive/advanced/modular 前置检索命中时据此拼装注入指令；agentic 走主循环内检索
        工具（knowledge_retrieve）时，前置只注入生成策略提示（检索结果由工具回传）；
        检索结果不足（insufficient）时强制模型如实说明缺失信息并向用户追问澄清
        （指令优先级最高，不依赖自身知识编造）。

        memory_block：L2 主动语义召回注入块（由 proactive 模块生成），叠加在 RAG 注入之后；
        均注入 user 消息，确保首轮模型调用可见。

        说明：知识库为受控内部语料，视为可信来源，不做「不可信外部数据」包装；
        仅对 web_search / run_command / memory_recall 等真正的外部来源做注入隔离。
        """
        content = message
        if rag_context and rag_context.get("hits"):
            content = f"{message}\n\n{rag_block_payload(rag_context, insufficient, generation_mode, message)}"
        elif insufficient:
            content = (
                f"{message}\n\n"
                "【知识库检索结果】本轮未检索到与问题相关的知识库内容。"
                "请如实告知用户当前检索未能获取足够信息，说明缺失的关键信息，"
                "并礼貌地向用户追问补充依据（如具体文件、部门名称等）；"
                "不要编造、不要依赖自身知识臆测内部数据，也不要声称工具不可用。"
            )
        elif generation_mode:
            # 循环内检索方案（modular/agentic）：前置只产出生成策略，检索结果由
            # knowledge_retrieve 工具回传；先注入策略提示引导组织方式。
            hint = {
                "direct": "请直接回答用户问题，无需标注引用来源。",
                "comparison": "请用 Markdown 对比表格组织回答，并标注对比项来源。",
                "citation": "请基于知识库来源回答，关键事实标注引用。",
            }.get(generation_mode)
            if hint:
                content = f"{content}\n\n【生成策略】{hint}"
        if memory_block:
            content = f"{content}\n\n{memory_block}"
        return content

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

    async def _run_rag_stage(self, session_id, message, graph, config, scheme, result: dict, context_holder: dict | None = None):
        """RAG 前置检索阶段（Agent 主循环外），按方案分流：

        - naive / advanced / modular：全量前置检索（L0/L1，不进入工具集），命中注入
          user 消息，过程事件逐条转发给前端；跨轮 seed 缓存在本阶段维护。
        - agentic：主循环内检索方案（L2）——检索封装为 knowledge_retrieve 工具，
          由主 Agent 按需触发（可与其他工具协同），跨轮 seed 由该工具维护；前置只做
          轻量语义路由（消费到 classify 即停止，不触发检索），产出 generation_mode 注入
          提示，并把最近会话上下文写入 context_holder 供工具内指代消解使用。

        作为异步生成器运行：runner.stream 用 `async for` 消费并逐条转发 naive/advanced
        的检索过程事件给前端；结束时把提取结果写入 result（生成器无独立返回值通道）：
        - rag_context：{name, hits} 命中上下文，注入主 LLM；
        - effective_message：指代消解/查询改写后的 query，替换主 LLM 输入；
        - generation_mode：语义路由产出的生成策略（citation/comparison/direct）；
        - insufficient：检索结果不足 → 强制模型追问澄清，不编造内部数据。
        """
        # 最近会话上下文（用户/助手回合）：供 naive/advanced 检索改写、modular/agentic
        # 前置指代消解；modular/agentic 还写入 context_holder 供主循环内工具做指代消解。
        context = await self._recent_context(graph, config)
        if context_holder is not None:
            context_holder["recent"] = context
        if getattr(scheme, "id", None) == "agentic":
            # 循环内工具方案：前置只做轻量语义路由——走独立 classify 接口（不进入完整
            # 检索链路、不做指代消解），产出 generation_mode 注入提示；路由结果写入
            # context_holder 供主循环内 knowledge_retrieve 工具复用（跳过二次路由）。
            route = await asyncio.to_thread(scheme.classify, message, context)
            if context_holder is not None:
                context_holder["route"] = route
            result.update(
                rag_context=None,
                effective_message=message,
                generation_mode=route.get("generation_mode"),
                insufficient=False,
            )
            return
        # naive / advanced / modular：全量前置检索（L0/L1）；modular 支持跨轮 seed 复用
        prev_hits = self._last_hits.get(session_id)
        stream_kwargs = {"context": context}
        if getattr(scheme, "id", None) == "modular" and prev_hits:
            stream_kwargs["seed_hits"] = prev_hits
        rag_context = None
        effective_message = message
        generation_mode = None
        insufficient = False
        async for ev in scheme.astream(message, self.settings.rag_top_k, **stream_kwargs):
            yield ev
            if ev["type"] == "rewrite" and ev.get("reason") == "指代消解" and ev.get("rewrites"):
                # rewrite 事件：指代消解后的 query 作为主 LLM 输入
                effective_message = ev["rewrites"][0]
            elif ev["type"] == "classify":
                generation_mode = ev.get("generation_mode") or generation_mode
            elif ev["type"] == "retrieve":
                # retrieve 事件携带实际用于检索的 query，作为无 rewrite 事件场景的兜底
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
        result.update(
            rag_context=rag_context,
            effective_message=effective_message,
            generation_mode=generation_mode,
            insufficient=insufficient,
        )

    async def _consolidate_memory_bg(self, graph, config, session_id, memory_enabled: bool = True, client_key: str = "default"):
        """轮末自动提取巩固（后台静默）：把本轮对话提炼为长期记忆写入库。

        由 stream 以 asyncio.create_task 在后台启动，不阻塞 SSE 事件流、不产
        memory_write 事件（静默）；提取出的 global 长期偏好/约束写入当前客户端
        的常驻库（跨会话生效），session 临时上下文写入会话库；任何异常被吞掉。
        """
        if not memory_enabled:
            return
        if not getattr(self.settings, "memory_enabled", True):
            return
        if self.registry.embeddings is None:
            return
        try:
            snap = await graph.aget_state(config)
        except Exception:  # noqa: BLE001
            return
        msgs = (snap.values.get("messages") or []) if snap is not None and snap.values else []
        if not msgs:
            return
        store = self.sessions.long_memory(session_id, self.registry.embeddings)
        constant = self.sessions.constant_memory(self.registry.embeddings, client_key)
        # 提取用独立轻量场景（memory_consolidate：关闭思考）而非 chat 主模型——实测 thinking 使提取慢到 40s+，关闭后 ~2s
        await maybe_consolidate(store, constant, msgs, self._scenario_llm("memory_consolidate"), self.settings, session_id)

    def _schedule_consolidate(self, graph, config, session_id, memory_enabled: bool = True, client_key: str = "default"):
        """后台静默启动轮末记忆提取：不 await、不产事件，任务句柄登记防 GC 取消。"""
        task = asyncio.create_task(
            self._consolidate_memory_bg(graph, config, session_id, memory_enabled, client_key)
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def stream(self, session_id, message, mode, enabled, prompt_strategy, approval_policy, rag_scheme=None, rag_enabled=True, memory_enabled=True, context_keep_rounds=0, client_key="default"):
        """启动新一轮对话并产出 SSE 事件；遇 HITL 中断产出 approval_request 后暂停。

        rag_scheme：本轮选定的 RAG 方案 id，在 meta 中回显。naive/advanced/modular
        为前置检索（L0/L1，不进入工具集）；agentic 把检索封装为 knowledge_task 工具
        进入主循环（L2），由 Agent 按需调用，前置仅做轻量语义路由。
        rag_enabled：本轮是否启用知识库检索（默认开启，由前端开关控制；总开关由
        settings.rag_enabled 控制，未开启时 registry.rag_manager 为 None，
        二者都满足才执行前置检索/注入检索工具）。
        context_keep_rounds：>0 时进入「每轮压缩」演示模式——保留最近 N 轮对话原文，
        更早的历史每轮都被压缩（页面持续展示压缩卡片）；0 使用系统默认阈值。
        """
        queue = asyncio.Queue()
        # 工具调用计数统一存到护栏层：本轮重置，后续 resume 复用同一计数器累计
        tool_count = self.harness.new_tool_counter(session_id)
        emit = self._make_emit(queue, tool_count)
        # 本轮关闭记忆时：从能力清单剔除 memory，常驻注入/轮末巩固一并关闭
        if not memory_enabled:
            enabled = [c for c in enabled if c != "memory"]
            
        tools = build_tools(self.registry, enabled, session_id, emit)
        # 主循环内检索工具：仅 agentic 把检索封装为 knowledge_task 工具交给主 Agent
        # 按需调用（L2，可与其他工具协同决策）；naive/advanced/modular 维持前置检索
        # （L0/L1，不进入工具集）。工具须在构建图之前注入，否则不在本轮工具集中。
        # context_holder 供工具做指代消解：由 _run_rag_stage 前置阶段填充最近会话上下文。
        rag_scheme_obj = None
        rag_ctx_holder: dict = {}
        if self.registry.rag_manager is not None and rag_enabled:
            rag_scheme_obj = self.registry.rag_manager.resolve(rag_scheme)
            if getattr(rag_scheme_obj, "id", None) == "agentic":
                # 单一检索入口（合并后）：knowledge_task 内先规则粗筛——简单问题单节点
                # 直通内层（零额外 LLM 成本），复合/链式才走任务图状态机；不再并列
                # knowledge_retrieve，消灭「两个工具让模型自觉二选一」的重复入口。
                ledger = self._task_ledgers.setdefault(session_id, SessionLedger(
                    max_inner_calls=self.settings.rag_agent_task_session_max_inner_calls,
                    token_budget=self.settings.rag_agent_task_session_token_budget,
                ))
                tools = [*tools, make_knowledge_task_tool(
                    rag_scheme_obj, self.settings, emit, session_id, self._last_hits, rag_ctx_holder,
                    ledger,
                )]
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
        # RAG 前置阶段：naive/advanced/modular 全量检索并注入 user 消息（L0/L1）；agentic 只做
        # 轻量语义路由（产出 generation_mode 注入提示），检索由主循环内 knowledge_retrieve
        # 工具按需触发。需总开关开启（rag_manager 非 None）且本轮开启（rag_enabled=True）
        # 才执行；naive/advanced 的过程事件经生成器逐条直发前端保持解耦。
        rag_context = None
        insufficient = False
        generation_mode = None  # 语义路由产出的生成策略（citation/comparison/direct），注入主 LLM 时使用
        # 注入给主 LLM 的用户消息：默认原消息；naive/advanced 改写后用它产出的 query 替换。
        effective_message = message
        if self.registry.rag_manager is not None and rag_enabled:
            # RAG 前置阶段整体收敛到 _run_rag_stage：流式转发 naive/advanced 过程事件给前端，
            # 并在内部完成信号提取（effective_message/generation_mode/rag_context/insufficient）
            # 与跨轮 seed 缓存更新（modular/agentic 由工具维护）；这里只逐条转发并取回结果。
            rag_result = {}
            async for ev in self._run_rag_stage(session_id, message, graph, config, rag_scheme_obj, rag_result, rag_ctx_holder):
                yield ev
            rag_context = rag_result.get("rag_context")
            effective_message = rag_result.get("effective_message") or message
            generation_mode = rag_result.get("generation_mode")
            insufficient = rag_result.get("insufficient", False)
        # L2 主动语义召回：本轮记忆开启时，把当前对话（含 RAG 消解后的 query）转召回，
        # 命中注入 user 消息（与 RAG 前置检索解耦、互不影响）。selector 判断/召回/注入全部
        # 静默失败容错；即使未命中也只产出 memory_read 事件（need=false/hits=[]），不报错。
        memory_block = None
        if memory_enabled:
            # L1/L2 去重打通：会话首次进入记忆路径时，先把 L1 常驻注入的记忆 id 预置进
            # 已见集合，避免 L2 主动召回把同一批常驻记忆再注入一遍（system 与 user 双份）。
            if session_id not in self._injected_memory_ids:
                self._seed_constant_ids(session_id, client_key, memory_enabled)
            memory_block = await self._proactive_memory_recall(
                session_id, client_key, effective_message, emit
            )
        inputs = await self._make_inputs(
            graph,
            config,
            effective_message,
            prompt_strategy,
            emit=emit,
            rag_context=rag_context,
            insufficient=insufficient,
            generation_mode=generation_mode,
            keep_rounds=context_keep_rounds,
            memory_enabled=memory_enabled,
            client_key=client_key,
            memory_block=memory_block,
        )
        async for ev in self._run_graph(session_id, graph, config, inputs, queue, tool_count):
            if ev["type"] == "done":
                # 轮末记忆提取改后台静默：不阻塞 done 下发、不产事件（记忆是增强项，写入失败也不影响本轮）
                self._schedule_consolidate(graph, config, session_id, memory_enabled, client_key)
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
