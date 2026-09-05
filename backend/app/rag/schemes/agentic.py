"""Agentic RAG 方案（独立范式）：多 Agent 编排 + 工具注册表 + 预算治理。

与 Modular RAG（静态执行计划 + 有界升级 1 轮）的本质差异——企业级 Agentic 形态：
- 编排独立：不再继承 ModularRagScheme，状态机/角色/工具/预算全部自有
  （app/rag/agentic/ 包）；仅复用底层原语（分块入库/重排/压缩/父块回填——基础设施）；
- 角色分离：Router（Adaptive RAG 路由）/ Planner（事实清单+首发计划）/
  Retriever（工具注册表并行执行）/ Grader（CRAG 证据评审）/ Corrector（CRAG 纠错）/
  Verifier（Self-RAG 事实-证据支持度校验），各自命名 LLM 场景 + 确定性规则回退；
- 双闭环：证据评审闭环（grade→correct→retrieve 回环，预算内）+ 答案校验闭环
  （verify 判定可答/缺口，不足如实上报 clarify）；
- 预算治理：步数 / 纠错轮数 / token / 墙钟超时 / 单工具上限 / 角色熔断，全部可配置。

本类是薄适配层：把 RagScheme 接口（ingest/retrieve/astream）桥接到编排器，
前置指代消解与跨轮 seed 闸门在此完成（与 runner/SSE 协议保持兼容——classify/
rewrite/retrieve/answerability 事件语义不变，新增 plan/agent_step/grade/correct/verify）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.memory.stores.base import StoreBackend
from app.rag.agentic.orchestrator import AgenticOrchestrator, OrchestratorBudgets
from app.rag.agentic.tools import cross_turn_seed
from app.rag.base import RetrieveResult
from app.rag.retrieval.context_compress import build_compressor
from app.rag.retrieval.iterative_retrieval import build_multi_hop_retriever
from app.rag.routing.deictic_resolver import build_deictic_resolver
from app.rag.schemes.advanced import AdvancedRagScheme

logger = logging.getLogger(__name__)


class AgenticRagScheme(AdvancedRagScheme):
    """Agentic RAG：继承 Advanced 仅取入库分块/重排/父块回填等底层原语，
    编排完全独立——多 Agent 状态机取代静态执行计划。"""

    id: str = "agentic"
    name: str = "Agentic RAG"
    description: str = "多 Agent 编排（路由/规划/检索/评审/纠错/校验）+ 工具注册表 + 预算治理"
    hybrid: bool = True
    multi_backend: bool = True

    def __init__(
        self,
        embeddings,
        store: StoreBackend,
        top_k: int = 3,
        rewrite_variants: int = 3,
        rerank_model: str = "qwen3-rerank",
        rewriter=None,
        reranker=None,
        hyde=None,
        deictic=None,
        compressor=None,
        multi_hop=None,
        max_hops: int = 3,
        orchestrator: AgenticOrchestrator | None = None,
        # 预算治理配置
        max_steps: int = 8,
        correction_rounds: int = 2,
        timeout_s: float = 90.0,
        token_budget: int = 8000,
        call_cap: int = 3,
        parallel: int = 4,
    ):
        super().__init__(
            embeddings, store, top_k,
            rewrite_variants=rewrite_variants, rerank_model=rerank_model,
            rewriter=rewriter, reranker=reranker, hyde=hyde,
        )
        self.compressor = compressor if compressor is not None else build_compressor(embeddings=embeddings)
        self.multi_hop = multi_hop if multi_hop is not None else build_multi_hop_retriever()
        self.max_hops = max_hops
        self.deictic = deictic if deictic is not None else build_deictic_resolver()
        self.orchestrator = orchestrator if orchestrator is not None else AgenticOrchestrator(
            store,
            embeddings,
            self.reranker,
            self.compressor,
            parent_resolver=self._resolve_parents,
            multi_hop=self.multi_hop,
            max_hops=max_hops,
            hyde=self.hyde,
            budgets=OrchestratorBudgets(
                max_steps=max_steps,
                correction_rounds=correction_rounds,
                timeout_s=timeout_s,
                token_budget=token_budget,
                call_cap=call_cap,
                parallel=parallel,
            ),
        )

    # ---- 接口实现：同步 ----

    def retrieve_full(
        self,
        query: str,
        top_k: int | None = None,
        context: str | None = None,
        seed_hits: list[dict[str, Any]] | None = None,
        memory: str | None = None,
    ) -> RetrieveResult:
        """同步完整检索：指代消解 → 跨轮 seed 闸门 → 状态机编排。

        memory：L2 主动语义召回的用户记忆块（背景参考），供指代消解参考，
        并挂入 AgentState 供 Planner/Corrector 实体化身份、Verifier 判定缺失（记忆先行）。
        """
        k = top_k or self.top_k
        resolved = self.deictic.resolve(query, context, memory) or query
        if resolved != query:
            logger.info("[agentic] 执行：指代消解 %r → %r", query, resolved)
        seed = cross_turn_seed(resolved, seed_hits) if seed_hits else []
        if seed:
            logger.info("[agentic] 执行：跨轮 seed 复用 → %d 条候选证据", len(seed))
        result = self.orchestrator.run(resolved, k=k, seed_hits=seed, memory=memory)
        logger.info(
            "[agentic] 执行：编排完成 → 命中 %d 条 answerable=%s 纠错 %d 轮",
            len(result.hits), result.answerable, result.corrections,
        )
        return RetrieveResult(
            query=resolved,
            hits=result.hits,
            reranked=result.reranked,
            compressed=result.compressed,
            answerability=result.verdict,
            trace=result.trace,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.retrieve_full(query, top_k).hits

    # ---- 接口实现：异步流式（SSE 逐事件） ----

    async def astream(
        self,
        query: str,
        top_k: int | None = None,
        context: str | None = None,
        seed_hits: list[dict[str, Any]] | None = None,
        memory: str | None = None,
    ):
        """异步流式：前置消解/seed 事件 → 编排器逐事件（classify/plan/agent_step/grade/
        correct/verify/retrieve/compress/answerability）。

        指代消解为同步 LLM 调用，放线程池（不阻塞事件循环，项目硬约束）。
        memory：L2 主动语义召回的用户记忆块（背景参考），供指代消解参考，
        并挂入 AgentState 供 Planner/Corrector 实体化身份、Verifier 判定缺失（记忆先行）。
        """
        k = top_k or self.top_k
        resolved = (await asyncio.to_thread(self.deictic.resolve, query, context, memory)) or query
        if resolved != query:
            logger.info("[agentic] 流式：指代消解 %r → %r", query, resolved)
            yield {
                "type": "rewrite", "query": query, "scheme": self.id,
                "rewrites": [resolved], "reason": "指代消解",
            }
        query = resolved
        seed = cross_turn_seed(query, seed_hits) if seed_hits else []
        if seed:
            logger.info("[agentic] 流式：跨轮 seed 复用 → %d 条候选证据", len(seed))
            yield {"type": "seed_reuse", "query": query, "scheme": self.id, "count": len(seed)}
        async for ev in self.orchestrator.astream(query, k=k, seed_hits=seed, memory=memory):
            yield ev
