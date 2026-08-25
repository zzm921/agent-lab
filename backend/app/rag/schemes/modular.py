"""模块化 RAG 方案：前置语义路由（Query Router）+ 执行计划（Execution Plan）编排。

对应 Modular RAG 企业级架构的「调度层 + 预处理/检索/后处理模块组」：
把检索链路拆成可插拔模块，由路由决策（RouteDecision）映射为执行计划（ExecutionPlan），
再按计划动态编排执行（模块可组合、可跳过，正是 Modular RAG 的核心价值）——

- 预处理模块组：查询改写（rewrite） / 查询分解（decompose）；
- 检索模块组：向量检索（search） / 混合检索（hybrid_search） / 多路召回（multi_recall）；
- 后处理模块组：重排（rerank） / 上下文压缩（compress）；
- 生成策略：direct / citation / comparison（前端据此提示生成方式）。

路由策略（Adaptive RAG 思路，小模型/规则先路由再选路）：
- retrieval_need=False → 不检索，直接生成（省延迟）；
- complexity=simple + vector → 单次向量检索（同 naive）；
- complexity=simple + hybrid → 单次混合检索（语义 + 关键词）；
- complexity=rewrite + hybrid/multi_recall → 改写后多路召回；
- complexity=decompose + multi_recall → 分解为子查询分别召回 → 融合 → 重排 → 压缩。

路由能力「随 modular 方案是否被配置而开/关」：路由器内置于本方案，
settings.rag_schemes 不含 "modular" 时方案不构建，路由逻辑即不执行。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.memory.stores.base import StoreBackend
from app.rag.retrieval.answerability import (
    ANSWER,
    CLARIFY,
    ESCALATE,
    AnswerabilityVerdict,
    AnswerabilityVerifier,
    build_answerability_verifier,
    verdict_to_dict,
)
from app.rag.base import RetrieveResult
from app.rag.routing.classifier import (
    DECOMPOSE,
    HYBRID,
    MULTIHOP,
    MULTI_RECALL,
    REWRITE,
    VECTOR,
    QueryClassifier,
    RouteDecision,
    build_classifier,
)
from app.rag.retrieval.context_compress import ContextCompressor, build_compressor
from app.rag.routing.deictic_resolver import DeicticResolver, build_deictic_resolver
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.iterative_retrieval import (
    MultiHopEvent,
    MultiHopRetriever,
    build_multi_hop_retriever,
    hop_to_dict,
    plan_to_dict,
    verify_to_dict,
)
from app.rag.schemes.advanced import AdvancedRagScheme
from app.rag.routing.query_decompose import QueryDecomposer, build_decomposer


@dataclass
class ModuleCall:
    """执行计划中的一次模块调用：模块名 + 该次调用的参数。"""

    name: str  # rewrite / decompose / search / hybrid_search / multi_recall / multi_hop / rerank / compress
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """调度层输出的执行计划：明确哪些模块、什么顺序、生成策略。"""

    need_retrieval: bool  # 要不要检索
    pre_retrieval: list[ModuleCall] = field(default_factory=list)  # 预处理模块调用链
    retrieval: list[ModuleCall] = field(default_factory=list)  # 检索模块调用链
    post_retrieval: list[ModuleCall] = field(default_factory=list)  # 后处理模块调用链
    generation_strategy: str = "citation"  # 生成策略（direct / citation / comparison）


class ModularRagScheme(AdvancedRagScheme):
    """Modular RAG：前置语义路由 → 执行计划 → 按计划动态编排模块执行。"""

    id: str = "modular"
    name: str = "模块化 RAG"
    description: str = "语义路由 + 执行计划编排（可插拔模块组合）"
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
        classifier: QueryClassifier | None = None,
        decomposer: QueryDecomposer | None = None,
        compressor: ContextCompressor | None = None,
        max_hops: int = 3,
        multi_hop: MultiHopRetriever | None = None,
        deictic: DeicticResolver | None = None,
        answerability: AnswerabilityVerifier | None = None,
    ):
        super().__init__(
            embeddings,
            store,
            top_k,
            rewrite_variants=rewrite_variants,
            rerank_model=rerank_model,
            rewriter=rewriter,
            reranker=reranker,
        )
        self.classifier = classifier if classifier is not None else build_classifier()
        self.decomposer = decomposer if decomposer is not None else build_decomposer()
        self.compressor = compressor if compressor is not None else build_compressor()
        self.max_hops = max_hops
        self.multi_hop = multi_hop if multi_hop is not None else build_multi_hop_retriever()
        self.deictic = deictic if deictic is not None else build_deictic_resolver()
        self.answerability = (
            answerability if answerability is not None else build_answerability_verifier()
        )

    # ---- 调度层：路由决策 → 执行计划 ----

    def _build_plan(self, decision: RouteDecision) -> ExecutionPlan:
        """把五维路由决策映射为执行计划（模块可组合、可跳过）。"""
        pre: list[ModuleCall] = []
        retrieval: list[ModuleCall] = []
        post: list[ModuleCall] = []
        if not decision.retrieval_need:
            return ExecutionPlan(
                need_retrieval=False,
                pre_retrieval=pre,
                retrieval=retrieval,
                post_retrieval=post,
                generation_strategy=decision.generation_mode,
            )
        # 预处理：按复杂度挂载改写/分解模块（多跳的规划-执行-验证由 multi_hop 模块自包含完成）
        if decision.complexity == DECOMPOSE:
            pre.append(ModuleCall("decompose"))
        elif decision.complexity == REWRITE:
            pre.append(ModuleCall("rewrite"))
        # 检索：多跳走迭代检索模块，否则按检索策略选择模块
        if decision.complexity == MULTIHOP:
            retrieval.append(ModuleCall("multi_hop", params={"max_hops": self.max_hops}))
            post.append(ModuleCall("rerank"))  # 多跳合并命中后必须精排压噪
        elif decision.retrieval_mode == MULTI_RECALL:
            retrieval.append(ModuleCall("multi_recall"))
            post.append(ModuleCall("rerank"))  # 多路宽召回后必须精排压噪
        elif decision.retrieval_mode == HYBRID:
            retrieval.append(ModuleCall("hybrid_search"))
        else:
            retrieval.append(ModuleCall("search"))
        # 后处理：多路/混合/多跳召回后压缩上下文噪声（单次向量检索噪声有限，不压缩）
        if decision.retrieval_mode in (MULTI_RECALL, HYBRID):
            post.append(ModuleCall("compress"))
        return ExecutionPlan(
            need_retrieval=True,
            pre_retrieval=pre,
            retrieval=retrieval,
            post_retrieval=post,
            generation_strategy=decision.generation_mode,
        )

    @staticmethod
    def _recall_k(k: int) -> int:
        """宽召回候选数：多路/多查询召回需要比最终 Top-K 更宽的候选集。"""
        return max(k * 3, 9)

    def _collect(
        self, query: str, sub_queries: list[str], retrieval: list[ModuleCall], k: int
    ) -> list[dict[str, Any]]:
        """按检索模块对每个（子）查询召回，多路/多查询结果经 RRF 融合去重。

        各路分数体系不同（纯向量余弦分 vs 混合检索内部的 RRF 分），不能直接取最大值，
        统一用倒数排名融合（只依赖排名位置、跨路可比），出现在越多路的文档融合分越高。
        """
        ranked: list[list[dict[str, Any]]] = []
        for sq in sub_queries:
            for mod in retrieval:
                if mod.name == "search":
                    ranked.append(self.store.search(sq, self._recall_k(k)))
                elif mod.name == "hybrid_search":
                    ranked.append(self.store.hybrid_search(sq, self._recall_k(k)))
                elif mod.name == "multi_recall":
                    ranked.append(self.store.search(sq, self._recall_k(k)))
                    ranked.append(self.store.hybrid_search(sq, self._recall_k(k)))
                else:
                    continue
        return reciprocal_rank_fusion(ranked)

    def _apply_post(
        self,
        query: str,
        hits: list[dict[str, Any]],
        post: list[ModuleCall],
        k: int,
        keep: int | None = None,
    ) -> tuple[list[dict[str, Any]], bool, dict[str, int] | None]:
        """按后处理模块链处理命中：重排取 keep 条、压缩去噪；返回 (hits, reranked, compress_metrics)。

        keep：最终保留的命中数（默认 k）。多跳链式证据跨多个步骤（申请/审批/发票/报销/时限…），
        按实际跳数放大 keep，避免只保留 top_k 导致链式上下文缺失。
        """
        keep = keep or k
        reranked = False
        compress_metrics: dict[str, int] | None = None
        for mod in post:
            if mod.name == "rerank":
                hits = self.reranker.rerank(query, hits)[:keep]
                reranked = True
            elif mod.name == "compress":
                hits, compress_metrics = self.compressor.compress(query, hits, keep)
        return hits, reranked, compress_metrics

    # ---- 编排执行：同步（非流式/测试） ----

    def _escalate(self, plan: ExecutionPlan, verdict) -> ExecutionPlan | None:
        """答案充分性不足时的有界升级：按验证器建议把当前计划升一级（返回 None=不再升级）。

        升级阶梯（一次只升一级，有界防死循环/控成本）：
        - 已是最全路径（多跳）→ 不升级；
        - 已多路召回 → 升到多跳（多路仍不足多因缺链式中间环节）；
        - 单次/混合检索 → 升到多路召回（或验证器判定的多跳）。
        """
        if not plan.need_retrieval:
            return None
        if any(m.name == "multi_hop" for m in plan.retrieval):
            return None
        if any(m.name == "multi_recall" for m in plan.retrieval):
            retrieval = [ModuleCall("multi_hop", params={"max_hops": self.max_hops})]
            post = [ModuleCall("rerank")]
        elif getattr(verdict, "escalate_to", None) == "multihop":
            retrieval = [ModuleCall("multi_hop", params={"max_hops": self.max_hops})]
            post = [ModuleCall("rerank")]
        else:
            retrieval = [ModuleCall("multi_recall")]
            post = [ModuleCall("rerank"), ModuleCall("compress")]
        return ExecutionPlan(
            need_retrieval=True,
            pre_retrieval=[],  # 多路/多跳自含召回与规划，无需再改写/分解
            retrieval=retrieval,
            post_retrieval=post,
            generation_strategy=plan.generation_strategy,
        )

    def _run_plan(self, query: str, plan: ExecutionPlan, k: int) -> RetrieveResult:
        """按执行计划执行一轮检索（预处理 → 召回 → 后处理），返回完整检索结果。"""
        if not plan.need_retrieval:
            return RetrieveResult(query=query, hits=[])
        sub_queries = [query]
        rewrites: list[str] = []
        decomposed: list[str] = []
        for mod in plan.pre_retrieval:
            if mod.name == "rewrite":
                rewrites = self.rewriter.rewrite(query)
                sub_queries = rewrites or [query]
            elif mod.name == "decompose":
                decomposed = self.decomposer.decompose(query)
                sub_queries = decomposed or [query]
        hits, hops, multihop_plan, verification = self._recall(query, sub_queries, plan.retrieval, k)
        # 多跳链式证据按实际检索跳数放大保留数（覆盖复用跳不计入；每条链一环的证据都该保留）
        retrieved_hops = sum(1 for h in hops if not h.get("skipped"))
        keep = k * retrieved_hops if retrieved_hops else k
        hits, reranked, compress_metrics = self._apply_post(query, hits, plan.post_retrieval, k, keep)
        return RetrieveResult(
            query=query,
            hits=hits,
            rewrites=rewrites,
            reranked=reranked,
            decomposed=decomposed,
            compressed=compress_metrics,
            hops=hops,
            plan=multihop_plan,
            verification=verification,
        )

    def _execute_plan(self, query: str, plan: ExecutionPlan, k: int) -> RetrieveResult:
        """同步执行执行计划，返回完整检索结果；检索后过答案充分性验证闸门（不足则有界升级 1 轮）。"""
        if not plan.need_retrieval:
            return RetrieveResult(query=query, hits=[])
        result = self._run_plan(query, plan, k)
        verdict = self.answerability.verify(result.query, result.hits)
        result.answerability = verdict_to_dict(verdict)
        if not verdict.answerable and verdict.recommendation == ESCALATE:
            escalated = self._escalate(plan, verdict)
            if escalated is not None:
                result = self._run_plan(query, escalated, k)
                final = self.answerability.verify(result.query, result.hits)
                result.answerability = verdict_to_dict(final)
                return result
        if not verdict.answerable:
            # 已是最全路径/升级不可行仍不足 → 如实上报缺口（追问澄清交给生成层/前端）
            result.answerability = verdict_to_dict(
                AnswerabilityVerdict(
                    answerable=False,
                    missing_facts=verdict.missing_facts,
                    recommendation=CLARIFY,
                )
            )
        return result

    def _recall(
        self, query: str, sub_queries: list[str], retrieval: list[ModuleCall], k: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
        """按检索模块召回：multi_hop 走规划-执行-验证检索（含逐跳/计划/验证记录），其余走 _collect。

        返回 (合并命中, 逐跳记录 dict 列表, 计划 dict, 验证 dict)，非多跳路径 plan/verification 为 None。
        """
        for mod in retrieval:
            if mod.name == "multi_hop":
                result = self.multi_hop.retrieve(
                    query,
                    self.store,
                    k,
                    mod.params.get("max_hops", self.max_hops),
                    self._recall_k(k),
                )
                hops = [hop_to_dict(h) for h in result.hops]
                return (
                    result.hits,
                    hops,
                    plan_to_dict(result.plan),
                    verify_to_dict(result.verification),
                )
        return self._collect(query, sub_queries, retrieval, k), [], None, None

    def retrieve_full(self, query: str, top_k: int | None = None, context: str | None = None) -> RetrieveResult:
        """同步完整检索结果：先指代消解，再路由，再按执行计划动态编排。"""
        k = top_k or self.top_k
        resolved = self.deictic.resolve(query, context) or query
        decision = self.classifier.classify(resolved)
        plan = self._build_plan(decision)
        return self._execute_plan(resolved, plan, k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.retrieve_full(query, top_k).hits

    # ---- 编排执行：异步流式（前端 SSE 事件） ----

    async def _astream_plan(self, query: str, plan: ExecutionPlan, k: int):
        """按执行计划流式执行一轮检索：预处理 → 召回（逐跳 multi_hop）→ 后处理，逐事件下发。

        yield rewrite / decompose / multi_hop_plan / 逐跳 multi_hop / multi_hop_verify /
        retrieve / compress 事件；retrieve 事件携带本轮最终命中（调用方据此做充分性验证）。
        """
        sub_queries = [query]
        rewrites: list[str] = []
        decomposed: list[str] = []
        for mod in plan.pre_retrieval:
            if mod.name == "rewrite":
                rewrites = self.rewriter.rewrite(query)
                if rewrites:
                    yield {"type": "rewrite", "query": query, "scheme": self.id, "rewrites": rewrites}
                sub_queries = rewrites or [query]
            elif mod.name == "decompose":
                decomposed = self.decomposer.decompose(query)
                if decomposed:
                    yield {
                        "type": "decompose",
                        "query": query,
                        "scheme": self.id,
                        "sub_queries": decomposed,
                    }
                sub_queries = decomposed or [query]
        # 检索：multi_hop 模块按「规划-执行-验证」逐事件流式产出
        # （先 plan，再逐跳 multi_hop，最后 verify），其余模块一次性宽召回后经 RRF 融合
        # （保持 classify/rewrite/decompose 事件先行的顺序）。
        keep = k
        multi_hop_mod = next((m for m in plan.retrieval if m.name == "multi_hop"), None)
        if multi_hop_mod is not None:
            hit_lists: list[list[dict[str, Any]]] = []
            hop_index = 0
            retrieved = 0
            async for ev in self.multi_hop.astream_retrieve(
                query,
                self.store,
                k,
                multi_hop_mod.params.get("max_hops", self.max_hops),
                self._recall_k(k),
            ):
                if isinstance(ev, MultiHopEvent):
                    if ev.kind == "plan":
                        yield {
                            "type": "multi_hop_plan",
                            "query": query,
                            "scheme": self.id,
                            "plan": plan_to_dict(ev.plan),
                        }
                    elif ev.kind == "hop" and ev.hop is not None:
                        hop_index += 1
                        if not ev.hop.skipped:
                            hit_lists.append(ev.hop.hits)
                            retrieved += 1
                        yield {
                            "type": "multi_hop",
                            "query": query,
                            "scheme": self.id,
                            "index": hop_index,
                            "hop": hop_to_dict(ev.hop),
                        }
                    elif ev.kind == "verify":
                        yield {
                            "type": "multi_hop_verify",
                            "query": query,
                            "scheme": self.id,
                            "verification": verify_to_dict(ev.verification),
                        }
                else:
                    # 兼容贪心迭代检索器（LLM/RuleMultiHopRetriever）的旧事件协议
                    hop_index += 1
                    retrieved += 1
                    hit_lists.append(ev.hits)
                    yield {
                        "type": "multi_hop",
                        "query": query,
                        "scheme": self.id,
                        "index": hop_index,
                        "hop": {
                            "query": ev.query,
                            "hits": ev.hits,
                            "next_query": ev.next_query,
                        },
                    }
            hits = reciprocal_rank_fusion(hit_lists)
            # 多跳链式证据按实际检索跳数放大最终保留数（覆盖复用跳不计入），
            # 避免只留 top_k 导致链式上下文缺失
            keep = k * retrieved if retrieved else k
        else:
            hits = await asyncio.to_thread(self._collect, query, sub_queries, plan.retrieval, k)
        hits, reranked, compress_metrics = await asyncio.to_thread(
            self._apply_post, query, hits, plan.post_retrieval, k, keep
        )
        yield {
            "type": "retrieve",
            "query": query,
            "scheme": self.id,
            "hits": hits,
            "reranked": reranked,
        }
        # 仅当真正发生了压缩（去重/截断）才下发 compress 事件，避免无意义徽标
        if compress_metrics and (
            compress_metrics["kept"] < compress_metrics["original"]
            or compress_metrics["truncated"] > 0
        ):
            yield {
                "type": "compress",
                "query": query,
                "scheme": self.id,
                "metrics": compress_metrics,
            }

    async def astream(self, query: str, top_k: int | None = None, context: str | None = None):
        """异步流式：先指代消解，再产出路由事件，再按计划产出 rewrite / decompose / 逐跳 multi_hop / retrieve / compress 事件；
        检索后过答案充分性验证闸门（不足则有界升级 1 轮，再产出 answerability 事件）。

        路由/召回/重排/压缩为同步调用（向量库/模型同步 HTTP），放线程池执行；
        多跳迭代检索逐跳流式产出——每完成一跳立即下发一个 multi_hop 事件（index 递增），
        而非一次性把全部跳合并返回，保证 classify / rewrite / decompose 事件先经 SSE 下发，再逐跳、最后 retrieve。
        """
        k = top_k or self.top_k
        resolved = self.deictic.resolve(query, context) or query
        if resolved != query:
            yield {
                "type": "rewrite",
                "query": query,
                "scheme": self.id,
                "rewrites": [resolved],
                "reason": "指代消解",
            }
        query = resolved
        decision = self.classifier.classify(query)
        plan = self._build_plan(decision)
        yield {
            "type": "classify",
            "query": query,
            "scheme": self.id,
            "retrieval_need": decision.retrieval_need,
            "retrieval_mode": decision.retrieval_mode,
            "complexity": decision.complexity,
            "generation_mode": decision.generation_mode,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }
        if not plan.need_retrieval:
            return
        # 第一轮执行 + 答案充分性验证
        current_hits: list[dict[str, Any]] = []
        async for ev in self._astream_plan(query, plan, k):
            if ev["type"] == "retrieve":
                current_hits = ev["hits"]
            yield ev
        verdict = self.answerability.verify(query, current_hits)
        # 与同步 _execute_plan 保持一致：仅当验证建议升级（escalate）且有更高路径时
        # 才有界升级 1 轮；建议澄清（clarify，如缺指代/信息确实缺失）不升级、直接如实上报，
        # 避免「该追问却升级检索后越权作答」。
        if not verdict.answerable and verdict.recommendation == ESCALATE:
            escalated = self._escalate(plan, verdict)
            if escalated is not None:
                # 检索不足且建议升级 → 有界升级 1 轮（前端可见二次检索），再验证
                yield {
                    "type": "answerability",
                    "query": query,
                    "scheme": self.id,
                    "verdict": verdict_to_dict(verdict),
                    "escalated": False,
                }
                current_hits = []
                async for ev in self._astream_plan(query, escalated, k):
                    if ev["type"] == "retrieve":
                        current_hits = ev["hits"]
                    yield ev
                final = self.answerability.verify(query, current_hits)
                yield {
                    "type": "answerability",
                    "query": query,
                    "scheme": self.id,
                    "verdict": verdict_to_dict(final),
                    "escalated": True,
                }
                return
            # 升级不可行（已是最全路径）仍不足 → 如实上报缺口（追问澄清交给生成层/前端）
            verdict = AnswerabilityVerdict(
                answerable=False,
                missing_facts=verdict.missing_facts,
                recommendation=CLARIFY,
            )
        elif not verdict.answerable:
            # 验证建议澄清（信息确实缺失，如缺指代/缺关键事实）→ 不升级，如实上报缺口
            verdict = AnswerabilityVerdict(
                answerable=False,
                missing_facts=verdict.missing_facts,
                recommendation=CLARIFY,
            )
        yield {
            "type": "answerability",
            "query": query,
            "scheme": self.id,
            "verdict": verdict_to_dict(verdict),
            "escalated": False,
        }
