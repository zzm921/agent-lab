"""模块化 RAG 方案：前置语义路由（Query Router）+ 执行计划（Execution Plan）编排。

对应 Modular RAG 企业级架构的「调度层 + 预处理/检索/后处理模块组」：
把检索链路拆成可插拔模块，由路由决策（RouteDecision）映射为执行计划（ExecutionPlan），
再按计划动态编排执行（模块可组合、可跳过，正是 Modular RAG 的核心价值）——

- 预处理模块组：查询改写（rewrite） / 查询分解（decompose）；
- 检索模块组：向量检索（search） / 混合检索（hybrid_search） / 多路召回（multi_recall）；
- 后处理模块组：重排（rerank） / 上下文压缩（compress，含语义去重）；
- 生成策略：direct / citation / comparison（runner 按路由产出的 generation_mode 定制注入指令）。

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
import logging
import re
from concurrent.futures import ThreadPoolExecutor
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
    expand_scale_query,
    hop_to_dict,
    plan_to_dict,
    verify_to_dict,
)
from app.rag.schemes.advanced import AdvancedRagScheme
from app.rag.routing.query_decompose import QueryDecomposer, build_decomposer
from app.rag.routing.query_hyde import HydeExpander, build_hyde

logger = logging.getLogger(__name__)

# ---- 路由 target（D6）→ 定向补召回的卷名白名单 ----
# 卷名 = 语料源文件 `# ` 一级标题（= metadata.volume，与 KNOWLEDGE_DOCS 键一致）。
# 用途：路由判定 target 后，检索层在白名单卷内**额外**召回一路（非替换全库检索），
# 保证「人名档案 / FAQ / 版本对比」等定向块与制度条款块同时进入候选，经重排竞争 top_k。
_TARGET_VOLUME_FILTERS: dict[str, tuple[str, ...]] = {
    "profile": ("卷十三·附录 全员权益明细档案",),
    "faq": (
        "卷十 员工常见问题问答库（FAQ）",
        "卷三十九 FAQ 补充问答（第三编）",
        "卷四十四 员工常见问题问答库（第四编）",
        "卷四十九 员工常见问题问答库（第五编）",
    ),
    "case": (
        "卷十一 典型案例与判例库",
        "卷四十 案例判例库（第三编）",
        "卷四十五 案例判例库（第四编）",
        "卷五十 案例判例库（第五编）",
    ),
    "scene": (
        "卷二十一 常见业务场景处理手册",
        "卷四十一 常见业务场景处理手册（第二编）",
        "卷四十六 常见业务场景处理手册（第三编）",
    ),
    "card": ("卷二十二 制度速查知识卡片", "卷四十七 制度速查知识卡片（第三编）"),
    "sop": ("卷十二 标准作业流程（SOP）与表单模板",),
    "version": ("卷九 制度版本演进与对比",),
    "duty": ("卷三十八 岗位职责说明书",),
}

# ---- 定向补召回防挤占（同模板块多样性截断） ----
# 档案/FAQ 等定向卷内同模板块极多（如「卷十三·附录」逐人分块，模板句相同仅人名/数字不同），
# 若整路进入 RRF 融合，会成群挤占 top_k 配额，把「档案 + 制度条款」组合证据中的另一类挤出候选。
# 对策：定向路结果先做贪心多样性选择——与已保留块 2 字词相对重叠过高的同模板块丢弃，
# 且进入融合的块数设上限（真证据在卷内排名靠前，截断不会伤及目标块）。
_VOLUME_ROUTE_MAX = 3  # 定向路参与融合的块数上限
_VOLUME_ROUTE_OVERLAP = 0.55  # 同模板块判定：2 字词相对重叠率阈值（交集/较短集合）

# ---- 跨轮 seed 复用护栏（保守方案） ----
# 上一轮已验证的检索命中可作为本轮「候选证据」复用（省掉重复检索 + 覆盖复用的跳）；
# 但它只是多一路 RRF 候选，不注入查询文本、不占当前轮召回配额。为防止「上次不准」传导伤害：
# ① 只收高置信命中（分数低于阈值的一律丢弃——幻觉/弱命中通常分数低）；
# ② 与当前查询无共现 2 字词的一律丢弃（跨主题噪音过滤）；
# ③ 最多取前 N 条（budget 保护）；
# ④ 覆盖跳过仍需验证器确认目标事实出现在证据中（不准的 seed 无法覆盖正确步骤 → 会重新检索）。
_SEED_MIN_SCORE = 0.5  # 上轮命中分数门槛（store/重排均为 0-1 余弦/词法分）
_SEED_MAX = 5  # 跨轮 seed 候选上限


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
    volume_filter: tuple[str, ...] | None = None  # 定向补召回卷名白名单（target 映射，None=不过滤）


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
        hyde: HydeExpander | None = None,
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
            hyde=hyde,
        )
        self.classifier = classifier if classifier is not None else build_classifier()
        self.decomposer = decomposer if decomposer is not None else build_decomposer()
        self.compressor = (
            compressor if compressor is not None else build_compressor(embeddings=embeddings)
        )
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
        # 定向补召回：路由判定的 target 映射为卷名白名单，检索层在白名单卷内额外召回一路
        return ExecutionPlan(
            need_retrieval=True,
            pre_retrieval=pre,
            retrieval=retrieval,
            post_retrieval=post,
            generation_strategy=decision.generation_mode,
            volume_filter=_TARGET_VOLUME_FILTERS.get(decision.target),
        )

    @staticmethod
    def _recall_k(k: int) -> int:
        """宽召回候选数：多路/多查询召回需要比最终 Top-K 更宽的候选集。"""
        return max(k * 3, 9)

    @staticmethod
    def _terms2(text: str) -> set[str]:
        """提取中文相邻 2 字词（重叠窗口）：用于跨轮 seed 相关性的轻量粗判。"""
        seg = re.findall(r"[\u4e00-\u9fff]+", text)
        return {s[i : i + 2] for s in seg for i in range(len(s) - 1)}

    @staticmethod
    def _diversify(
        hits: list[dict[str, Any]],
        max_items: int = _VOLUME_ROUTE_MAX,
        max_overlap: float = _VOLUME_ROUTE_OVERLAP,
    ) -> list[dict[str, Any]]:
        """定向路卷内多样性截断：同模板块（逐人档案/同类 FAQ 条目）只留代表。

        贪心保留与已选块 2 字词相对重叠率（交集/较短集合）不超过阈值的块；
        模板句相同的档案块（仅人名/数字不同）重叠率接近 1，只会留卷内排名最高的代表，
        目标人块与制度条款块字面重叠低、互不排斥。进入融合的块数上限 max_items。
        """
        kept: list[dict[str, Any]] = []
        kept_terms: list[set[str]] = []
        for hit in hits:
            terms = ModularRagScheme._terms2(hit.get("text") or "")
            if not terms:
                continue
            if any(
                len(terms & prev) / max(1, min(len(terms), len(prev))) > max_overlap
                for prev in kept_terms
            ):
                continue
            kept.append(hit)
            kept_terms.append(terms)
            if len(kept) >= max_items:
                break
        return kept

    @staticmethod
    def _cross_turn_seed(query: str, prev_hits: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """跨轮 seed 复用闸门（保守方案）：从上一轮已验证命中里筛出本轮可用候选证据。

        - 分数门槛：score < _SEED_MIN_SCORE 丢弃（已按分降序，一旦低于即整体截断）；
        - 相关性把关：与当前查询无共现 2 字词的一律丢弃（防止跨主题噪音混入）；
        - 数量上限：最多取 _SEED_MAX 条（budget 保护，不挤占当前轮召回配额）。

        返回的 seed 只作「多一路 RRF 候选 + 覆盖检测既有基础」，不注入查询文本；
        空列表 = 本轮无可用 seed，照常全新检索。
        """
        if not prev_hits:
            return []
        q_terms = ModularRagScheme._terms2(query)
        kept: list[dict[str, Any]] = []
        for hit in sorted(prev_hits, key=lambda h: h.get("score") or 0.0, reverse=True):
            if len(kept) >= _SEED_MAX:
                break
            if (hit.get("score") or 0.0) < _SEED_MIN_SCORE:
                break  # 已按分数降序，其后只会更低
            text = hit.get("text") or ""
            if q_terms and not (q_terms & ModularRagScheme._terms2(text)):
                continue
            kept.append(hit)
        return kept

    def _collect(
        self,
        query: str,
        sub_queries: list[str],
        retrieval: list[ModuleCall],
        k: int,
        seed_hits: list[dict[str, Any]] | None = None,
        volume_filter: tuple[str, ...] | None = None,
        hyde_out: dict | None = None,  # 可选输出通道：HyDE 真触发时回填 {"doc","recall"}（流式 hyde 事件用）
    ) -> list[dict[str, Any]]:
        """按检索模块对每个（子）查询召回，多路/多查询结果经 RRF 融合去重。

        各路分数体系不同（纯向量余弦分 vs 混合检索内部的 RRF 分），不能直接取最大值，
        统一用倒数排名融合（只依赖排名位置、跨路可比），出现在越多路的文档融合分越高。
        seed_hits：跨轮/升级既有证据，作为额外一路候选参与融合（已按分数降序、限量），
        不占当前轮召回配额；无关或弱命中在 _cross_turn_seed 闸门已被滤掉。
        """
        recall_k = self._recall_k(k)
        # 多路召回并行化：各子查询 × 各检索模式（向量/混合）互不依赖，放线程池并发执行，
        # 召回延迟从「各路耗时之和」降为「最慢一路」；HyDE 的 LLM 假想文档生成同时在本线程进行，
        # 生成完再并入线程池做一路稠密召回——避免串行 for 循环累加各路的向量库/网络往返。
        calls: list[tuple] = []
        for sq in sub_queries:
            for mod in retrieval:
                if mod.name == "search":
                    calls.append((self.store.search, sq, recall_k))
                elif mod.name == "hybrid_search":
                    calls.append((self.store.hybrid_search, expand_scale_query(sq), recall_k))
                elif mod.name == "multi_recall":
                    calls.append((self.store.search, sq, recall_k))
                    # 混合路对人数/规模意图查询追加规模表规范词（与 _multi_recall 对齐）
                    calls.append((self.store.hybrid_search, expand_scale_query(sq), recall_k))
                else:
                    continue
        ranked: list[list[dict[str, Any]]] = []
        if seed_hits:
            ranked.append(seed_hits)
        hyde_fut: Any = None
        hyde_doc = ""
        with ThreadPoolExecutor(max_workers=max(1, min(len(calls), 8))) as ex:
            futures = [ex.submit(fn, *args) for fn, *args in calls]
            # 定向补召回与各路并发：路由 target 映射的卷内额外一路（Qdrant filter 精确卷名），
            # 保证档案/FAQ/版本对比等定向块与全库召回块同台竞争（rrf 融合），不替换全库检索；
            # 结果先经卷内多样性截断——同模板块只留代表，防止成群挤占 top_k 配额
            volume_future = (
                ex.submit(self.store.search, query, recall_k, volume_filter)
                if volume_filter
                else None
            )
            # HyDE：用 LLM 生成的假想答案文档做一路稠密 doc-space 召回（规则回退时为原查询，跳过）；
            # 生成调用与上述各路召回并发重叠
            hyde_doc = self.hyde.expand(query)
            if hyde_doc and hyde_doc != query:
                hyde_fut = ex.submit(self.store.search, hyde_doc, recall_k)
                futures.append(hyde_fut)
            ranked.extend(f.result() for f in futures)
            if volume_future is not None:
                ranked.append(self._diversify(volume_future.result()))
        # 把 HyDE 一路的信息经可选输出通道回传（供流式 hyde 事件展示；同步路径不传则忽略）
        if hyde_fut is not None and hyde_out is not None:
            hyde_out["doc"] = hyde_doc
            hyde_out["recall"] = len(hyde_fut.result())
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
            volume_filter=plan.volume_filter,  # 升级路径继承定向补召回
        )

    def _run_plan(
        self,
        query: str,
        plan: ExecutionPlan,
        k: int,
        seed_hits: list[dict[str, Any]] | None = None,
        memory: str | None = None,
    ) -> RetrieveResult:
        """按执行计划执行一轮检索（预处理 → 召回 → 后处理），返回完整检索结果。

        seed_hits：既有证据（充分性验证升级前的首轮命中），升级多跳时传入执行器
        作覆盖检测与合并的基础——已覆盖步骤复用跳过、不重复检索（增量补缺）。
        memory：L2 主动语义召回的用户记忆块（背景参考），透传给查询改写辅助个性化改写。
        """
        if not plan.need_retrieval:
            return RetrieveResult(query=query, hits=[])
        sub_queries = [query]
        rewrites: list[str] = []
        decomposed: list[str] = []
        for mod in plan.pre_retrieval:
            if mod.name == "rewrite":
                rewrites = self.rewriter.rewrite(query, memory)
                logger.info("[modular] 执行：查询改写 → %d 个变体 %s", len(rewrites), rewrites)
                sub_queries = rewrites or [query]
            elif mod.name == "decompose":
                decomposed = self.decomposer.decompose(query)
                logger.info("[modular] 执行：查询分解 → %d 个子查询 %s", len(decomposed), decomposed)
                sub_queries = decomposed or [query]
        hits, hops, multihop_plan, verification = self._recall(
            query, sub_queries, plan.retrieval, k, seed_hits=seed_hits,
            volume_filter=plan.volume_filter,
        )
        # 多跳链式证据按实际检索跳数放大保留数（覆盖复用跳不计入；每条链一环的证据都该保留）
        retrieved_hops = sum(1 for h in hops if not h.get("skipped"))
        keep = k * retrieved_hops if retrieved_hops else k
        if seed_hits:
            # 升级多跳并入的种子证据也是链路一环：为种子与新召回预留保留位，避免被重排截断
            keep = max(keep, len(seed_hits) + k)
        hits, reranked, compress_metrics = self._apply_post(query, hits, plan.post_retrieval, k, keep)
        hits = self._resolve_parents(hits)  # 重排/压缩后子块命中回填父块全文（含 answerability 验证基准）
        logger.info(
            "[modular] 执行：检索完成 → 命中 %d 条（reranked=%s%s）",
            len(hits),
            reranked,
            f"，压缩 {compress_metrics['original']}→{compress_metrics['kept']}" if compress_metrics else "",
        )
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

    def _execute_plan(
        self,
        query: str,
        plan: ExecutionPlan,
        k: int,
        seed_hits: list[dict[str, Any]] | None = None,
        memory: str | None = None,
    ) -> RetrieveResult:
        """同步执行执行计划，返回完整检索结果；检索后过答案充分性验证闸门（不足则有界升级 1 轮）。

        seed_hits：跨轮既有证据（上一轮已验证命中经 _cross_turn_seed 闸门筛选），
        首轮即作为候选证据参与召回融合与覆盖检测（省重复检索）。
        memory：L2 主动语义召回的用户记忆块（背景参考），透传给查询改写辅助个性化改写。
        """
        if not plan.need_retrieval:
            return RetrieveResult(query=query, hits=[])
        result = self._run_plan(query, plan, k, seed_hits=seed_hits, memory=memory)
        verdict = self.answerability.verify(result.query, result.hits)
        result.answerability = verdict_to_dict(verdict)
        logger.info(
            "[modular] 执行：答案充分性验证 → answerable=%s（%s）",
            verdict.answerable,
            verdict.recommendation,
        )
        if not verdict.answerable and verdict.recommendation == ESCALATE:
            escalated = self._escalate(plan, verdict)
            if escalated is not None:
                logger.info(
                    "[modular] 执行：检索不足 → 升级为 %s 路径重试",
                    [m.name for m in escalated.retrieval],
                )
                # 升级重跑时把首轮命中作为既有证据传入多跳执行器（seed）：已覆盖步骤复用跳过、
                # 不重复检索，实现增量补缺而非整轮重跑
                result = self._run_plan(query, escalated, k, seed_hits=result.hits, memory=memory)
                final = self.answerability.verify(result.query, result.hits)
                result.answerability = verdict_to_dict(final)
                logger.info(
                    "[modular] 执行：升级后验证 → answerable=%s（%s）",
                    final.answerable,
                    final.recommendation,
                )
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
        self,
        query: str,
        sub_queries: list[str],
        retrieval: list[ModuleCall],
        k: int,
        seed_hits: list[dict[str, Any]] | None = None,
        volume_filter: tuple[str, ...] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
        """按检索模块召回：multi_hop 走规划-执行-验证检索（含逐跳/计划/验证记录），其余走 _collect。

        返回 (合并命中, 逐跳记录 dict 列表, 计划 dict, 验证 dict)，非多跳路径 plan/verification 为 None。
        seed_hits：既有证据（升级前的首轮命中），multi_hop 作第 0 路参与覆盖检测与合并（增量补缺）。
        """
        for mod in retrieval:
            if mod.name == "multi_hop":
                result = self.multi_hop.retrieve(
                    query,
                    self.store,
                    k,
                    mod.params.get("max_hops", self.max_hops),
                    self._recall_k(k),
                    seed_hits=seed_hits,
                )
                hops = [hop_to_dict(h) for h in result.hops]
                return (
                    result.hits,
                    hops,
                    plan_to_dict(result.plan),
                    verify_to_dict(result.verification),
                )
        return (
            self._collect(
                query, sub_queries, retrieval, k, seed_hits=seed_hits, volume_filter=volume_filter
            ),
            [],
            None,
            None,
        )

    def retrieve_full(
        self,
        query: str,
        top_k: int | None = None,
        context: str | None = None,
        seed_hits: list[dict[str, Any]] | None = None,
        memory: str | None = None,
    ) -> RetrieveResult:
        """同步完整检索结果：先指代消解，再路由，再按执行计划动态编排。

        seed_hits：跨轮既有证据（上一轮已验证命中，由 _cross_turn_seed 按分数/相关性过滤）。
        memory：L2 主动语义召回的用户记忆块（背景参考），供指代消解/语义路由参考（记忆先行）。
        """
        k = top_k or self.top_k
        resolved = self.deictic.resolve(query, context, memory) or query
        if resolved != query:
            logger.info("[modular] 执行：指代消解 %r → %r", query, resolved)
        seed = self._cross_turn_seed(resolved, seed_hits) if seed_hits else []
        if seed:
            logger.info("[modular] 执行：跨轮 seed 复用 → %d 条候选证据", len(seed))
        decision = self.classifier.classify(resolved, memory)
        plan = self._build_plan(decision)
        logger.info(
            "[modular] 执行：语义路由 → need=%s mode=%s complexity=%s generation=%s conf=%.2f（%s）",
            decision.retrieval_need,
            decision.retrieval_mode,
            decision.complexity,
            decision.generation_mode,
            decision.confidence,
            decision.reason,
        )
        logger.info(
            "[modular] 执行：执行计划 → pre=%s retrieval=%s post=%s strategy=%s",
            [m.name for m in plan.pre_retrieval],
            [m.name for m in plan.retrieval],
            [m.name for m in plan.post_retrieval],
            plan.generation_strategy,
        )
        if not plan.need_retrieval:
            logger.info("[modular] 执行：无需检索，直接生成")
        return self._execute_plan(resolved, plan, k, seed_hits=seed, memory=memory)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.retrieve_full(query, top_k).hits

    # ---- 编排执行：异步流式（前端 SSE 事件） ----

    async def _astream_plan(
        self,
        query: str,
        plan: ExecutionPlan,
        k: int,
        seed_hits: list[dict[str, Any]] | None = None,
        memory: str | None = None,
    ):
        """按执行计划流式执行一轮检索：预处理 → 召回（逐跳 multi_hop）→ 后处理，逐事件下发。

        yield rewrite / decompose / multi_hop_plan / 逐跳 multi_hop / multi_hop_verify /
        retrieve / compress 事件；retrieve 事件携带本轮最终命中（调用方据此做充分性验证）。
        seed_hits：既有证据（充分性验证升级前的首轮命中），升级多跳时作为第 0 路参与
        覆盖检测与合并——已覆盖的步骤复用跳过、不重复检索（增量补缺而非整轮重跑）。
        memory：L2 主动语义召回的用户记忆块（背景参考），透传给查询改写辅助个性化改写。
        """
        sub_queries = [query]
        rewrites: list[str] = []
        decomposed: list[str] = []
        for mod in plan.pre_retrieval:
            if mod.name == "rewrite":
                # 改写为同步 LLM 调用，放线程池避免阻塞事件循环（否则 rewrite 事件与后续事件攒在一起刷出）
                rewrites = await asyncio.to_thread(self.rewriter.rewrite, query, memory)
                if rewrites:
                    logger.info("[modular] 流式：查询改写 → %d 个变体 %s", len(rewrites), rewrites)
                    yield {"type": "rewrite", "query": query, "scheme": self.id, "rewrites": rewrites}
                sub_queries = rewrites or [query]
            elif mod.name == "decompose":
                decomposed = await asyncio.to_thread(self.decomposer.decompose, query)
                if decomposed:
                    logger.info("[modular] 流式：查询分解 → %d 个子查询 %s", len(decomposed), decomposed)
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
            # 既有种子证据作为第 0 路参与最终 RRF 合并：升级多跳时首轮命中不得丢失——
            # 覆盖复用的跳不带命中（hits=[]），若不并入种子，首轮已召回的关键证据
            # （如「张三→研发部、李雪→产品部」的映射）会在升级重跑后被丢弃，
            # 导致答案充分性验证只见人数、不见归属部门，误判「无法回答」。
            hit_lists: list[list[dict[str, Any]]] = [seed_hits] if seed_hits else []
            hop_index = 0
            retrieved = 0
            async for ev in self.multi_hop.astream_retrieve(
                query,
                self.store,
                k,
                multi_hop_mod.params.get("max_hops", self.max_hops),
                self._recall_k(k),
                seed_hits=seed_hits,
            ):
                if isinstance(ev, MultiHopEvent):
                    if ev.kind == "plan_running":
                        # 多跳规划（纯 LLM 阻塞调用）：先发「规划中」占位事件，规划完成后再填充计划
                        logger.info("[modular] 流式：多跳 → 规划中…")
                        yield {
                            "type": "multi_hop_plan",
                            "query": query,
                            "scheme": self.id,
                            "status": "running",
                        }
                    elif ev.kind == "plan":
                        plan_dict = plan_to_dict(ev.plan) or {}
                        logger.info(
                            "[modular] 流式：多跳 → 计划生成（%d 步）",
                            len(plan_dict.get("steps", [])),
                        )
                        yield {
                            "type": "multi_hop_plan",
                            "query": query,
                            "scheme": self.id,
                            "status": "done",
                            "plan": plan_dict,
                        }
                    elif ev.kind == "hop" and ev.hop is not None:
                        hop_index += 1
                        if not ev.hop.skipped:
                            hit_lists.append(ev.hop.hits)
                            retrieved += 1
                        logger.info(
                            "[modular] 流式：多跳 → 第 %d 跳%s（命中 %d 条）",
                            hop_index,
                            "（复用跳过）" if ev.hop.skipped else "",
                            len(ev.hop.hits),
                        )
                        yield {
                            "type": "multi_hop",
                            "query": query,
                            "scheme": self.id,
                            "index": hop_index,
                            "hop": hop_to_dict(ev.hop),
                        }
                    elif ev.kind == "verify":
                        vdict = verify_to_dict(ev.verification) or {}
                        logger.info(
                            "[modular] 流式：多跳 → 验证（covered=%d missing=%d）",
                            len(vdict.get("covered", [])),
                            len(vdict.get("missing", [])),
                        )
                        yield {
                            "type": "multi_hop_verify",
                            "query": query,
                            "scheme": self.id,
                            "verification": vdict,
                        }
                else:
                    # 兼容贪心迭代检索器（LLM/RuleMultiHopRetriever）的旧事件协议
                    hop_index += 1
                    retrieved += 1
                    hit_lists.append(ev.hits)
                    logger.info(
                        "[modular] 流式：多跳 → 第 %d 跳（命中 %d 条）",
                        hop_index,
                        len(ev.hits),
                    )
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
            if seed_hits:
                # 种子证据也是链路一环：为种子与新召回预留保留位，避免被重排截断
                keep = max(keep, len(seed_hits) + k)
        else:
            # HyDE 假想文档检索为 LLM 阻塞生成：先发 running 占位（转圈），完成后再发 done 就地填充
            yield {
                "type": "hyde",
                "query": query,
                "scheme": self.id,
                "status": "running",
            }
            hyde_out: dict = {}
            hits = await asyncio.to_thread(
                self._collect,
                query,
                sub_queries,
                plan.retrieval,
                k,
                seed_hits,
                plan.volume_filter,
                hyde_out,
            )
            yield {
                "type": "hyde",
                "query": query,
                "scheme": self.id,
                "status": "done",
                "fired": bool(hyde_out),
                "doc": hyde_out.get("doc"),
                "recall": hyde_out.get("recall"),
            }
            if seed_hits:
                # 种子证据与召回结果并存：为种子预留保留位，避免挤占新证据的 top_k 名额
                keep = max(keep, len(seed_hits) + k)
        hits, reranked, compress_metrics = await asyncio.to_thread(
            self._apply_post, query, hits, plan.post_retrieval, k, keep
        )
        hits = self._resolve_parents(hits)  # 重排/压缩后子块命中回填父块全文，注入 LLM 前完成
        logger.info(
            "[modular] 流式：检索完成 → 命中 %d 条（reranked=%s%s）",
            len(hits),
            reranked,
            f"，压缩 {compress_metrics['original']}→{compress_metrics['kept']}" if compress_metrics else "",
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
            logger.info(
                "[modular] 流式：上下文压缩 → %d→%d 条（截断 %d）",
                compress_metrics["original"],
                compress_metrics["kept"],
                compress_metrics["truncated"],
            )
            yield {
                "type": "compress",
                "query": query,
                "scheme": self.id,
                "metrics": compress_metrics,
            }

    async def astream(
        self,
        query: str,
        top_k: int | None = None,
        context: str | None = None,
        seed_hits: list[dict[str, Any]] | None = None,
        memory: str | None = None,
    ):
        """异步流式：先指代消解，再产出路由事件，再按计划产出 rewrite / decompose / 逐跳 multi_hop / retrieve / compress 事件；
        检索后过答案充分性验证闸门（不足则有界升级 1 轮，再产出 answerability 事件）。

        seed_hits：跨轮既有证据（上一轮已验证命中，由 _cross_turn_seed 按分数/相关性过滤），
        首轮即作为候选证据参与召回融合与覆盖检测；复用命中数 >0 时下发 seed_reuse 事件（可观测）。
        memory：L2 主动语义召回的用户记忆块（背景参考），供指代消解/改写/路由参考（记忆先行）。

        路由/改写/分解/召回/重排/压缩/充分性验证等均为同步调用（向量库/模型同步 HTTP），
        统一放线程池（asyncio.to_thread）执行、不阻塞事件循环——否则各事件会被攒到同一次
        SSE 刷出、在前端「同时」出现；多跳迭代检索逐跳流式产出——每完成一跳立即下发一个
        multi_hop 事件（index 递增），保证 classify / rewrite / decompose 事件先经 SSE 下发，再逐跳、最后 retrieve。
        """
        k = top_k or self.top_k
        # 指代消解为同步 LLM 调用，放线程池避免阻塞事件循环
        resolved = (await asyncio.to_thread(self.deictic.resolve, query, context, memory)) or query
        if resolved != query:
            logger.info("[modular] 流式：指代消解 %r → %r", query, resolved)
            yield {
                "type": "rewrite",
                "query": query,
                "scheme": self.id,
                "rewrites": [resolved],
                "reason": "指代消解",
            }
        query = resolved
        seed = self._cross_turn_seed(query, seed_hits) if seed_hits else []
        if seed:
            logger.info("[modular] 流式：跨轮 seed 复用 → %d 条候选证据", len(seed))
            yield {
                "type": "seed_reuse",
                "query": query,
                "scheme": self.id,
                "count": len(seed),
            }
        # 语义路由（纯 LLM 阻塞调用）：先发「路由中」占位事件让前端立即展示卡片，
        # 路由完成后再填充五维决策——与多跳逐跳流式保持一致的渐进呈现，而非跟首跳结果一起弹出。
        yield {
            "type": "classify",
            "query": query,
            "scheme": self.id,
            "status": "running",
        }
        decision = await asyncio.to_thread(self.classifier.classify, query, memory)
        plan = self._build_plan(decision)
        logger.info(
            "[modular] 流式：语义路由 → need=%s mode=%s complexity=%s generation=%s conf=%.2f（%s）",
            decision.retrieval_need,
            decision.retrieval_mode,
            decision.complexity,
            decision.generation_mode,
            decision.confidence,
            decision.reason,
        )
        logger.info(
            "[modular] 流式：执行计划 → pre=%s retrieval=%s post=%s strategy=%s",
            [m.name for m in plan.pre_retrieval],
            [m.name for m in plan.retrieval],
            [m.name for m in plan.post_retrieval],
            plan.generation_strategy,
        )
        yield {
            "type": "classify",
            "query": query,
            "scheme": self.id,
            "status": "done",
            "retrieval_need": decision.retrieval_need,
            "retrieval_mode": decision.retrieval_mode,
            "complexity": decision.complexity,
            "generation_mode": decision.generation_mode,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }
        if not plan.need_retrieval:
            logger.info("[modular] 流式：无需检索，直接生成")
            return
        # 第一轮执行 + 答案充分性验证（seed 为跨轮候选证据，参与召回融合与覆盖检测）
        current_hits: list[dict[str, Any]] = []
        async for ev in self._astream_plan(query, plan, k, seed_hits=seed, memory=memory):
            if ev["type"] == "retrieve":
                current_hits = ev["hits"]
            yield ev
        verdict = await asyncio.to_thread(self.answerability.verify, query, current_hits)
        logger.info(
            "[modular] 流式：答案充分性验证 → answerable=%s（%s）",
            verdict.answerable,
            verdict.recommendation,
        )
        # 与同步 _execute_plan 保持一致：仅当验证建议升级（escalate）且有更高路径时
        # 才有界升级 1 轮；建议澄清（clarify，如缺指代/信息确实缺失）不升级、直接如实上报，
        # 避免「该追问却升级检索后越权作答」。
        if not verdict.answerable and verdict.recommendation == ESCALATE:
            escalated = await asyncio.to_thread(self._escalate, plan, verdict)
            if escalated is not None:
                logger.info(
                    "[modular] 流式：检索不足 → 升级为 %s 路径重试",
                    [m.name for m in escalated.retrieval],
                )
                # 检索不足且建议升级 → 有界升级 1 轮（前端可见二次检索），再验证
                yield {
                    "type": "answerability",
                    "query": query,
                    "scheme": self.id,
                    "verdict": verdict_to_dict(verdict),
                    "escalated": False,
                }
                # 升级重跑时把首轮命中作为既有证据传入多跳执行器（seed）：已覆盖步骤复用跳过、
                # 不重复检索，实现增量补缺而非整轮重跑；current_hits 保留首轮命中作种子，
                # 由第二轮 retrieve 事件覆盖为合并后的最终命中（已含种子 + 新增）
                async for ev in self._astream_plan(query, escalated, k, seed_hits=current_hits, memory=memory):
                    if ev["type"] == "retrieve":
                        current_hits = ev["hits"]
                    yield ev
                final = await asyncio.to_thread(self.answerability.verify, query, current_hits)
                logger.info(
                    "[modular] 流式：升级后验证 → answerable=%s（%s）",
                    final.answerable,
                    final.recommendation,
                )
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
