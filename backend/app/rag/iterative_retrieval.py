"""多跳迭代检索（Iterative Retrieval）：检索→判断缺失→生成下一跳查询→再检索。

对齐 Modular RAG 企业级架构的「Agent 模式/迭代检索」思想：多跳问题（流程/原因/步骤链）
无法在一次宽召回中保证中间环节齐全，需要基于「当前已召回的证据」判断还缺哪一环，
并生成下一跳子查询继续召回，直至证据充分或达到最大跳数。

关键设计：决策时携带「累积证据」（此前所有跳已召回的命中），而不是只给当前跳——
实体链问题（如「张三的领导有多少天年假」）首跳往往已命中「张三的领导是王刚」与年假准则，
只有把已有命中当作参考交给下一跳判断，模型才能直接推进到「王刚的年假有多少天」，
避免重复查询已被证据解决的环节。

- LLMMultiHopRetriever：按命名场景 rag_next_step 懒取聊天模型做「下一跳查询生成」——
  每跳检索后由模型基于累积证据判断是否还需信息、生成聚焦缺失环节的子查询；
  解析失败/异常保守停止（不中断整条链路）；
- RuleMultiHopRetriever：无 LLM（离线/仅配 Embedding）时的确定性回退——
  「顺藤摸瓜」：取最近一跳 top-1 命中里「原查询未含」的领域关键词，拼回原查询形成下一跳；
  有新材料才续跳，无新材料即停（最多 2 跳，受 max_hops 上限约束）。
"""
from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model
from app.rag.fusion import reciprocal_rank_fusion

# 领域关键词：来自 classifier/query_rewrite 的关键词表（避免循环导入，此处内联一份）
_KEYWORDS = [
    "考勤", "打卡", "迟到", "早退", "旷工", "补卡",
    "年假", "事假", "病假", "福利", "补贴",
    "出差", "差旅", "报销", "住宿", "交通", "餐补", "发票",
    "审批", "工资", "绩效", "城市", "一线", "二线",
]

# 规则兜底最多续跳次数（受 max_hops 上限约束）
_RULE_MAX_HOPS = 2


@dataclass
class HopRecord:
    """一次检索跳的记录：该跳子查询 + 该跳命中 + 目标 + 是否被复用跳过。"""

    query: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    next_query: str | None = None
    target: str | None = None  # 该跳要解决的"目标维度/事实"（来自计划步骤，可观测）
    skipped: bool = False  # True=覆盖检测命中，已被既有证据解决，未实际检索（复用）


@dataclass
class PlanStep:
    """规划器产出的一个子查询步骤（含依赖标注 = 计划的有向边）。"""

    target: str  # 本步要填充的目标维度/事实（如"领导是谁"/"年假天数"）
    query: str  # 可独立检索的子查询
    entity: str | None = None  # 可预判的关键实体/概念（覆盖检测依据；None=不参与复用跳过）
    depends_on: list[str] = field(default_factory=list)  # 依赖的先前 target
    status: str = "pending"  # pending / covered(复用跳过) / done / unexecuted(超预算)


@dataclass
class HopPlan:
    """多跳检索计划：规划阶段的产出，供执行器逐跳执行、验证器对表。"""

    steps: list[PlanStep]
    reason: str = ""  # 规划理由（可观测）


@dataclass
class VerifyResult:
    """质量闸门结果：计划目标覆盖对表 + 补缺子查询（局部修正依据）。"""

    covered: list[str] = field(default_factory=list)  # 已被证据覆盖的 target
    missing: list[str] = field(default_factory=list)  # 未覆盖的 target
    patched: list[dict[str, Any]] = field(default_factory=list)  # [{"target","query"}] 补缺子查询


@dataclass
class MultiHopEvent:
    """流式多跳事件：plan / 逐跳 hop / verify，供前端按序展示（替代一次返回全部）。"""

    kind: str  # "plan" / "hop" / "verify"
    index: int | None = None  # 仅 hop：从 1 递增
    plan: HopPlan | None = None
    hop: HopRecord | None = None
    verification: VerifyResult | None = None


@dataclass
class MultiHopResult:
    """多跳检索结果：逐跳记录（可观测）+ 计划 + 验证结果 + 全跳合并去重的命中。"""

    hops: list[HopRecord] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)
    plan: HopPlan | None = None
    verification: VerifyResult | None = None


# ---- 结构化序列化（供 modular 编排 / 前端事件 / RetrieveResult 使用） ----


def hop_to_dict(hop: HopRecord) -> dict[str, Any]:
    """HopRecord → dict（含 target / skipped，前端逐跳展示与覆盖复用标记）。"""
    return {
        "query": hop.query,
        "hits": hop.hits,
        "next_query": hop.next_query,
        "target": hop.target,
        "skipped": hop.skipped,
    }


def plan_to_dict(plan: HopPlan | None) -> dict[str, Any] | None:
    """HopPlan → dict（步骤含 target / entity / depends_on / status）。"""
    if plan is None:
        return None
    return {
        "steps": [
            {
                "target": s.target,
                "query": s.query,
                "entity": s.entity,
                "depends_on": s.depends_on,
                "status": s.status,
            }
            for s in plan.steps
        ],
        "reason": plan.reason,
    }


def verify_to_dict(verification: VerifyResult | None) -> dict[str, Any] | None:
    """VerifyResult → dict（covered / missing / patched）。"""
    if verification is None:
        return None
    return {
        "covered": verification.covered,
        "missing": verification.missing,
        "patched": verification.patched,
    }


class MultiHopRetriever(ABC):
    """多跳迭代检索抽象：输入原查询，输出逐跳记录与合并命中。"""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        store,
        top_k: int,
        max_hops: int,
        recall_k: int,
    ) -> MultiHopResult:
        """迭代检索：每跳基于已有证据判断是否续跳并生成下一跳子查询。"""

    async def astream_retrieve(
        self,
        query: str,
        store,
        top_k: int,
        max_hops: int,
        recall_k: int,
    ):
        """异步逐跳迭代检索：每完成一跳立即 yield 该跳记录（HopRecord），供前端逐跳流式展示。

        阻塞调用（双路召回、下一跳判断）放线程池执行，逐跳产出而不是一次性把全部跳返回；
        各跳命中由调用方按 RRF 合并后，再做重排/压缩等后处理。
        下一跳决策携带「累积证据」（此前所有跳的命中），参考已有内容推进而非重复查询。
        """
        current = query
        limit = self._hop_limit(max_hops)
        all_hits: list[list[dict[str, Any]]] = []
        for hop_index in range(1, limit + 1):
            hits = await asyncio.to_thread(_multi_recall, store, current, recall_k)
            all_hits.append(hits)
            if hop_index >= limit:
                yield HopRecord(query=current, hits=hits, next_query=None)
                return
            next_query = await asyncio.to_thread(self._decide_next, query, all_hits, top_k)
            yield HopRecord(query=current, hits=hits, next_query=next_query)
            if not next_query:
                return
            current = next_query

    def _hop_limit(self, max_hops: int) -> int:
        """该实现允许的最大跳数（LLM 用满 max_hops，规则兜底受 _RULE_MAX_HOPS 限制）。"""
        return max(1, int(max_hops))

    def _decide_next(
        self,
        query: str,
        all_hits: list[list[dict[str, Any]]],
        top_k: int,
    ) -> str | None:
        """子类实现：基于「累积证据」（此前所有跳的命中）判断是否续跳，返回下一跳子查询（None/空 表示停止）。"""
        raise NotImplementedError


def _multi_recall(store, query: str, recall_k: int) -> list[dict[str, Any]]:
    """双路召回（向量 + 混合）经 RRF 融合：两路分数体系不同，不可直接比较取最大。"""
    return reciprocal_rank_fusion(
        [store.search(query, recall_k), store.hybrid_search(query, recall_k)]
    )


def _merge_hits(results: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """多跳合并：各跳召回列表经 RRF 融合，出现在多跳 / 更靠前的文档（共同证据）得分更高。"""
    return reciprocal_rank_fusion(results)


def step_covered(step: PlanStep, original: str, evidence: str) -> bool:
    """内容级覆盖检测：该步要解决的事实是否已被证据文本覆盖（复用跳过依据）。

    - 实体命中：可预判实体（step.entity）已出现在证据中 → 已解决；
    - 关键词命中：该步相对原查询新引入的领域词已全部出现在证据中 → 已解决。

    供验证器在无 LLM 时兜底使用（逐跳覆盖对表与终局对表同口径）——
    有 LLM 时逐跳与终局对表均由 LLMMultiHopVerifier 做语义判断（关闭思考模式）。
    """
    if step.entity and step.entity in evidence:
        return True
    new_keywords = [kw for kw in _KEYWORDS if kw in step.query and kw not in original]
    return bool(new_keywords) and all(kw in evidence for kw in new_keywords)


class LLMMultiHopRetriever(MultiHopRetriever):
    """LLM 驱动的多跳迭代检索：按场景懒取模型，每跳检索后让模型判断是否续跳并生成下一跳子查询。

    下一跳决策是轻量决策调用（分析还缺哪一环 → 生成聚焦缺失环节的子查询），
    场景 rag_next_step 关闭思考模式。任何异常/非法输出都保守视为「无需续跳」，
    保证不因模型抖动中断检索链路。
    """

    # 本阶段使用的模型/参数场景（qwen3.5-flash / temp=0.3 / max_tokens=300 / thinking=False，见 service.DEFAULT_PROFILES）
    scenario = "rag_next_step"

    def retrieve(
        self,
        query: str,
        store,
        top_k: int,
        max_hops: int,
        recall_k: int,
    ) -> MultiHopResult:
        hops: list[HopRecord] = []
        all_hits: list[list[dict[str, Any]]] = []
        current = query
        for hop_index in range(1, max(1, int(max_hops)) + 1):
            hits = _multi_recall(store, current, recall_k)
            all_hits.append(hits)
            if hop_index >= max(1, int(max_hops)):
                hops.append(HopRecord(query=current, hits=hits, next_query=None))
                break
            next_query = self._decide_next(query, all_hits, top_k)
            hops.append(HopRecord(query=current, hits=hits, next_query=next_query))
            if not next_query:
                break
            current = next_query
        return MultiHopResult(hops=hops, hits=_merge_hits(all_hits))

    def _decide_next(
        self,
        query: str,
        all_hits: list[list[dict[str, Any]]],
        top_k: int,
    ) -> str | None:
        """LLM 判断是否续跳：把此前所有跳的命中按 RRF 合并取 Top-K 作为参考证据，
        避免下一跳重复查询已在证据中解决的事实（如"张三的领导是谁"）；
        从结构化决策中取下一跳子查询（continue=false 或解析失败即停）。"""
        evidence = _merge_hits(all_hits)[: top_k * 2]
        decision = self._next_step(query, evidence)
        next_query = (decision.get("next_query") or "").strip() if decision.get("continue") else ""
        return next_query or None

    def _next_step(self, query: str, evidence: list[dict[str, Any]]) -> dict:
        """让模型基于「已累计检索到的内容」判断：是否足以回答原始问题，否则生成下一跳子查询。"""
        llm = get_chat_model(self.scenario)
        if llm is None:
            return {"continue": False}
        try:
            evidence_text = "\n".join(
                f"- {h.get('text', '')[:200]}" for h in evidence
            ) or "（暂无）"
            messages = [
                SystemMessage(
                    content=(
                        "你是多跳检索的「下一跳查询生成器」。给定原始问题与目前已累计检索到的全部内容，判断：\n"
                        "1. 先基于已检索内容解析原始问题中已知的环节（例如已检索到「张三的领导是王刚」，"
                        "则「张三的领导是谁」这一环已解决，不应再查）；\n"
                        "2. 若已检索内容足以回答原始问题 → 输出 {\"continue\": false}；\n"
                        "3. 若仍缺关键信息（中间环节、前置条件、后续步骤、原因链的下一环），且该信息未出现在"
                        "已检索内容中 → 输出 {\"continue\": true, \"next_query\": \"聚焦缺失环节、可独立检索的子问题\"}；\n"
                        "4. 下一跳查询应直接使用已解析出的实体与结论推进（如已得出「领导是王刚」，下一跳应为"
                        "「王刚的年假有多少天」，而不是「张三的领导是谁」）；\n"
                        "5. 不得重复查询已检索内容中已出现的信息。\n"
                        "输出必须严格是 JSON，不要输出任何其他文字。"
                    )
                ),
                HumanMessage(
                    content=f"原始问题：{query}\n已累计检索到的内容：\n{evidence_text}"
                ),
            ]
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return {"continue": False}
            data = json.loads(match.group(0))
            if not isinstance(data, dict):
                return {"continue": False}
            return {"continue": bool(data.get("continue", False)), "next_query": str(data.get("next_query") or "")}
        except Exception:  # noqa: BLE001 — 模型抖动时保守停止，不中断链路
            return {"continue": False}


class RuleMultiHopRetriever(MultiHopRetriever):
    """确定性规则多跳迭代检索（无 LLM 回退）：「顺藤摸瓜」关键词扩展。

    首跳用原查询双路召回；取 top-1 命中里「原查询未含」的领域关键词拼回原查询形成下一跳，
    有新材料才续跳、无新材料即停（最多 2 跳）。确定性、离线可测。
    """

    def retrieve(
        self,
        query: str,
        store,
        top_k: int,
        max_hops: int,
        recall_k: int,
    ) -> MultiHopResult:
        hops: list[HopRecord] = []
        all_hits: list[list[dict[str, Any]]] = []
        current = query
        limit = min(max(1, int(max_hops)), _RULE_MAX_HOPS)
        for hop_index in range(1, limit + 1):
            hits = _multi_recall(store, current, recall_k)
            all_hits.append(hits)
            if hop_index >= limit:
                hops.append(HopRecord(query=current, hits=hits, next_query=None))
                break
            next_query = self._decide_next(query, all_hits, top_k)
            hops.append(HopRecord(query=current, hits=hits, next_query=next_query))
            if not next_query:
                break
            current = next_query
        return MultiHopResult(hops=hops, hits=_merge_hits(all_hits))

    def _hop_limit(self, max_hops: int) -> int:
        return min(max(1, int(max_hops)), _RULE_MAX_HOPS)

    def _decide_next(
        self,
        query: str,
        all_hits: list[list[dict[str, Any]]],
        top_k: int,
    ) -> str | None:
        # 规则只看最近一跳的 top-1 命中做关键词扩展（确定性，不需要跨跳累积上下文）
        return self._expand(query, all_hits[-1])

    @staticmethod
    def _expand(query: str, hits: list[dict[str, Any]]) -> str | None:
        """顺藤摸瓜：取 top-1 命中里「原查询未含」的领域关键词，拼回原查询形成下一跳。"""
        if not hits:
            return None
        base = [kw for kw in _KEYWORDS if kw in query]
        trail = [kw for kw in _KEYWORDS if kw in (hits[0].get("text", "") or "") and kw not in base]
        if not trail:
            return None
        next_query = " ".join(base + trail)
        if next_query == query:
            return None
        return next_query


class PlanExecuteRetriever(MultiHopRetriever):
    """规划-执行-验证（Plan-Execute-Verify）多跳检索：企业级编排模板。

    [规划] 规划器一次拆出子查询计划（目标/依赖/可预判实体）；
    [执行] 按计划逐跳检索，每跳前做覆盖检测——复用验证器判断「累积证据是否已回答下一跳」，
           覆盖 → 复用跳过（不重查），否则照常检索；
    [验证] 验证器质量闸门：计划目标是否全被证据覆盖？缺口 → 预算内局部修正（补缺子查询），
           超预算则如实上报缺口（可观测）；最多补修一轮后二次对表。

    相比贪心迭代（LLMMultiHopRetriever）的差异：规划让全局视野不漏查、依赖显式化；
    覆盖检测把"不重复查已解决事实"从模型自觉升级为语义闸门——逐跳与终局均复用验证器
    （有 LLM 时 LLMMultiHopVerifier 关闭思考模式语义判断，无 LLM 回退规则），命中即复用跳过；
    验证给出质量结论。
    """

    def __init__(self, planner, verifier):
        from app.rag.planner import MultiHopPlanner  # noqa: F401 — 仅类型提示
        from app.rag.verifier import MultiHopVerifier  # noqa: F401 — 仅类型提示

        self.planner = planner
        self.verifier = verifier

    def _hop_limit(self, max_hops: int) -> int:
        return max(1, int(max_hops))

    def _covered(self, step: PlanStep, query: str, all_hits: list[list[dict[str, Any]]]) -> bool:
        """逐跳覆盖检测：执行下一跳前，判断累积证据是否已覆盖该步目标（复用跳过依据）。

        复用验证器做语义判定（有 LLM 时关闭思考模式，无 LLM 回退规则），与终局验证同口径；
        覆盖 → 该步复用跳过；否则照常检索，由终局验证器兜底补缺。
        无累积证据（首跳/空召回）时不做无谓调用，直接返回 False。
        """
        if not all_hits:
            return False
        evidence = _merge_hits(all_hits)
        if not evidence:
            return False
        verification = self.verifier.verify(query, HopPlan(steps=[step]), evidence)
        return step.target in verification.covered

    def _execute(
        self, query: str, store, budget: int, recall_k: int
    ) -> tuple[HopPlan, list[HopRecord], list[list[dict[str, Any]]], VerifyResult]:
        """同步执行计划：规划 → 覆盖检测逐跳执行 → 验证闸门 + 局部修正 → 二次对表。"""
        plan = self.planner.plan(query)
        hops: list[HopRecord] = []
        all_hits: list[list[dict[str, Any]]] = []
        retrieved = 0
        for step in plan.steps:
            if retrieved >= budget:
                step.status = "unexecuted"
                continue
            if self._covered(step, query, all_hits):
                step.status = "covered"
                hops.append(HopRecord(query=step.query, hits=[], target=step.target, skipped=True))
                continue
            hits = _multi_recall(store, step.query, recall_k)
            all_hits.append(hits)
            retrieved += 1
            step.status = "done"
            hops.append(HopRecord(query=step.query, hits=hits, target=step.target))
        # 验证闸门：缺口 → 预算内局部修正（补缺子查询追加执行）；最多一轮补修后二次对表
        verification = self.verifier.verify(query, plan, _merge_hits(all_hits))
        if verification.patched and retrieved < budget:
            for patch in verification.patched:
                if retrieved >= budget:
                    break
                hits = _multi_recall(store, patch["query"], recall_k)
                all_hits.append(hits)
                retrieved += 1
                hops.append(HopRecord(query=patch["query"], hits=hits, target=patch.get("target")))
            verification = self.verifier.verify(query, plan, _merge_hits(all_hits))
        return plan, hops, all_hits, verification

    def retrieve(
        self,
        query: str,
        store,
        top_k: int,
        max_hops: int,
        recall_k: int,
    ) -> MultiHopResult:
        plan, hops, all_hits, verification = self._execute(query, store, self._hop_limit(max_hops), recall_k)
        return MultiHopResult(
            hops=hops,
            hits=_merge_hits(all_hits),
            plan=plan,
            verification=verification,
        )

    async def astream_retrieve(
        self,
        query: str,
        store,
        top_k: int,
        max_hops: int,
        recall_k: int,
    ):
        """异步流式：先产出 plan 事件，再逐跳产出 hop 事件（覆盖跳过带标记），最后产出 verify 事件。"""
        plan = self.planner.plan(query)
        yield MultiHopEvent(kind="plan", plan=plan)
        budget = self._hop_limit(max_hops)
        hops: list[HopRecord] = []
        all_hits: list[list[dict[str, Any]]] = []
        retrieved = 0
        idx = 0
        for step in plan.steps:
            if retrieved >= budget:
                step.status = "unexecuted"
                continue
            if await asyncio.to_thread(self._covered, step, query, all_hits):
                step.status = "covered"
                idx += 1
                hop = HopRecord(query=step.query, hits=[], target=step.target, skipped=True)
                hops.append(hop)
                yield MultiHopEvent(kind="hop", index=idx, hop=hop)
                continue
            hits = await asyncio.to_thread(_multi_recall, store, step.query, recall_k)
            all_hits.append(hits)
            retrieved += 1
            step.status = "done"
            idx += 1
            hop = HopRecord(query=step.query, hits=hits, target=step.target)
            hops.append(hop)
            yield MultiHopEvent(kind="hop", index=idx, hop=hop)
        # 验证闸门 + 局部修正（预算内），最多一轮补修
        verification = await asyncio.to_thread(self.verifier.verify, query, plan, _merge_hits(all_hits))
        if verification.patched and retrieved < budget:
            for patch in verification.patched:
                if retrieved >= budget:
                    break
                hits = await asyncio.to_thread(_multi_recall, store, patch["query"], recall_k)
                all_hits.append(hits)
                retrieved += 1
                idx += 1
                hop = HopRecord(query=patch["query"], hits=hits, target=patch.get("target"))
                hops.append(hop)
                yield MultiHopEvent(kind="hop", index=idx, hop=hop)
            verification = await asyncio.to_thread(self.verifier.verify, query, plan, _merge_hits(all_hits))
        yield MultiHopEvent(kind="verify", verification=verification)


def build_multi_hop_retriever(planner=None, verifier=None) -> MultiHopRetriever:
    """构造多跳检索器：默认「规划-执行-验证」（Planner + Verifier，各自按场景懒取 LLM）；
    也保留贪心迭代检索器（LLMMultiHopRetriever / RuleMultiHopRetriever）作为回退路径。"""
    from app.rag.planner import build_planner
    from app.rag.verifier import build_verifier

    planner = planner if planner is not None else build_planner()
    verifier = verifier if verifier is not None else build_verifier()
    return PlanExecuteRetriever(planner, verifier)
