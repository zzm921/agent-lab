"""检索工具注册表（Tool Registry）：把库内检索能力封装为带治理的工具。

企业级 Agentic RAG 的工具治理口径：
- 注册表白名单：Agent 只能调用注册表内的工具（name/description/call_cap）；
- 单工具调用上限（call_cap）：防止同一工具被反复滥用；
- 重复调用去重：同（action, query, volume）已被正常执行过则跳过（护栏拦截，消耗预算）；
- 非法卷名降级：volume_search 的卷名必须取自卷目录，否则降级为全库 search；
- 并行波次执行：一波独立调用放线程池并发（首发按事实清单并行，纠错波同理）；
- 扩展点：新增工具（web/SQL 等）只需注册新 spec + executor，不改角色与编排。
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from app.rag.agentic.state import AgentState, ToolCallSpec
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.iterative_retrieval import expand_scale_query
from app.rag.schemes.modular import _TARGET_VOLUME_FILTERS

logger = logging.getLogger(__name__)


@dataclass
class WaveResult:
    """一波中单个调用的执行记录：调用 + 护栏备注 + 命中（RRF 融合与轨迹用）。

    query_pipeline：本次调用的查询变换明细（检索类型/是否 HyDE/展开后查询），
    供 pipeline 完整链路日志（查询向量生成方法）展示。
    """

    call: ToolCallSpec
    note: str = ""  # 空 = 正常执行；非空 = 被护栏拦截/降级的原因
    hits: list[dict[str, Any]] = field(default_factory=list)
    query_pipeline: dict[str, Any] = field(default_factory=dict)

# 工具白名单（注册表内置 4 个库内检索工具；扩展 web/SQL 在此登记）
ACTION_SEARCH = "search"
ACTION_HYBRID = "hybrid"
ACTION_VOLUME = "volume_search"
ACTION_MULTI_HOP = "multi_hop"


@dataclass(frozen=True)
class ToolSpec:
    """工具描述（注册表条目）：Agent 决策可见的能力清单。"""

    name: str
    description: str
    call_cap: int  # 单工具整轮调用上限（预算治理）


def volume_catalog() -> list[str]:
    """聚合定向卷白名单为工具可用的卷目录（去重、保序）。"""
    seen: list[str] = []
    for vols in _TARGET_VOLUME_FILTERS.values():
        for v in vols:
            if v not in seen:
                seen.append(v)
    return seen


def default_registry_specs(call_cap: int) -> list[ToolSpec]:
    """默认工具清单（multi_hop 成本高，cap 独立收紧为 1）。"""
    return [
        ToolSpec(ACTION_SEARCH, "纯向量语义检索（适合语义相似、同义改写类查询）", call_cap),
        ToolSpec(ACTION_HYBRID, "混合检索：语义+关键词，含人数/规模规范词扩展（适合制度条款/表格）", call_cap),
        ToolSpec(ACTION_VOLUME, f"定向卷内检索（params.volume 须取自卷目录；适合档案/FAQ/案例/版本对比）", call_cap),
        ToolSpec(ACTION_MULTI_HOP, "多跳规划-执行-验证检索（适合实体链/流程链问题）", 1),
    ]


def terms2(text: str) -> set[str]:
    """中文相邻 2 字词集合（重叠窗口）：相关性/seed 复用的轻量粗判。"""
    seg = re.findall(r"[\u4e00-\u9fff]+", text)
    return {s[i : i + 2] for s in seg for i in range(len(s) - 1)}


# 查询术语归一：口语/用户措辞 → 制度用语（检索前统一口径，缩小与语料表头/职位词的词面鸿沟）
_QUERY_TERM_ALIASES = (("领导", "部门主管"),)


def normalize_query(text: str) -> str:
    """查询术语归一：把口语措辞替换为制度用语（如 领导→部门主管）。

    与 expand_scale_query 互补：后者在混合检索执行时追加规模表规范词，
    这里在 Planner/Corrector 生成子查询后立即替换，保证检索词与语料职位词对齐
    （语料用「部门主管」，用户口语用「领导」，词面不匹配会压低语义分）。
    """
    for src, dst in _QUERY_TERM_ALIASES:
        text = text.replace(src, dst)
    return text


def diversify(
    hits: list[dict[str, Any]],
    max_items: int = 3,
    max_overlap: float = 0.55,
) -> list[dict[str, Any]]:
    """定向卷内多样性截断：同模板块（逐人档案/同类 FAQ）只留代表，防成群挤占融合配额。"""
    kept: list[dict[str, Any]] = []
    kept_terms: list[set[str]] = []
    for hit in hits:
        t = terms2(hit.get("text") or "")
        if not t:
            continue
        if any(len(t & prev) / max(1, min(len(t), len(prev))) > max_overlap for prev in kept_terms):
            continue
        kept.append(hit)
        kept_terms.append(t)
        if len(kept) >= max_items:
            break
    return kept


def cross_turn_seed(query: str, prev_hits: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """跨轮 seed 复用闸门：上轮已验证命中经分数门槛/相关性把关/限量后作本轮候选证据。

    - score < 0.5 丢弃（弱命中/幻觉通常低分）；与当前查询无共现 2 字词丢弃；
    - 最多 5 条；只作额外一路融合候选，不注入查询文本、不占当前轮召回配额。
    """
    if not prev_hits:
        return []
    q_terms = terms2(query)
    kept: list[dict[str, Any]] = []
    for hit in sorted(prev_hits, key=lambda h: h.get("score") or 0.0, reverse=True):
        if len(kept) >= 5:
            break
        if (hit.get("score") or 0.0) < 0.5:
            break  # 已按分数降序，其后只会更低
        if q_terms and not (q_terms & terms2(hit.get("text") or "")):
            continue
        kept.append(hit)
    return kept


class ToolRegistry:
    """检索工具注册表：执行一波工具调用（并行），统一施加预算护栏。"""

    def __init__(
        self,
        store,
        multi_hop=None,
        max_hops: int = 3,
        call_cap: int = 3,
        parallel: int = 4,
        specs: list[ToolSpec] | None = None,
        hyde=None,  # HyDE 展开器：隐式附加在 hybrid 工具内（Agent 无感知，不占工具位/预算）
    ):
        self.store = store
        self.multi_hop = multi_hop
        self.max_hops = max_hops
        self.parallel = max(1, parallel)
        self.catalog = volume_catalog()
        self.specs = {s.name: s for s in (specs or default_registry_specs(call_cap))}
        self.hyde = hyde

    # ---- 护栏（无状态：预算/去重状态在 per-query 的 AgentState 上，注册表可跨请求共享） ----

    def guard(self, call: ToolCallSpec, state: AgentState, wave_calls: dict[str, int]) -> str:
        """护栏校验：返回放行则空串，否则返回拦截原因（保留决策意图的拦截不执行）。

        call_cap 按「每波」配额（wave_calls 为单波累计）：首发/每轮纠错各自独立配额，
        避免首发探路检索（如 volume_search 锁定部门）耗尽整轮配额导致纠错路无检索可用；
        总调用仍受 max_steps/token 预算整轮兜底。
        """
        spec = self.specs.get(call.action)
        if spec is None:
            return f"工具 {call.action} 不在注册表内"
        if wave_calls.get(call.action, 0) >= spec.call_cap:
            return f"工具 {call.action} 已达调用上限 {spec.call_cap}"
        key = (call.action, call.query, call.volume)
        if key in state.executed_keys:
            return "与已执行调用重复"
        return ""

    def _degrade(self, call: ToolCallSpec) -> ToolCallSpec:
        """非法卷名降级为全库检索（保留检索意图）；未知工具降级为 hybrid。"""
        if call.action == ACTION_VOLUME and call.volume not in self.catalog:
            return ToolCallSpec(ACTION_SEARCH, call.query, "", call.reason + "（卷名不在目录，降级全库）")
        if call.action not in self.specs:
            return ToolCallSpec(ACTION_HYBRID, call.query, "", call.reason + "（未知工具降级 hybrid）")
        return call

    # ---- 执行 ----

    def _execute_one(self, call: ToolCallSpec, k: int, recall_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """执行单个工具调用（同步阻塞；并行波次在线程池内运行）。k/recall_k 走参数不落实例，防跨请求竞态。

        返回 (hits, query_pipeline)：query_pipeline 记录本次调用的查询变换明细——
        检索类型/向量体系（dense|dense+sparse）/是否隐式 HyDE/展开后查询，供完整链路日志。
        """
        call = self._degrade(call)
        if call.action == ACTION_SEARCH:
            meta = {"type": "search", "embedding": "dense"}
            return self.store.search(call.query, recall_k), meta
        if call.action == ACTION_HYBRID:
            # hybrid 工具内隐式 HyDE：假想答案文档作一路 doc-space 稠密召回（与 semantic+keyword 路 RRF 融合）。
            # Agent 无感知——不新增工具位、不增加决策/事件；规则回退（无 Key/失败返回原查询）时自动跳过。
            query = expand_scale_query(call.query)
            ranked = [self.store.hybrid_search(query, recall_k)]
            meta: dict[str, Any] = {
                "type": "hybrid", "embedding": "dense+sparse",
                "hyde": False, "expanded": query,
            }
            hyde_doc = self.hyde.expand(call.query) if self.hyde is not None else ""
            if hyde_doc and hyde_doc != call.query:
                ranked.append(self.store.search(hyde_doc, recall_k))
                meta["hyde"] = True
                meta["hyde_doc"] = hyde_doc[:120]
                logger.info("[registry] hybrid 隐式 HyDE：假想文档 %r → 追加一路 doc-space 召回", hyde_doc)
            return reciprocal_rank_fusion(ranked), meta
        if call.action == ACTION_VOLUME:
            meta = {"type": "volume_search", "embedding": "dense", "volume": call.volume}
            return self.store.search(call.query, recall_k, (call.volume,)), meta
        if call.action == ACTION_MULTI_HOP and self.multi_hop is not None:
            res = self.multi_hop.retrieve(call.query, self.store, k, self.max_hops, recall_k)
            meta = {"type": "multi_hop", "embedding": "dense", "hops": len(getattr(res, "trace", []) or [])}
            return res.hits, meta
        return [], {"type": call.action}

    def execute_wave(self, calls: list[ToolCallSpec], k: int, recall_k: int, state: AgentState) -> list[WaveResult]:
        """执行一波工具调用：护栏过滤 → 线程池并行 → 逐调用结果（含拦截备注）。

        被拦截/降级的调用也返回 WaveResult（note 说明原因、hits 为空）——
        调用方据此记录轨迹事件；融合时只取 note 为空的检索结果。
        预算/去重记账（含被拦截计数）写入 per-query 的 state，注册表自身无状态。
        """
        results = [WaveResult(call=call) for call in calls]
        runnable: list[tuple[int, ToolCallSpec]] = []
        wave_calls: dict[str, int] = {}  # 单波内各工具累计（call_cap 按波配额）
        for i, call in enumerate(calls):
            reason = self.guard(call, state, wave_calls)
            state.tool_calls[call.action] = state.tool_calls.get(call.action, 0) + 1
            if reason:
                results[i].note = reason
                logger.info("[registry] 拦截 [%s] %r：%s", call.action, call.query, reason)
                continue
            wave_calls[call.action] = wave_calls.get(call.action, 0) + 1
            runnable.append((i, self._degrade(call)))
        if runnable:
            with ThreadPoolExecutor(max_workers=min(self.parallel, len(runnable))) as ex:
                futures = {ex.submit(self._execute_one, c, k, recall_k): i for i, c in runnable}
                for fut, i in futures.items():
                    try:
                        results[i].hits, results[i].query_pipeline = fut.result()
                    except Exception as exc:  # noqa: BLE001 — 单工具失败不中断整波
                        logger.warning(
                            "[registry] 工具执行失败（%s %r）: %s", calls[i].action, calls[i].query, exc
                        )
                        results[i].note = f"工具执行失败: {exc}"
                        results[i].query_pipeline = {"type": calls[i].action, "error": type(exc).__name__}
        for r in results:
            if not r.note and r.hits:
                state.executed_keys.add((r.call.action, r.call.query, r.call.volume))
        return results
