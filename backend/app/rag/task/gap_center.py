"""缺口策略中心：缺口分类（LLM + 规则回退）+ 决策表（可配置、可审计、可评测）。

P2 外层任务闭环的核心策略件：内层上报「证据不足（缺口）」后，由策略中心决定
下一步动作——改写重查 / 如实上报追问 / 转交工具 / 部分接受。
企业级要点：LLM 只负责「缺口属于哪一类」（rag_task_gap 场景），决策表负责
「该类怎么处理」——行为确定性、可回灌调参，不靠模型临场发挥。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.rag.task.decomposer import task_llm_json

logger = logging.getLogger(__name__)

SCENARIO_GAP = "rag_task_gap"

# 缺口类型（决策表主键）
GAP_QUERY = "query"          # 查询表达问题：换词后缺失项与已查内容语义重合 → 改写重查
GAP_DATA = "data"            # 数据确实缺失：库内无此记录 → 如实上报 + 追问
GAP_CROSS = "cross_domain"   # 跨域缺口：属于其它工具能力 → 转交对应工具
GAP_LOW = "low_value"        # 低价值缺口：不影响主问题核心结论 → 部分回答 + 标注

# 决策动作
ACT_REWRITE = "rewrite"   # 改写查询重查（消耗 retries，上限 max_retries）
ACT_REPORT = "report"     # 如实上报 + 向用户追问（不阻塞，可答部分先行）
ACT_DELEGATE = "delegate"  # 转交其它工具（本系统工具面无对应目标时降级为上报 + 建议）
ACT_ACCEPT = "accept"     # 部分回答 + 明确标注（不阻塞核心结论）


@dataclass
class GapDecision:
    """缺口决策（分类 + 决策表合并结果）：动作 + 改写查询 + 依据。"""

    gap_type: str = GAP_DATA
    action: str = ACT_REPORT
    rewrite_query: str = ""  # rewrite 动作的改写后查询（可直接检索）
    reason: str = ""
    note: str = ""  # 非空 = 规则回退 / 次数触顶 / 降级原因


# 决策表：类型 → 默认动作（LLM 只给分类，动作由表决定）
GAP_ACTION = {
    GAP_QUERY: ACT_REWRITE,
    GAP_DATA: ACT_REPORT,
    GAP_CROSS: ACT_DELEGATE,
    GAP_LOW: ACT_ACCEPT,
}
_VALID_TYPES = frozenset(GAP_ACTION)


class GapClassifier:
    """缺口分类器：LLM（rag_task_gap 轻量 JSON）+ 规则回退（保守：数据缺失上报）。"""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def classify(
        self,
        query: str,
        missing_facts: list[str],
        evidence: list[dict[str, Any]] | None = None,
        ledger: dict[str, int] | None = None,
    ) -> GapDecision:
        """分类缺口。LLM 不可用/不可解析 → 规则回退，保证任务不因单次决策故障中断。"""
        if self.use_llm:
            def parse(data: dict[str, Any]) -> GapDecision | None:
                t = str(data.get("gap_type") or "")
                if t not in _VALID_TYPES:
                    return None
                rq = str(data.get("rewrite_query") or "").strip()
                return GapDecision(
                    gap_type=t,
                    action=GAP_ACTION[t],
                    rewrite_query=rq,
                    reason=str(data.get("reason") or ""),
                )

            outcome = task_llm_json(
                SCENARIO_GAP,
                self._system(),
                self._user(query, missing_facts, evidence),
                parse,
                ledger=ledger,
            )
            if outcome is not None:
                logger.info("[gap] 分类 %s：%s（%s）", outcome.gap_type, outcome.action, outcome.reason or "无理由")
                return outcome
        return self._fallback(query, missing_facts)

    @staticmethod
    def _is_subseq(a: str, b: str) -> bool:
        """a 是否为 b 的字符子序列（可跳字）：判定缺失项是否为查询的「扩展/更完整表达」。"""
        it = iter(b)
        return all(ch in it for ch in a)

    @classmethod
    def _fallback(cls, query: str, missing_facts: list[str]) -> GapDecision:
        """规则回退：缺失项与查询互为子序列（同语义更完整表达）→ 表达问题（拼词改写重查）；
        否则（缺失项是另一事实方面，如「发票 vs 时限」）保守报数据缺失。"""
        q = query or ""
        qn = q.replace(" ", "")
        for m in missing_facts:
            mn = (m or "").replace(" ", "")
            if qn and mn and (cls._is_subseq(qn, mn) or cls._is_subseq(mn, qn)):
                rq = q
                for term in missing_facts:
                    if term and term not in rq:
                        rq = f"{rq} {term}"
                return GapDecision(
                    gap_type=GAP_QUERY, action=ACT_REWRITE, rewrite_query=rq,
                    reason="规则回退：缺失项与查询强重叠", note="缺口分类规则回退",
                )
        return GapDecision(
            gap_type=GAP_DATA, action=ACT_REPORT,
            reason="规则回退：默认数据缺失", note="缺口分类规则回退",
        )

    @staticmethod
    def _system() -> str:
        return (
            "你是检索缺口分类器。给定节点查询、缺失事实与已检证据，把缺口归为四类之一，"
            "并在改写类时给出下一步可直接检索的查询。\n"
            "类型判定：query=查询表达问题（换词后缺失项与已查内容语义重合，改写查询可再查）；"
            "data=数据确实缺失（库内无此记录，重查无意义）；"
            "cross_domain=跨域缺口（缺口属于其它工具能力，如计算/外部数据，需转交对应工具）；"
            "low_value=低价值缺口（不影响主问题核心结论，可部分回答并标注）。\n"
            "query 类必须给出 rewrite_query（改写后的具体查询，不要带套话）；其它类型 rewrite_query 留空串。\n"
            '输出严格 JSON：{"gap_type": "query|data|cross_domain|low_value", '
            '"rewrite_query": "改写查询或空串", "reason": "简短理由"}'
        )

    @staticmethod
    def _user(
        query: str,
        missing_facts: list[str],
        evidence: list[dict[str, Any]] | None,
    ) -> str:
        body = f"节点查询：{query}\n缺失事实：{'、'.join(missing_facts[:4])}"
        if evidence:
            lines = "\n".join(f"· {(h.get('text') or '')[:120]}" for h in evidence[:5])
            body += f"\n已检证据：\n{lines}"
        return body + "\n请输出缺口分类 JSON。"


class GapStrategyCenter:
    """缺口策略中心：分类器 + 决策表（次数上限 / 降级），返回最终动作。"""

    def __init__(self, classifier: GapClassifier | None = None):
        self.classifier = classifier or GapClassifier()

    async def decide(
        self,
        query: str,
        missing_facts: list[str],
        evidence: list[dict[str, Any]] | None,
        retries: int,
        max_retries: int,
        ledger: dict[str, int] | None = None,
    ) -> GapDecision:
        """分类缺口并按决策表收敛到最终动作（改写可执行 / 其它如实上报）。"""
        decision = await asyncio.to_thread(
            self.classifier.classify, query, missing_facts, evidence, ledger
        )
        # 决策表收敛：改写类超次数上限 / 无改写查询 → 降级上报；跨域 → 降级上报 + 建议
        if decision.action == ACT_REWRITE and (retries >= max_retries or not decision.rewrite_query):
            decision.action = ACT_REPORT
            decision.note = (
                f"{decision.note}；改写重查次数达上限({max_retries})" if decision.note
                else f"改写重查次数达上限({max_retries})"
            )
        elif decision.action == ACT_DELEGATE:
            decision.action = ACT_REPORT
            decision.note = (
                f"{decision.note}；跨域缺口建议转交其它工具（计算/外部数据）" if decision.note
                else "跨域缺口建议转交其它工具（计算/外部数据）"
            )
        return decision
