"""Agent 角色层：路由/规划/评审/纠错/校验五个决策角色（LLM + 确定性规则回退）。

企业级角色分离（对照 CRAG / Self-RAG）：
- RouterAgent（Adaptive RAG 路由）：要不要检索、生成策略、定向语料提示；
- PlannerAgent：查询 → 事实清单 + 首发检索计划（决策「查什么、用什么工具」）；
- GraderAgent（CRAG 证据评审）：逐条证据相关性评分 + 归纳缺失事实；
- CorrectorAgent（CRAG 纠错）：证据不足时的纠错决策——改写/换卷/换工具的下一波调用；
- VerifierAgent（Self-RAG 校验）：事实-证据支持度矩阵，判定可答/缺口。

治理约定（编排层调度，角色层只负责决策与回退）：
- 每个角色一个命名 LLM 场景（rag_agent_route/plan/grade/correct/verify，qwen3.5-flash）；
- use_llm=False（无 Key/超预算/超时/熔断）→ 确定性规则回退，离线评测可跑；
- LLM 输出不可解析/动作越界 → 规则回退（note 记录原因），不中断链路；
- 检索执行不设 LLM 角色：决策智能集中在 Planner/Corrector，执行治理在 ToolRegistry。
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model
from app.rag.agentic.state import ToolCallSpec
from app.rag.agentic.tools import ACTION_HYBRID, ACTION_MULTI_HOP, ACTION_VOLUME, normalize_query

logger = logging.getLogger(__name__)

ROLE_ROUTE = "route"
ROLE_PLAN = "plan"
ROLE_GRADE = "grade"
ROLE_CORRECT = "correct"
ROLE_VERIFY = "verify"

# 链式查询检测：需先解析中间实体（如部门）再查其属性（人数/领导）→ 应走 multi_hop。
# 形如「X的(所在)部门/团队…(人数|领导|主管|规模)」——modular 由同类模式路由到多跳规划-
# 执行-验证；agentic Planner 若只用 parallel hybrid 硬拆多实体，会丢链式依赖、召回变差。
_CHAIN_QUERY = re.compile(
    r"[\u4e00-\u9fff]{2,}的(?:所在)?(?:部门|团队|组织)[^，。]{0,12}"
    r"(?:人数|多少人|几人|在职人数|人员规模|规模|编制|领导|主管|负责人)"
)

SCENARIOS = {
    ROLE_ROUTE: "rag_agent_route",
    ROLE_PLAN: "rag_agent_plan",
    ROLE_GRADE: "rag_agent_grade",
    ROLE_CORRECT: "rag_agent_correct",
    ROLE_VERIFY: "rag_agent_verify",
}


@dataclass
class RoleOutcome:
    """角色决策结果：结构化产物 + 备注（LLM 失败/解析失败回退原因）。"""

    thought: str = ""
    note: str = ""  # 空 = 正常 LLM 决策；非空 = 回退/拦截原因


@dataclass
class RouteOutcome(RoleOutcome):
    retrieval_need: bool = True
    generation_mode: str = "citation"
    target: str = "none"  # 定向语料提示（映射卷目录，未知 target 无提示）


@dataclass
class PlanOutcome(RoleOutcome):
    facts: list[str] = field(default_factory=list)
    calls: list[ToolCallSpec] = field(default_factory=list)


@dataclass
class GradeOutcome(RoleOutcome):
    keep: list[int] = field(default_factory=list)  # 相关证据下标（含部分相关）
    missing_facts: list[str] = field(default_factory=list)


@dataclass
class CorrectOutcome(RoleOutcome):
    calls: list[ToolCallSpec] = field(default_factory=list)  # 纠错波工具调用（空=放弃纠错）


@dataclass
class VerifyOutcome(RoleOutcome):
    answerable: bool = False
    missing_facts: list[str] = field(default_factory=list)


def _extract_json(content: str) -> dict[str, Any] | None:
    """从 LLM 输出提取首个 JSON 对象；不可解析返回 None（角色层回退规则）。"""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _tokens_of(resp: Any) -> dict[str, int]:
    """从响应提取 token 用量（LangChain usage_metadata 优先，DashScope 原生次之）。"""
    usage = getattr(resp, "usage_metadata", None) or {}
    prompt = usage.get("input_tokens") or 0
    completion = usage.get("output_tokens") or 0
    if not prompt and not completion:
        native = (getattr(resp, "response_metadata", None) or {}).get("token_usage") or {}
        prompt = native.get("prompt_tokens") or 0
        completion = native.get("completion_tokens") or 0
    return {"prompt": int(prompt or 0), "completion": int(completion or 0)}


def _llm_json(
    role: str,
    state,
    system: str,
    user: str,
    parse: "callable[[dict[str, Any]], RoleOutcome | None]",
) -> RoleOutcome | None:
    """角色 LLM 调用：调模型 → 记账（次数/token/时延）→ 解析 JSON。

    返回 None 表示未调 LLM（无 Key）或解析失败——调用方回退规则；异常同样回退。
    """
    llm = get_chat_model(SCENARIOS[role])
    if llm is None:
        return None
    start = time.perf_counter()
    state.role_llm_calls[role] = state.role_llm_calls.get(role, 0) + 1
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as exc:  # noqa: BLE001 — 角色调用失败回退规则，不中断链路
        logger.warning("[roles] %s LLM 调用失败: %s", role, exc)
        return None
    usage = _tokens_of(resp)
    state.add_tokens(usage)
    logger.info(
        "[roles] %s LLM 完成 latency=%.2fs tokens=%s",
        role,
        time.perf_counter() - start,
        usage,
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
    data = _extract_json(content)
    if data is None:
        return None
    return parse(data)


class RouterAgent:
    """路由角色：检索必要性 + 生成策略 + 定向语料提示（Adaptive RAG）。"""

    def run(self, query: str, state, use_llm: bool = True) -> RouteOutcome:
        if use_llm:
            def parse(data: dict[str, Any]) -> RouteOutcome | None:
                if "retrieval_need" not in data:
                    return None
                mode = str(data.get("generation_mode") or "citation")
                if mode not in ("direct", "citation", "comparison"):
                    mode = "citation"
                return RouteOutcome(
                    thought=str(data.get("reason") or ""),
                    retrieval_need=bool(data.get("retrieval_need")),
                    generation_mode=mode,
                    target=str(data.get("target") or "none"),
                )

            outcome = _llm_json(ROLE_ROUTE, state, self._system(), self._user(query), parse)
            if outcome is not None:
                return outcome
        # 规则回退：保守全检索 + 引用生成（不猜定向）
        return RouteOutcome(thought="规则回退：默认检索", retrieval_need=True, note="路由规则回退")

    @staticmethod
    def _system() -> str:
        return (
            "你是 RAG 系统的路由角色。判断用户问题是否需要检索知识库，并给出生成策略与定向语料提示。\n"
            '输出严格 JSON：{"retrieval_need": true/false, "generation_mode": "direct|citation|comparison", '
            '"target": "none|profile|faq|case|scene|card|sop|version|duty", "reason": "简短理由"}\n'
            "retrieval_need=false 仅用于寒暄/闲聊；target 是对后续定向检索的提示（不知道就 none）。"
        )

    @staticmethod
    def _user(query: str) -> str:
        return f"用户问题：{query}\n请输出路由决策 JSON。"


class PlannerAgent:
    """规划角色：问题 → 事实清单（要查到什么才能回答）+ 首发检索计划。"""

    _ACTIONS = (ACTION_HYBRID, "search", ACTION_VOLUME, ACTION_MULTI_HOP)

    def run(self, query: str, catalog: list[str], state, use_llm: bool = True) -> PlanOutcome:
        if use_llm:
            def parse(data: dict[str, Any]) -> PlanOutcome | None:
                facts = [str(f) for f in (data.get("facts") or []) if str(f).strip()]
                if not facts:
                    return None
                raw_calls = data.get("calls")
                calls: list[ToolCallSpec] = []
                if isinstance(raw_calls, list):
                    for c in raw_calls[:4]:
                        if not isinstance(c, dict):
                            continue
                        action = str(c.get("action") or "").strip()
                        if action not in self._ACTIONS:
                            continue
                        calls.append(
                            ToolCallSpec(
                                action=action,
                                query=normalize_query(str(c.get("query") or "").strip() or query),
                                volume=str(c.get("volume") or "").strip(),
                                reason=str(c.get("reason") or ""),
                            )
                        )
                return PlanOutcome(
                    thought=str(data.get("reason") or ""),
                    facts=facts,
                    calls=calls or [ToolCallSpec(ACTION_HYBRID, normalize_query(query), "", "默认首发：混合检索")],
                )

            outcome = _llm_json(ROLE_PLAN, state, self._system(), self._user(query), parse)
            if outcome is not None:
                if _CHAIN_QUERY.search(query):
                    # 链式查询（先解析中间实体再查其属性）→ 强制首发改走 multi_hop，
                    # 由 PlanExecuteRetriever 像 modular 一样规划-执行-验证，避免 parallel hybrid 丢链式依赖
                    outcome.calls = [ToolCallSpec(
                        ACTION_MULTI_HOP, normalize_query(query), "",
                        "链式查询：多跳规划-执行-验证",
                    )]
                    outcome.thought = f"{outcome.thought}；检测到链式依赖，首发改走 multi_hop"
                return outcome
        # 规则回退：链式查询 → 多跳；否则单事实 + 单路混合检索（确定性）
        if _CHAIN_QUERY.search(query):
            return PlanOutcome(
                thought="规则回退：链式查询多跳",
                facts=[query],
                calls=[ToolCallSpec(ACTION_MULTI_HOP, normalize_query(query), "", "规则首发：链式多跳")],
                note="规划规则回退",
            )
        return PlanOutcome(
            thought="规则回退：单事实混合检索",
            facts=[query],
            calls=[ToolCallSpec(ACTION_HYBRID, normalize_query(query), "", "规则首发：混合检索")],
            note="规划规则回退",
        )

    @staticmethod
    def _system() -> str:
        return (
            "你是 RAG 系统的规划角色。把用户问题拆解为「回答它必须查到的事实清单」，"
            "并给出首发检索计划（可多路并行，每路一个工具调用）。\n"
            "可用工具：search（纯向量）/ hybrid（语义+关键词，默认首选）/ "
            "multi_hop（实体链/流程链）。首发默认跨卷检索（hybrid 首选），不做定向选卷；"
            "仅当问题明确指向某卷时才用 volume_search。\n"
            '输出严格 JSON：{"facts": ["事实1", "事实2"], '
            '"calls": [{"action": "search|hybrid|volume_search|multi_hop", "query": "该路查询", '
            '"volume": "卷名或空串", "reason": "该路理由"}], "reason": "规划思路"}\n'
            "规则：事实按「谁/什么/多少/何时」原子化拆分；每个事实对应至少一路子查询，"
            "子查询按「实体 × 属性」粒度拆分（如「张三 部门」「研发部 人数」），"
            "不要把一个实体的多个属性（部门/人数/领导）混进同一路查询——混查会稀释召回精度；"
            "查询术语统一用制度用语（领导→部门主管、人数→在职人数/人员规模）；"
            "对比类问题每个实体/方面至少一路；首发计划 1-3 路即可，后续纠错还有机会。"
            "重点：若回答需先解析中间实体再查其属性（如先查「张三在哪个部门」、"
            "再查该部门的人数/领导）——这是链式查询，必须用 multi_hop 并传入完整查询，"
            "由检索器自动规划-执行-验证；不要只用 parallel hybrid 硬拆多实体。"
        )

    @staticmethod
    def _user(query: str) -> str:
        return f"用户问题：{query}\n请输出规划 JSON。"


class GraderAgent:
    """评审角色（CRAG）：逐条证据相关性评分 + 归纳缺失事实。

    prior_hits：先前轮累积证据（orchestrator 传入 state.evidence）——缺失事实归纳
    必须结合先前已确认的证据，否则只看当前波会「遗忘」前面确认的事实（如已确认
    张三=研发部，当前波未含该信息时仍把部门归属报为缺失，触发重复检索）。
    """

    def run(
        self,
        query: str,
        hits: list[dict[str, Any]],
        state,
        use_llm: bool = True,
        prior_hits: list[dict[str, Any]] | None = None,
    ) -> GradeOutcome:
        if use_llm and hits:
            def parse(data: dict[str, Any]) -> GradeOutcome | None:
                raw = data.get("relevant")
                if not isinstance(raw, list):
                    return None
                keep = sorted({int(i) for i in raw if isinstance(i, (int, float)) and 0 <= int(i) < len(hits)})
                missing = [str(m) for m in (data.get("missing_facts") or []) if str(m).strip()]
                return GradeOutcome(
                    thought=str(data.get("reason") or ""),
                    keep=keep,
                    missing_facts=missing[:4],
                )

            outcome = _llm_json(
                ROLE_GRADE, state, self._system(), self._user(query, hits, prior_hits or []), parse
            )
            if outcome is not None:
                return outcome
        # 规则回退：查询与证据 2 字词共现 ≥2 或分数 ≥0.5 视为相关（宽松，防误杀）
        q_terms = None
        keep: list[int] = []
        from app.rag.agentic.tools import terms2

        q_terms = terms2(query)
        for i, h in enumerate(hits):
            overlap = len(q_terms & terms2(h.get("text") or ""))
            if overlap >= 2 or (h.get("score") or 0.0) >= 0.5:
                keep.append(i)
        return GradeOutcome(
            thought="规则回退：词法共现粗评", keep=keep, note="评审规则回退",
        )

    @staticmethod
    def _system() -> str:
        return (
            "你是 RAG 系统的证据评审角色（Corrective RAG 的检索评估器）。"
            "逐条判断证据对回答用户问题是否相关：支持回答或提供背景都算相关。\n"
            '输出严格 JSON：{"relevant": [相关证据的下标数组], '
            '"missing_facts": ["仍缺失的关键事实（最多 4 条，无则空数组）"], "reason": "简短理由"}\n'
            "规则：仅剔除与用户问题毫无主题关联的明显无关证据；"
            "不确定是否相关的一律保留——宁可多给证据让生成器自行取舍，也不要误杀支撑答案的条款。"
            "缺失事实仅指『候选证据与先前已确认证据都不支持』的关键事实；"
            "先前证据已支持的事实（如某人的部门归属）勿再报为缺失。"
        )

    @staticmethod
    def _user(query: str, hits: list[dict[str, Any]], prior_hits: list[dict[str, Any]]) -> str:
        lines = [f"{i}. {(h.get('metadata') or {}).get('volume') or ''}：{(h.get('text') or '')[:120]}" for i, h in enumerate(hits)]
        body = f"用户问题：{query}\n\n候选证据：\n" + "\n".join(lines)
        if prior_hits:
            prior = "\n".join(
                f"· {(h.get('metadata') or {}).get('volume') or ''}：{(h.get('text') or '')[:80]}" for h in prior_hits[:5]
            )
            body += f"\n\n先前轮已确认的证据（缺失事实归纳时请结合，勿再把其中已支持的事实报为缺失）：\n{prior}"
        return body + "\n\n请输出评审 JSON。"


class CorrectorAgent:
    """纠错角色（CRAG correct 分支）：证据不足时的纠错决策——下一波工具调用。

    prior_hits：先前轮累积证据（orchestrator 传入 state.evidence）——下一跳查询
    生成必须结合已确认证据，把已解析出的实体/结论拼进新查询（如首轮确认「张三=研发部」，
    下一跳应为「研发部 在职人数」而非宽泛「部门人数」）；否则盲写查询查不准/查回同类，
    导致纠错空转（correction_success_rate 低）。
    """

    _ACTIONS = (ACTION_HYBRID, "search", ACTION_VOLUME, ACTION_MULTI_HOP)

    def run(
        self,
        query: str,
        missing_facts: list[str],
        executed: list[ToolCallSpec],
        catalog: list[str],
        state,
        use_llm: bool = True,
        prior_hits: list[dict[str, Any]] | None = None,
    ) -> CorrectOutcome:
        if use_llm:
            def parse(data: dict[str, Any]) -> CorrectOutcome | None:
                raw = data.get("calls")
                if not isinstance(raw, list) or not raw:
                    return None
                calls: list[ToolCallSpec] = []
                for c in raw[:3]:
                    if not isinstance(c, dict):
                        continue
                    action = str(c.get("action") or "").strip()
                    if action not in self._ACTIONS:
                        continue
                    calls.append(
                        ToolCallSpec(
                            action=action,
                            query=normalize_query(str(c.get("query") or "").strip() or query),
                            volume=str(c.get("volume") or "").strip(),
                            reason=str(c.get("reason") or ""),
                        )
                    )
                return CorrectOutcome(thought=str(data.get("reason") or ""), calls=calls)

            outcome = _llm_json(
                ROLE_CORRECT, state, self._system(executed),
                self._user(query, missing_facts, prior_hits or []), parse,
            )
            if outcome is not None:
                return outcome
        # 规则回退：针对缺失事实构造纠错波——优先定向卷（路由提示未用尽时），再换表述 hybrid
        calls: list[ToolCallSpec] = []
        used_volumes = {c.volume for c in executed if c.action == ACTION_VOLUME}
        hint = next((v for v in catalog if v not in used_volumes), None)
        facts = missing_facts or [query]
        for fact in facts[:2]:
            if hint and not calls:
                calls.append(ToolCallSpec(ACTION_VOLUME, normalize_query(fact), hint, "规则纠错：定向卷补检索"))
            else:
                calls.append(ToolCallSpec(ACTION_HYBRID, normalize_query(fact), "", "规则纠错：换表述混合检索"))
        if not calls:
            calls.append(ToolCallSpec(ACTION_MULTI_HOP, normalize_query(query), "", "规则纠错：多跳兜底"))
        return CorrectOutcome(thought="规则回退：按缺失事实补检索", calls=calls, note="纠错规则回退")

    @staticmethod
    def _system(executed: list[ToolCallSpec]) -> str:
        done = "；".join(f"[{c.action}] {c.query}" + (f"@{c.volume}" if c.volume else "") for c in executed) or "（无）"
        return (
            "你是 RAG 系统的纠错角色（Corrective RAG 的纠正分支）。当前证据不足以回答，"
            "请针对缺失事实给出下一波检索调用（换表述/换工具，禁止重复已执行调用）。\n"
            f"已执行调用：{done}\n"
            "可用工具：search（纯向量）/ hybrid（语义+关键词，默认首选）/ "
            "multi_hop（实体链/流程链）。默认跨卷检索（hybrid 首选）；"
            "仅当问题明确点名某卷时才用 volume_search（如「卷十 FAQ 里…」）。\n"
            "提示：用户消息中会给出「先前轮已确认的证据」。下一跳查询应直接使用其中已解析出的"
            "实体/结论推进——例如首轮已确认「张三在研发部」，缺口是「部门人数」，下一跳应为"
            "「研发部 在职人数」，而不是宽泛的「部门人数」；也不要重复查询已确认的证据已覆盖的事实。\n"
            '输出严格 JSON：{"calls": [{"action": "search|hybrid|volume_search|multi_hop", '
            '"query": "新查询表述", "volume": "卷名或空串", "reason": "针对哪个缺口"}], "reason": "纠错思路"}\n'
            "规则：1-3 路即可；每路针对一个缺失事实；volume_search 仅用于问题明确点名的卷。"
        )

    @staticmethod
    def _user(query: str, missing_facts: list[str], prior_hits: list[dict[str, Any]]) -> str:
        facts = "\n".join(f"- {m}" for m in missing_facts) or "- （未归纳，请围绕原问题换路）"
        body = f"用户问题：{query}\n\n缺失事实：\n{facts}"
        if prior_hits:
            prior = "\n".join(
                f"· {(h.get('metadata') or {}).get('volume') or ''}：{(h.get('text') or '')[:80]}"
                for h in prior_hits[:5]
            )
            body += f"\n\n先前轮已确认的证据（生成下一跳查询时请结合，直接使用其中已解析的实体/结论）：\n{prior}"
        return body + "\n\n请输出纠错 JSON。"


class VerifierAgent:
    """校验角色（Self-RAG）：事实-证据支持度矩阵，判定可答/缺口。

    confirmed_facts：先前轮已确认的事实（orchestrator 校验后写回）——跨轮记忆：
    只对「尚未确认」的事实判定缺失，已确认的不再重复报缺失（防多轮遗忘，
    尤其规则回退只认词法时，抽象事实会被误判）。
    """

    def run(
        self,
        query: str,
        facts: list[str],
        hits: list[dict[str, Any]],
        state,
        use_llm: bool = True,
        confirmed_facts: list[str] | None = None,
    ) -> VerifyOutcome:
        confirmed = [f for f in (confirmed_facts or []) if f in (facts or [query])]
        if use_llm:
            def parse(data: dict[str, Any]) -> VerifyOutcome | None:
                if "answerable" not in data:
                    return None
                missing = [
                    m for m in (data.get("missing_facts") or [])
                    if str(m).strip() and str(m) not in confirmed
                ]
                return VerifyOutcome(
                    thought=str(data.get("reason") or ""),
                    answerable=bool(data.get("answerable")),
                    missing_facts=missing[:4],
                )

            outcome = _llm_json(
                ROLE_VERIFY, state, self._system(),
                self._user(query, facts, hits, confirmed), parse,
            )
            if outcome is not None:
                return outcome
        # 规则回退：每条未确认事实的 2 字词 ≥50% 被任一证据覆盖 → 该事实受支持
        # （已确认事实直接跳过，不再遍历——词法对抽象事实天然误判，跨轮记忆兜底）
        missing: list[str] = []
        from app.rag.agentic.tools import terms2

        for fact in facts or [query]:
            if fact in confirmed:
                continue
            f_terms = terms2(fact)
            if not f_terms:
                continue
            covered = any(
                len(f_terms & terms2(h.get("text") or "")) / len(f_terms) >= 0.5 for h in hits
            )
            if not covered:
                missing.append(fact)
        answerable = bool(hits) and not missing
        return VerifyOutcome(
            thought="规则回退：词法覆盖校验", answerable=answerable,
            missing_facts=missing, note="校验规则回退",
        )

    @staticmethod
    def _system() -> str:
        return (
            "你是 RAG 系统的答案校验角色（Self-RAG 的支持度校验）。"
            "对照事实清单逐条判断：当前证据是否足以回答用户问题。\n"
            '输出严格 JSON：{"answerable": true/false, '
            '"missing_facts": ["证据不支持的事实（最多 4 条，可答则空数组）"], "reason": "简短理由"}\n'
            "规则：\n"
            "1) 证据能支撑用户问题所需的答案核心信息即 answerable=true，无需逐字对应；\n"
            "2) 只把『用户问题明确要求、且证据完全没有』的关键事实列为缺失；\n"
            "3) 不要脑补额外要求（如精确计算基数、工龄差异化比例等原问题未要求的细节）；\n"
            "4) 证据已能支撑的部分视为支持；不猜测、不脑补证据中不存在的事实；\n"
            "5) 用户消息中标注『先前轮已确认的事实』一律视为已被支持，不得再报为缺失；\n"
            "6) 按用户问题的字面粒度判定即可：问『领导/负责人/主管是谁』时，证据中出现该部门的"
            "负责人/主管/领导姓名即视为已支持——不要强求用户未问到的『直接上级/汇报关系』等更细粒度；"
            "证据指向具体人员姓名即算支持，名称旁的『（兼）』等备注不影响其作为负责人身份的确认。"
        )

    @staticmethod
    def _user(query: str, facts: list[str], hits: list[dict[str, Any]], confirmed: list[str]) -> str:
        # 证据池跨轮累积，逐条全部呈现；evidence 文本已被 compressor 压缩（≤400 字符），
        # 此处不再二次截断——否则表格类证据（如"部门|在职人数|研发部130|产品部120"）后半行被切，
        # Verifier 会把已支持的事实误判为缺失（产品部人数"看不到"问题）。
        fact_lines = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts or [query]))
        ev_lines = "\n".join(f"{i+1}. {(h.get('text') or '')}" for i, h in enumerate(hits))
        body = (
            f"用户问题：{query}\n\n回答所需事实：\n{fact_lines}\n\n当前证据：\n{ev_lines or '（无）'}"
        )
        if confirmed:
            body += "\n\n先前轮已确认的事实（视为已支持，不得再报缺失）：\n" + "\n".join(
                f"- {f}" for f in confirmed
            )
        return body + "\n\n请输出校验 JSON。"
