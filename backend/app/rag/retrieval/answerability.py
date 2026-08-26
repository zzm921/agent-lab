"""检索后答案充分性验证（Answerability Verification）：生成前的质量闸门。

对齐 Modular RAG 企业级架构的「验证/质量闸门 + 多轮追问生成」：检索（含后处理）完成后、
生成之前，判断当前命中的上下文是否足以支撑回答用户问题——而非把可能残缺的上下文
直接丢给主 LLM（否则模型要么答不出、要么凭运气追问、要么编造内部数据）。

本闸门是**跨复杂度路径的统一兜底**（simple / rewrite / decompose / multihop 都经过），
补上多跳路径已有 MultiHopVerifier 但其他路径缺失的覆盖检查：

- answerable：上下文足以回答 → 正常生成；
- insufficient + escalate：信息可能在库里但当前路径漏召回（如单次向量 top_k 未命中
  关键事实片段）→ 升级检索（有界 1 轮，由 modular 编排决定升级目标）；
- insufficient + clarify：知识库确实缺关键事实 → 如实说明 + 追问澄清（不编造）。

- LLMAnswerabilityVerifier：复用轻量验证场景 rag_verify（关闭思考），按场景懒取聊天模型，
  输出 {answerable, missing_facts, recommendation, escalate_to} 结构化 JSON；无模型回退规则；
- RuleAnswerabilityVerifier：无 LLM（离线）时的确定性兜底——关键词覆盖启发式：
  查询中的领域关键词须在命中文本中出现（缺失即判不足）；无领域词时保守视为可答。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model
from app.rag.retrieval.iterative_retrieval import _KEYWORDS

# recommendation（验证结论 → 生成/编排策略）
ANSWER = "answer"  # 上下文足以回答，正常生成
ESCALATE = "escalate"  # 信息可能在库但漏召回，升级检索
CLARIFY = "clarify"  # 知识库缺关键事实，追问澄清

_VALID_RECOMMENDATION = {ANSWER, ESCALATE, CLARIFY}
_VALID_ESCALATE_TO = {"multi_recall", "multihop"}

# 确定性兜底：结构性「部门/公司规模」对比问题（如「张三的部门比李雪的部门哪个人多」）
# 需要库内的部门规模表（在职人数/编制）才能回答；仅凭人员花名册/明细中的零散行无法支撑人数对比。
# 这类查询的判定不依赖 LLM 心情——证据缺规模数据即升级，避免误判可答/追问（回归防护）。
_DEPT_SCALE_QUERY = re.compile(r"(部门|公司|单位)")
_DEPT_SCALE_INTENT = re.compile(r"(人数|人多|多少人|规模|编制|有多少人|比.{0,12}(多|少))")
# 规模证据：表头规范词（在职人数/人员规模/部门人数/部门编制）或明确人数数据（如「研发部 130 人」）
_DEPT_SCALE_EVIDENCE = re.compile(r"(在职人数|人员规模|部门\s*人数|部门\s*编制|\d+\s*人)")


def _needs_dept_scale_escalate(query: str, hits: list[dict[str, Any]]) -> bool:
    """查询含「部门/公司 + 人数/规模」信号、但证据中无部门规模表 → 强制升级（确定性）。"""
    if not _DEPT_SCALE_QUERY.search(query) or not _DEPT_SCALE_INTENT.search(query):
        return False
    evidence = "\n".join(h.get("text", "") or "" for h in hits)
    return not _DEPT_SCALE_EVIDENCE.search(evidence)


@dataclass
class AnswerabilityVerdict:
    """一次检索后的答案充分性判定：是否可答 + 缺失事实 + 处置建议。"""

    answerable: bool = True
    missing_facts: list[str] = field(default_factory=list)  # 缺失的关键事实/信息（可观测）
    recommendation: str = ANSWER  # answer / escalate / clarify
    escalate_to: str | None = None  # 仅 escalate 时：升级目标（multi_recall / multihop）


def verdict_to_dict(v: AnswerabilityVerdict) -> dict[str, Any]:
    """AnswerabilityVerdict → dict（供 modular 编排 / 前端事件 / RetrieveResult 使用）。"""
    return {
        "answerable": v.answerable,
        "missing_facts": v.missing_facts,
        "recommendation": v.recommendation,
        "escalate_to": v.escalate_to,
    }


class AnswerabilityVerifier(ABC):
    """答案充分性验证器抽象：输入问题与检索命中，输出是否可答 + 缺失事实 + 处置建议。"""

    @abstractmethod
    def verify(self, query: str, hits: list[dict[str, Any]]) -> AnswerabilityVerdict:
        """返回 AnswerabilityVerdict。"""


class LLMAnswerabilityVerifier(AnswerabilityVerifier):
    """LLM 验证：复用轻量验证场景，一次调用输出结构化 JSON；异常/无模型回退规则验证器。

    充分性判定是轻量决策调用（判断当前命中是否足以回答、缺什么、该升级还是追问），
    复用场景 rag_verify（关闭思考模式），避免决策型调用拖慢检索链路。
    """

    # 复用多跳验证场景（qwen3.5-flash / temp=0.2 / max_tokens=400 / thinking=False，见 service.DEFAULT_PROFILES）
    scenario = "rag_verify"

    def __init__(self):
        self._fallback = RuleAnswerabilityVerifier()

    def verify(self, query: str, hits: list[dict[str, Any]]) -> AnswerabilityVerdict:
        # 确定性兜底优先：部门规模对比类查询缺规模表 → 直接升级检索（不依赖 LLM 判定）。
        # LLM 可能仅凭人员花名册/明细误判「可答」，但人数对比必须靠部门编制规模表支撑，
        # 证据缺「在职人数/人员规模」即强制升级，防止用残缺上下文生成错误答案（回归防护）。
        if _needs_dept_scale_escalate(query, hits):
            return AnswerabilityVerdict(
                answerable=False,
                missing_facts=["部门规模/在职人数（需部门编制规模表）"],
                recommendation=ESCALATE,
                escalate_to="multihop",
            )
        llm = get_chat_model(self.scenario)
        if llm is None:
            return self._fallback.verify(query, hits)
        try:
            # 证据按条放宽到 800 字符：检索命中常为表格型明细（人员花名册/部门规模表/流程表），
            # 关键行可能位于 200 字符之外；过小截断会把已检索到的事实误判为缺失
            #（回归：对比「张三/李雪所在部门人数」时「李雪→产品部」被 200 字符切掉误判缺失）。
            evidence_text = "\n".join(
                f"- {h.get('text', '')[:800]}" for h in hits
            ) or "（未检索到任何相关内容）"
            messages = [
                SystemMessage(
                    content=(
                        "你是 RAG 系统的「答案充分性验证器」。给定用户问题与检索到的知识库内容，"
                        "判断当前内容是否足以支撑回答该问题。你不是问答助手，不回答用户问题。\n"
                        "判定前必须精确定位回答该问题真正需要的关键事实（如问题问的是谁的什么属性，"
                        "只需该实体的对应属性与相关规则），再逐项核对检索内容。\n"
                        "判定规则：\n"
                        "1. answerable（bool）：检索内容是否足以支撑回答用户问题；\n"
                        "2. missing_facts（list）：仅当「回答该问题真正需要的关键事实」确实未出现在"
                        "任何一条检索内容中，才列入（如「王刚的在岗工龄」「具体报销时限」）；"
                        "足以回答则为 []；\n"
                        "3. recommendation（str）：answer=足以回答直接作答；"
                        "escalate=该关键事实很可能在知识库中、只是本次未检索到"
                        "（扩大召回/换更全的检索方式有机会命中），建议升级检索；"
                        "clarify=知识库明显缺乏该关键事实（需用户补充信息），建议追问澄清，"
                        "不要依赖自身知识编造内部人事数据；\n"
                        "4. escalate_to（str，仅 recommendation=escalate 时）："
                        "\"multi_recall\"=升级为多路召回（双路宽召回，针对漏召回）；"
                        "\"multihop\"=升级为多跳检索（缺失的是实体链/流程中间环节）。"
                        "默认 \"multi_recall\"。\n"
                        "硬性约束：\n"
                        "- 只核对回答问题所必需的事实；与回答问题无关的信息（如问题问领导时，"
                        "下属本人的工龄）一律不得列为缺失；\n"
                        "- 检索内容中已出现的事实，即使表述略有差异（如内容写「在岗6年」而缺失项写"
                        "「王刚的在岗工龄」），视为已具备，不得判为缺失；\n"
                        "- 能依据检索内容直接推断出答案时，answerable 必须为 true、"
                        "recommendation 必须为 answer；\n"
                        "- 当问题本身信息完整、缺失的是库内结构化数据（人员花名册、部门编制规模、"
                        "表格型明细）时，应判 recommendation=escalate（升级检索再试一次），"
                        "而不是 clarify——追问用户无助于获取库内本就存在的数据；\n"
                        "示例（仅示范判定逻辑，与当前问题无关）：问题「A 领导的年假天数」，"
                        "内容含「A 在岗6年」「工龄满6年及以上员工年假固定为10天」→ "
                        'answerable=true, missing_facts=[], recommendation="answer"。\n'
                        "输出必须严格是以下 JSON（不要输出任何其他文字）：\n"
                        '{"answerable": true/false, "missing_facts": ["..."], '
                        '"recommendation": "answer|escalate|clarify", '
                        '"escalate_to": "multi_recall|multihop"}'
                    )
                ),
                HumanMessage(content=f"用户问题：{query}\n检索到的知识库内容：\n{evidence_text}"),
            ]
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            return self._parse(content)
        except Exception:  # noqa: BLE001 — 模型抖动时回退规则验证
            return self._fallback.verify(query, hits)

    @staticmethod
    def _parse(content: str) -> AnswerabilityVerdict:
        """提取 JSON 并校验枚举白名单；非法/不可解析抛异常交给调用方回退。"""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("LLM 验证输出中未找到 JSON")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("验证 JSON 必须是对象")
        recommendation = str(data.get("recommendation") or ANSWER)
        if recommendation not in _VALID_RECOMMENDATION:
            raise ValueError("验证 recommendation 枚举值非法")
        escalate_to = data.get("escalate_to")
        if escalate_to is not None and escalate_to not in _VALID_ESCALATE_TO:
            escalate_to = None
        missing = [str(i).strip() for i in (data.get("missing_facts") or []) if str(i).strip()]
        answerable = bool(data.get("answerable", True))
        # 归一化：answerable=True 时不允许 escalate / clarify 并存（可观测一致性）
        if answerable and recommendation != ANSWER:
            recommendation = ANSWER
        return AnswerabilityVerdict(
            answerable=answerable,
            missing_facts=missing,
            recommendation=recommendation,
            escalate_to=escalate_to if recommendation == ESCALATE else None,
        )


class RuleAnswerabilityVerifier(AnswerabilityVerifier):
    """确定性规则验证（无 LLM 回退）。

    覆盖判定：命中为空 → 明确不足（升级）；查询中的领域关键词未在命中文本中出现 →
    缺失对应事实（升级）；无领域词可查（如纯实体/通用查询）→ 保守视为可答
    （规则无法做语义判定，尽量不误伤，主路径由 LLM 验证器把关）。
    """

    def verify(self, query: str, hits: list[dict[str, Any]]) -> AnswerabilityVerdict:
        # 与 LLM 验证器一致：部门规模对比类查询缺规模表 → 直接升级（无模型时的确定性兜底）
        if _needs_dept_scale_escalate(query, hits):
            return AnswerabilityVerdict(
                answerable=False,
                missing_facts=["部门规模/在职人数（需部门编制规模表）"],
                recommendation=ESCALATE,
                escalate_to="multihop",
            )
        if not hits:
            return AnswerabilityVerdict(
                answerable=False,
                missing_facts=["未检索到任何相关内容"],
                recommendation=ESCALATE,
                escalate_to="multi_recall",
            )
        evidence = "\n".join(h.get("text", "") or "" for h in hits)
        query_kws = [kw for kw in _KEYWORDS if kw in query]
        if query_kws:
            missing = [kw for kw in query_kws if kw not in evidence]
            if missing:
                return AnswerabilityVerdict(
                    answerable=False,
                    missing_facts=[f"缺少关于「{kw}」的检索内容" for kw in missing],
                    recommendation=ESCALATE,
                    escalate_to="multi_recall",
                )
        return AnswerabilityVerdict(
            answerable=True, missing_facts=[], recommendation=ANSWER, escalate_to=None
        )


def build_answerability_verifier() -> AnswerabilityVerifier:
    """构造答案充分性验证器：有 LLM 场景配置用 LLM 验证（内部懒取），否则规则回退。"""
    return LLMAnswerabilityVerifier()
