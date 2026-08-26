"""多跳检索验证器（MultiHop Verifier）：规划-执行-验证的「验证」质量闸门。

对齐 Modular RAG 企业级架构的「验证/质量闸门」：执行完计划后对表——
计划里的每个目标维度（target）是否已被已召回证据覆盖？缺口则生成"补缺子查询"
供执行器局部修正（而非整盘重来）；超预算则如实上报缺口（可观测）。

- LLMMultiHopVerifier：按命名场景 rag_verify 懒取聊天模型（模型/参数见 service.DEFAULT_PROFILES），
  输出 {covered, missing, patched} 结构化 JSON，无模型（未配 Key）/解析失败回退规则；
- RuleMultiHopVerifier：无 LLM（离线/仅配 Embedding）时的确定性回退——
  实体命中或"本步新领域词已全在证据中"判定覆盖；补缺查询用「顺藤摸瓜」关键词扩展
  （与规则多跳迭代一致的思路），有新材料才补。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model
from app.rag.retrieval.iterative_retrieval import HopPlan, VerifyResult, _KEYWORDS, step_covered


class MultiHopVerifier(ABC):
    """多跳验证器抽象：输入原始问题、计划与已累计证据，输出覆盖对表 + 补缺子查询。"""

    @abstractmethod
    def verify(self, query: str, plan: HopPlan, evidence_hits: list[dict]) -> VerifyResult:
        """返回 VerifyResult：covered（已覆盖目标）/ missing（缺口）/ patched（补缺子查询）。"""


class LLMMultiHopVerifier(MultiHopVerifier):
    """LLM 验证：按场景懒取模型，一次调用输出结构化对表 JSON；异常回退规则验证器。

    覆盖对表是轻量决策调用（判断目标是否已被证据覆盖、缺口补缺），
    场景 rag_verify 关闭思考模式，避免决策型调用拖慢多跳链路。
    """

    # 本阶段使用的模型/参数场景（qwen3.5-flash / temp=0.2 / max_tokens=400 / thinking=False，见 service.DEFAULT_PROFILES）
    scenario = "rag_verify"

    def __init__(self):
        self._fallback = RuleMultiHopVerifier()

    def verify(self, query: str, plan: HopPlan, evidence_hits: list[dict]) -> VerifyResult:
        llm = get_chat_model(self.scenario)
        if llm is None:
            return self._fallback.verify(query, plan, evidence_hits)
        try:
            # 证据按条放宽到 800 字符：多跳累积证据多为表格型明细（花名册/部门规模表），
            # 关键行可能位于 200 字符之外；过小截断会把已检索到的事实误判为缺口
            #（与 answerability.py 对齐，避免同类回归）。
            evidence_text = "\n".join(
                f"- {h.get('text', '')[:800]}" for h in evidence_hits
            ) or "（暂无）"
            steps_text = "\n".join(
                f"- target={s.target} | 子查询={s.query}" for s in plan.steps
            )
            messages = [
                SystemMessage(
                    content=(
                        "你是多跳检索的「质量验证器」。给定原始问题、子查询计划（含目标）"
                        "与已累计检索到的内容，判断：\n"
                        "1. 对计划中的每个 target，判断其要解决的事实/环节是否已被已检索内容覆盖"
                        "（该事实或等价信息是否已出现）；\n"
                        "2. covered：已覆盖的 target 列表；missing：未覆盖的 target 列表；\n"
                        "3. 对每个 missing target，若仍有可能检索到，生成一个聚焦该缺口的、"
                        "可独立检索的补缺子查询 patched（每项 {target, query}）；\n"
                        "4. 若已检索内容足以覆盖全部 target，patched 输出 []。\n"
                        "输出必须严格是以下 JSON（不要输出任何其他文字）：\n"
                        '{"covered": ["..."], "missing": ["..."], '
                        '"patched": [{"target": "...", "query": "..."}]}'
                    )
                ),
                HumanMessage(
                    content=f"原始问题：{query}\n子查询计划：\n{steps_text}\n已累计检索到的内容：\n{evidence_text}"
                ),
            ]
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            return self._parse(content)
        except Exception:  # noqa: BLE001 — 模型抖动时回退规则验证
            return self._fallback.verify(query, plan, evidence_hits)

    @staticmethod
    def _parse(content: str) -> VerifyResult:
        """提取 JSON 并归一化字段；非法/不可解析抛异常交给调用方回退。"""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("LLM 验证输出中未找到 JSON")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("验证 JSON 必须是对象")

        def _as_list(key: str) -> list[str]:
            items = data.get(key) or []
            return [str(i).strip() for i in items if str(i).strip()]

        patched = []
        for item in data.get("patched") or []:
            if isinstance(item, dict) and str(item.get("query") or "").strip():
                patched.append(
                    {"target": str(item.get("target") or "").strip(), "query": str(item.get("query") or "").strip()}
                )
        return VerifyResult(covered=_as_list("covered"), missing=_as_list("missing"), patched=patched)


class RuleMultiHopVerifier(MultiHopVerifier):
    """确定性规则验证（无 LLM 回退）。

    覆盖判定：可预判实体已出现在证据中，或本步相对原查询引入的新领域词已全部在证据中出现；
    补缺：顺藤摸瓜——取最近命中里「原查询未含」的领域词拼回原查询，有新材料才补。
    """

    def verify(self, query: str, plan: HopPlan, evidence_hits: list[dict]) -> VerifyResult:
        evidence = "\n".join(h.get("text", "") or "" for h in evidence_hits)
        covered: list[str] = []
        missing: list[str] = []
        patched: list[dict] = []
        for step in plan.steps:
            if self._covered(step, query, evidence):
                covered.append(step.target)
            else:
                missing.append(step.target)
                patch_query = self._patch(query, evidence_hits)
                if patch_query:
                    patched.append({"target": step.target, "query": patch_query})
        return VerifyResult(covered=covered, missing=missing, patched=patched)

    @staticmethod
    def _covered(step, original: str, evidence: str) -> bool:
        """覆盖判定：复用执行器的内容级覆盖检测（实体命中或新领域词全命中），
        保证验证器终局对表与执行器逐跳复用跳过判定口径一致，防重复查已解决事实。"""
        return step_covered(step, original, evidence)

    @staticmethod
    def _patch(query: str, hits: list[dict]) -> str | None:
        """顺藤摸瓜：取最近命中里「原查询未含」的领域词，拼回原查询作为补缺子查询。"""
        if not hits:
            return None
        base = [kw for kw in _KEYWORDS if kw in query]
        top_text = hits[0].get("text", "") or ""
        trail = [kw for kw in _KEYWORDS if kw in top_text and kw not in base]
        if not trail:
            return None
        next_query = " ".join(base + trail)
        if next_query == query:
            return None
        return next_query


def build_verifier() -> MultiHopVerifier:
    """构造验证器：有 LLM 场景配置用 LLM 验证（内部懒取），否则规则回退。"""
    return LLMMultiHopVerifier()
