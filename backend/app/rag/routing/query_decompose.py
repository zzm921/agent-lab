"""查询分解（Query Decomposition）：把对比/多实体复杂问题拆成可独立检索的子问题。

对齐 Modular RAG 企业级架构的「预处理模块组-查询分解（4.5）」：
单跳检索只能回答「X 是什么」，无法回答「X 和 Y 的区别」这类多步推理问题；
分解就是把复杂问题降维为若干可独立检索的简单子问题，检索后合并回答。

- LLMQueryDecomposer：按命名场景 rag_decompose 懒取聊天模型（模型/参数见 service.DEFAULT_PROFILES），
  一次生成多个可独立检索的子问题（每行一个）；无模型/异常时至少保留原始查询；
- RuleQueryDecomposer：无 LLM（离线/仅配 Embedding）时的确定性规则回退：
  命中对比信号（对比/比较/区别/不同/差异）时按 和/与/以及/、 切出实体段，
  为每段补「的规则/规定」形成独立检索目标；首位始终保留原始查询保证基础召回。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model

# 对比信号：命中才做切分解（与 classifier 的 _COMPARISON 对齐）
_COMPARISON = re.compile(r"(对比|比较|区别|不同|差异|有什么不同|哪个更)")

# 连接词：多实体并列的分隔点
_SEPARATORS = re.compile(r"[和与以及、及]")


class QueryDecomposer(ABC):
    """查询分解抽象：把一个复杂问题拆成多个可独立检索的子问题。"""

    @abstractmethod
    def decompose(self, query: str) -> list[str]:
        """返回子问题列表（首位为原始查询，保证基础召回）。"""


class LLMQueryDecomposer(QueryDecomposer):
    """LLM 分解：按场景懒取模型，一次调用生成多个可独立检索的子问题（每行一个）。"""

    # 本阶段使用的模型/参数场景（qwen3.5-flash / temp=0.3 / max_tokens=300 / thinking=False，见 service.DEFAULT_PROFILES）
    scenario = "rag_decompose"

    def __init__(self):
        self._fallback = RuleQueryDecomposer()

    def decompose(self, query: str) -> list[str]:
        parsed: list[str] = []
        llm = get_chat_model(self.scenario)
        if llm is not None:
            try:
                messages = [
                    SystemMessage(
                        content=(
                            "你是检索系统的查询分解器。请把用户的复杂问题拆分为若干个"
                            "可以独立在知识库中检索到答案的简单子问题。要求：\n"
                            "1. 每个子问题可独立检索；\n"
                            "2. 子问题之间尽量无重叠；\n"
                            "3. 所有子问题答案合并后可回答原始问题。\n"
                            "每个子问题单独一行输出，不要编号、不要解释、不要其他文字。"
                        )
                    ),
                    HumanMessage(content=query),
                ]
                resp = llm.invoke(messages)
                content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
                parsed = self._parse_lines(content)
            except Exception:  # noqa: BLE001 — LLM 失败时至少保留原始查询
                parsed = []
        return self._dedupe([query, *parsed])

    @staticmethod
    def _parse_lines(content: str) -> list[str]:
        out = []
        for line in content.splitlines():
            line = re.sub(r"^\s*\d+[.、)）]\s*", "", line).strip()
            if not line:
                continue
            out.append(line)
        return out

    @staticmethod
    def _dedupe(queries: list[str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for q in queries:
            if not q or q in seen:
                continue
            seen.add(q)
            result.append(q)
        return result


class RuleQueryDecomposer(QueryDecomposer):
    """确定性规则分解（无 LLM 回退）：对比信号下按连接词切实体段，补检索目标。"""

    def decompose(self, query: str) -> list[str]:
        if not _COMPARISON.search(query):
            return [query]
        parts = [p.strip() for p in _SEPARATORS.split(query) if p.strip()]
        sub_queries = [query]  # 首位保留原始查询
        for part in parts:
            # 去掉对比词残留，避免「区别/不同」成为独立检索目标
            cleaned = _COMPARISON.sub("", part).strip()
            if not cleaned or cleaned == query:
                continue
            sub_queries.append(self._target(cleaned))
        return self._dedupe(sub_queries)

    @staticmethod
    def _target(segment: str) -> str:
        """实体段补检索目标：过短的词补「的规则/规定」以贴近语料章节标题。"""
        if len(segment) <= 4:
            return f"{segment}的规则"
        return segment

    @staticmethod
    def _dedupe(queries: list[str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for q in queries:
            if not q or q in seen:
                continue
            seen.add(q)
            result.append(q)
        return result


def build_decomposer() -> QueryDecomposer:
    """构造分解器：有 LLM 场景配置用 LLM 分解（内部懒取），否则规则回退。"""
    return LLMQueryDecomposer()
