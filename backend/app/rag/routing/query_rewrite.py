"""Query 重写：把用户口语化/含混问题改写成更贴近语料的检索查询（Multi-Query）。

- LLMQueryRewriter：按命名场景 rag_rewrite 懒取聊天模型（模型/参数见 service.DEFAULT_PROFILES），
  一次生成多个查询变体（工业界 Multi-Query 做法）；
- RuleQueryRewriter：无 LLM（离线/仅配 Embedding）时的确定性回退：去客套语 + 关键词变体。

检索方拿到多个变体分别召回再融合，弥补「单次查询覆盖不全」的缺陷。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model

# 客套/疑问前缀：去掉后得到更接近检索词的表达
_POLITE_PREFIX = re.compile(r"^(请问|我想知道|我想问|麻烦问一下|帮我查一下|帮忙查一下|咨询一下|了解一下|请教一下|能不能告诉我|可以告诉我)")

# 领域同义词表：口语化说法 → 语料用词（用于规则改写时扩展）
_SYNONYMS: dict[str, str] = {
    "差旅": "出差",
    "出差费": "出差",
    "报销凭证": "报销",
    "发票": "报销",
    "凭证": "报销",
    "罚": "处罚",
    "休假": "年假",
    "请假": "事假",
    "薪资": "工资",
    "头等舱": "机票",
    "住宿费": "住宿",
    "餐补": "餐补",
    "审批人": "审批",
    "审核": "审批",
}

# 领域关键词：出现在查询中即作为关键词召回信号（含语料高频专有名词）
_KEYWORDS = [
    "考勤", "打卡", "迟到", "早退", "旷工", "补卡",
    "年假", "事假", "病假", "福利", "补贴",
    "出差", "差旅", "报销", "住宿", "交通", "餐补", "发票",
    "审批", "工资", "绩效", "城市", "一线", "二线",
]


class QueryRewriter(ABC):
    """查询改写抽象：把一个问题扩展为多个检索查询变体。"""

    @abstractmethod
    def rewrite(self, query: str, memory: str | None = None) -> list[str]:
        """返回查询变体列表（首个为原始查询，保证基础召回）。

        memory：L2 主动语义召回的用户记忆块（背景参考），追加进提示词辅助个性化改写。
        """


class LLMQueryRewriter(QueryRewriter):
    """LLM 改写：按场景懒取模型，一次调用生成 N 个更贴近制度语料的查询变体。

    始终把原始查询置于首位：即使 LLM 输出不佳，原查询也一定参与召回。
    """

    # 本阶段使用的模型/参数场景（qwen3.5-flash / temp=0.3 / max_tokens=200 / thinking=False，见 service.DEFAULT_PROFILES）
    scenario = "rag_rewrite"

    def __init__(self, variants: int = 3):
        self.variants = max(1, int(variants))

    def rewrite(self, query: str, memory: str | None = None) -> list[str]:
        parsed: list[str] = []
        llm = get_chat_model(self.scenario)
        if llm is not None:
            try:
                hint = (
                    f"\n\n相关用户记忆（仅作背景参考，可能与问题无关）：\n{memory}"
                    if memory
                    else ""
                )
                messages = [
                    SystemMessage(
                        content=(
                            f"你是企业知识库检索助手。请把用户的问题改写成 {self.variants} 个"
                            "更适合检索企业制度语料的查询变体：保留关键实体与数字，"
                            "用更正式、贴近规章制度原文的措辞；若原问题已足够正式，可做关键词化压缩。"
                            "相关用户记忆可帮助补充省略的信息（如「上次那个」对应的具体名词），"
                            "但仅作背景参考，不得把记忆内容当作制度事实写进变体。"
                            "每个变体单独一行输出，不要编号、不要解释、不要其他文字。"
                            f"{hint}"
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
            line = line.strip().strip("。-·*`\"'").strip()
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


class RuleQueryRewriter(QueryRewriter):
    """确定性规则改写（无 LLM 回退）：去客套语 + 关键词化变体。"""

    def rewrite(self, query: str, memory: str | None = None) -> list[str]:
        # 规则改写无提示词，memory 仅作接口兼容（供上层统一传参）
        stripped = _POLITE_PREFIX.sub("", query).strip()
        keyword = self._keyword_query(query)
        variants = [query]
        if stripped and stripped != query:
            variants.append(stripped)
        if keyword and keyword != query and keyword != stripped:
            variants.append(keyword)
        return variants

    @staticmethod
    def _keyword_query(query: str) -> str:
        """提取查询中的领域关键词 + 数字，生成紧凑关键词查询。"""
        tokens = []
        for word in _KEYWORDS:
            if word in query and word not in tokens:
                tokens.append(word)
        for syn, canonical in _SYNONYMS.items():
            if syn in query and canonical not in tokens:
                tokens.append(canonical)
        numbers = re.findall(r"\d+", query)
        tokens.extend(numbers)
        return " ".join(tokens) if tokens else ""


def build_rewriter(variants: int = 3) -> QueryRewriter:
    """构造改写器：有 LLM 场景配置用 LLM 改写（内部懒取），否则规则回退。"""
    return LLMQueryRewriter(variants=variants)
