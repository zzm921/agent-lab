"""指代消解（Deictic Resolution）：把依赖上文的指代词替换为具体实体，得到可独立检索的查询。

Modular RAG 前置检索的第一步：RAG 是独立检索阶段，只会拿到当前用户消息，若不消解，
「他的年假有多少天」这类指代问题会因拿不到上文而无法定位实体（多跳规划器也会报「无上下文」）。

- LLMDeicticResolver：按场景 rag_rewrite 懒取聊天模型，结合会话上下文把
  「他/她/它/这个/那个…」替换为上下文中的具体实体；上下文不足以消解或原文无指代时原样返回。
- RuleDeicticResolver：无 LLM（离线）兜底——指代消解属语义判定，不做规则猜测，原样返回
  （离线不消解也不误伤，由检索/改写自行降级）。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model

# 指代词/指代短语信号：命中才触发消解（避免无谓的 LLM 调用）
_DEICTIC = re.compile(
    r"他|她|它|他们|她们|它们|这个|那个|这些|那些|"
    r"该员工|该领导|该部门|该同事|这里|那里|这人|这位"
)
# 句首「这/那」开头（如「那补卡流程呢」）也属省略/指代，需结合上下文
_DEICTIC_PREFIX = re.compile(r"^(这|那)")


class DeicticResolver(ABC):
    """指代消解抽象：输入当前问题 + 会话上下文，输出无指代、可独立检索的查询。"""

    @abstractmethod
    def resolve(self, query: str, context: str | None, memory: str | None = None) -> str:
        """返回消解后的查询；无指代/无上下文/无法消解时原样返回。

        memory：L2 主动语义召回的用户记忆块（背景参考），追加进提示词辅助定位指代对象。
        """


class LLMDeicticResolver(DeicticResolver):
    """LLM 指代消解：根据会话上下文替换指代词，输出一条可独立检索的查询（单行）。"""

    # 复用 Query 改写场景（qwen3.5-flash / temp=0.3 / max_tokens=200 / thinking=False，见 service.DEFAULT_PROFILES）
    scenario = "rag_rewrite"

    def resolve(self, query: str, context: str | None, memory: str | None = None) -> str:
        if not context or not (_DEICTIC.search(query) or _DEICTIC_PREFIX.match(query)):
            return query
        llm = get_chat_model(self.scenario)
        if llm is None:
            return query
        try:
            hint = (
                f"\n\n相关用户记忆（仅作背景参考，可能与当前问题无关）：\n{memory}"
                if memory
                else ""
            )
            messages = [
                SystemMessage(
                    content=(
                        "你是企业知识库检索系统的指代消解器。用户当前问题可能含指代词"
                        "（他/她/它/这个/那个/该…），需要根据下方对话上下文把它替换为具体实体，"
                        "输出一条**不含指代词、可独立检索**的查询。\n"
                        "规则：\n"
                        "- 指代对象**优先取上一轮助手回答中「给出/确定」的实体**"
                        "（即上轮用户问题的答案对象），而不是上轮用户问题的主语。"
                        "例如：用户问「张三的领导是谁」、助手答「是王刚」时，"
                        "后续「他的年假有多少天」中的「他」指王刚，应改写为「王刚的年假有多少天」；\n"
                        "- 相关用户记忆可用于辅助定位指代对象（如「上次说的那个流程」），"
                        "但仅作参考，与对话上下文冲突时以对话上下文为准；\n"
                        "- 上下文不足以确定指代对象，或问题本来就没有指代 → 原样输出当前问题；\n"
                        "- 只输出一条查询，不要解释、不要其他文字。\n\n"
                        f"对话上下文：\n{context}{hint}"
                    )
                ),
                HumanMessage(content=f"当前问题：{query}"),
            ]
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            resolved = content.strip().strip("。-·*`\"'").strip()
            return resolved or query
        except Exception:  # noqa: BLE001 — LLM 失败时原样返回，不阻断检索
            return query


class RuleDeicticResolver(DeicticResolver):
    """确定性兜底：指代消解属语义判定，不做规则猜测——原样返回（离线不消解也不误伤）。"""

    def resolve(self, query: str, context: str | None, memory: str | None = None) -> str:
        return query


def build_deictic_resolver() -> DeicticResolver:
    """构造指代消解器：有 LLM 场景配置用 LLM 消解（内部懒取），否则规则 no-op 兜底。"""
    return LLMDeicticResolver()
