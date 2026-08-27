"""HyDE（Hypothetical Document Embeddings）：为查询生成一段"假想答案文档"，用文档语义做稠密检索。

对齐 Modular RAG 企业级架构的「预处理模块组-查询扩展（6.1）」：
直接向量化查询常因查询过短、措辞与语料不一致而召回偏差，HyDE 先让 LLM 生成一段
可能的答案文档，再对该文档做向量检索（doc-space 检索），弥补「查询空间」与「文档空间」
不一致的问题。

- LLMHydeExpander：按命名场景 rag_hyde 懒取聊天模型（模型/参数见 service.DEFAULT_PROFILES），
  生成一段假设性答案文档；
- RuleHydeExpander：无 LLM（离线/仅配 Embedding）时的确定性回退：返回原查询（no-op），
  调用方据此跳过额外检索，避免降级空跑。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model


class HydeExpander(ABC):
    """HyDE 扩展抽象：把查询扩展为一段假想文档（用于稠密 doc-space 检索）。"""

    @abstractmethod
    def expand(self, query: str) -> str:
        """返回假想答案文档；无 LLM 时回退原查询（no-op）。"""


class LLMHydeExpander(HydeExpander):
    """LLM HyDE：按场景懒取模型，生成一段面向知识库语料的假设性答案文档。

    LLM 失败 / 未配置 Key 时回退原查询（no-op），保证检索不因 HyDE 而劣化。
    """

    # 本阶段使用的模型/参数场景（qwen3.5-flash / temp=0.3 / max_tokens=300 / thinking=False，见 service.DEFAULT_PROFILES）
    scenario = "rag_hyde"

    def expand(self, query: str) -> str:
        llm = get_chat_model(self.scenario)
        if llm is None:
            return query
        try:
            messages = [
                SystemMessage(
                    content=(
                        "你是企业制度知识库的检索助手。请针对用户的问题，写一段"
                        "「假设性的答案文档」：按规章制度的口吻，把可能出现在制度原文中、"
                        "与该问题相关的关键信息（流程/时限/部门/条件/责任人等）写成一段"
                        "连贯的说明文字，不要提问、不要评价、不要输出其他内容。"
                    )
                ),
                HumanMessage(content=query),
            ]
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            doc = content.strip()
            return doc or query
        except Exception:  # noqa: BLE001 — HyDE 失败时回退原查询，不影响检索
            return query


class RuleHydeExpander(HydeExpander):
    """确定性规则回退：返回原查询（no-op）——无 LLM 时 HyDE 不改变检索，避免降级劣化。"""

    def expand(self, query: str) -> str:
        return query


def build_hyde() -> HydeExpander:
    """构造 HyDE 扩展器：有 LLM 场景配置用 LLM 生成（内部懒取），否则规则回退。"""
    return LLMHydeExpander()
