"""RAG 知识库检索工具：查询向量化后在语料库中检索 top-k 并返回带相关度的片段。"""
from langchain_core.tools import tool


def make_rag_tool(vector_store, top_k: int = 3, emit=None):
    """构建知识库检索工具；提供 emit 时可推送 retrieve 事件。"""

    @tool
    def knowledge_search(query: str) -> str:
        """在平台内置知识库中检索与查询最相关的资料片段，返回带相关度分数的内容。"""
        hits = vector_store.search(query, top_k)
        if emit is not None:
            emit({"type": "retrieve", "query": query, "hits": hits})
        if not hits:
            return "知识库中未检索到相关内容。"
        return "\n\n".join(f"[相关度 {h['score']}] {h['text']}" for h in hits)

    return knowledge_search
