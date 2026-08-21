"""长期记忆工具：写入/召回会话级向量记忆。"""
from langchain_core.tools import tool


def make_memory_tools(long_memory, top_k: int = 3, emit=None):
    """构建长期记忆读写工具；提供 emit 时可推送 memory_write/memory_read 事件。"""

    @tool
    def memory_write(fact: str) -> str:
        """把一条重要事实写入长期记忆，供后续对话回忆。"""
        long_memory.add(fact, {"kind": "fact"})
        if emit is not None:
            emit({"type": "memory_write", "content": fact})
        return "已记住：" + fact

    @tool
    def memory_recall(query: str) -> str:
        """从长期记忆中检索与查询相关的事实。"""
        hits = long_memory.search(query, top_k)
        if emit is not None:
            emit({"type": "memory_read", "query": query, "hits": hits})
        if not hits:
            return "长期记忆中没有相关记录。"
        return "\n".join(f"- {h['text']}" for h in hits)

    return [memory_write, memory_recall]
