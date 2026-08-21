"""会话存储：LangGraph 检查点（多轮/中断恢复）+ 每会话长期记忆库。"""
from __future__ import annotations

import uuid

from langgraph.checkpoint.memory import MemorySaver

from app.memory.vector_store import VectorStore


class SessionStore:
    def __init__(self):
        self.checkpointer = MemorySaver()
        self._long_memories: dict[str, VectorStore] = {}

    def create(self) -> str:
        """新建会话并返回 session_id。"""
        return uuid.uuid4().hex

    def long_memory(self, session_id: str, embeddings) -> VectorStore:
        """获取（必要时创建）某会话的长期记忆库。"""
        store = self._long_memories.get(session_id)
        if store is None:
            store = VectorStore(embeddings, name=f"memory:{session_id}")
            self._long_memories[session_id] = store
        return store
