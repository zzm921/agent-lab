"""会话存储：LangGraph 检查点（多轮/中断恢复）+ 每会话/按客户端的常驻记忆库。

常驻（全局）记忆按客户端隔离：每个试用者（按设备指纹 X-Client-Id 或 IP 标识）
各有一份常驻库，互不可见，防止「记忆串台」；会话记忆仍按随机 session_id 隔离。
"""
from __future__ import annotations

import re
import uuid

from langgraph.checkpoint.memory import MemorySaver

from app.memory.long_memory import LongMemoryStore

_GLOBAL_ID = "_global"


def _safe_key(key: str) -> str:
    """把 client_key（cid:xxx / ip:xxx）转成文件安全名，避免冒号等非法字符进路径。"""
    return re.sub(r"[^0-9A-Za-z_\-]", "_", key)[:64]


class SessionStore:
    """单机形态的会话与记忆存储。

    memory_dir 为空时记忆库仅存内存（测试/离线回退，不落盘）；
    配置了 memory_dir 时，每会话一个 {memory_dir}/{session_id}.jsonl，
    每个客户端的常驻记忆为 {memory_dir}/_global_{client_key}.jsonl。
    """

    def __init__(self, memory_dir: str | None = None, **memory_kwargs):
        self.checkpointer = MemorySaver()
        self._memory_dir = memory_dir
        self._memory_kwargs = memory_kwargs
        self._long_memories: dict[str, LongMemoryStore] = {}
        self._constants: dict[str, LongMemoryStore] = {}

    def create(self) -> str:
        """新建会话并返回 session_id。"""
        return uuid.uuid4().hex

    def _store_path(self, session_id: str) -> str | None:
        if not self._memory_dir:
            return None
        return f"{self._memory_dir}/{session_id}.jsonl"

    def long_memory(self, session_id: str, embeddings) -> LongMemoryStore:
        """获取（必要时创建）某会话的长期记忆库。"""
        store = self._long_memories.get(session_id)
        if store is None:
            store = LongMemoryStore(
                session_id,
                embeddings,
                self._store_path(session_id),
                **self._memory_kwargs,
            )
            self._long_memories[session_id] = store
        return store

    def constant_memory(self, embeddings, client_key: str = "default") -> LongMemoryStore:
        """获取（必要时创建）某客户端的常驻记忆库：按 client_key 各一份，互不可见。

        client_key 由请求层判定（设备指纹 X-Client-Id 优先、IP 兜底）；
        未传时回退 "default"（兼容测试/无请求场景）。
        """
        key = client_key or "default"
        store = self._constants.get(key)
        if store is None:
            store = LongMemoryStore(
                f"{_GLOBAL_ID}:{key}",
                embeddings,
                self._store_path(f"{_GLOBAL_ID}_{_safe_key(key)}"),
                **self._memory_kwargs,
            )
            self._constants[key] = store
        return store
