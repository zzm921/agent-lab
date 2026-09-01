"""内存存储后端：未配置 Qdrant 时的回退实现（复用 VectorStore 余弦检索逻辑）。

实现 StoreBackend 接口，使 RAG 方案层在无向量库时也能正常运行（naive 纯稠密语义检索）。
"""
from __future__ import annotations

from typing import Any

from app.memory.stores.base import StoreBackend
from app.memory.vector_store import VectorStore


class MemoryStore(StoreBackend):
    """基于内存 VectorStore 的 StoreBackend 实现（离线/测试回退用）。"""

    name: str = "memory"

    def __init__(self, embeddings, collection: str = "knowledge"):
        self.embeddings = embeddings
        self._collection = collection
        self._store = VectorStore(embeddings, name=collection)

    @property
    def collection(self) -> str:
        return self._collection

    def add(self, text: str, metadata: dict | None = None) -> None:
        self._store.add(text, metadata)

    def search(self, query: str, top_k: int = 3, volume_filter: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        # 测试桩：volume_filter 忽略（无过滤能力，安全 no-op）
        return self._store.search(query, top_k)

    def all_texts(self) -> list[str]:
        return list(self._store.texts)

    def clear(self) -> None:
        self._store.texts = []
        self._store.metadatas = []
        self._store.vectors = []

    def delete_source(self, source: str) -> int:
        """按 metadata.source 删除匹配块（增量更新：文档变更先删旧块）。"""
        keep = [
            i
            for i, meta in enumerate(self._store.metadatas)
            if (meta or {}).get("source") != source
        ]
        removed = len(self._store.texts) - len(keep)
        if removed == 0:
            return 0
        self._store.texts = [self._store.texts[i] for i in keep]
        self._store.metadatas = [self._store.metadatas[i] for i in keep]
        self._store.vectors = [self._store.vectors[i] for i in keep]
        return removed

    def __len__(self) -> int:
        return len(self._store)
