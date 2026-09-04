"""长期记忆存储：持久化 + 分类 + 元数据 + 治理（语义去重 / 上限 LRU / TTL）。

数据模型（每条记录 JSONL 一行）：
{id, kind, text, vector, importance, created_at, last_access_at,
 access_count, source_session, ttl}

向量随记录落盘，启动 load 直接重建内存索引（不重算 embedding）——
单机「温/冷合一」形态：检索毫秒级、不调 LLM。
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.memory.vector_store import VectorStore

MEMORY_KINDS = ("fact", "preference", "episodic", "procedural")


class LongMemoryStore:
    """会话级/全局级长期记忆：写入（去重）、召回（阈值+kind 过滤）、管理与遗忘。"""

    def __init__(
        self,
        session_id: str,
        embeddings,
        path: str | None = None,
        *,
        top_k: int = 3,
        threshold: float = 0.0,
        dedup_threshold: float = 0.92,
        max_per_namespace: int = 500,
        ttl_days: int = 0,
    ):
        self.session_id = session_id
        self.embeddings = embeddings
        self.path = Path(path) if path else None
        self.top_k = top_k
        self.threshold = threshold
        self.dedup_threshold = dedup_threshold
        self.max_per_namespace = max_per_namespace
        self.ttl_days = ttl_days
        self._store = VectorStore(embeddings, name=f"memory:{session_id}")
        self._index: dict[str, int] = {}  # record_id -> 内存索引
        self._load()

    # ---------- 内部工具 ----------

    def _rebuild_index(self) -> None:
        self._index = {
            meta.get("id"): i
            for i, meta in enumerate(self._store.metadatas)
            if meta.get("id")
        }

    def _load(self) -> None:
        if self.path is not None:
            self._store.load(str(self.path))
        self._rebuild_index()

    def _persist(self) -> None:
        if self.path is not None:
            self._store.save(str(self.path))

    def _drop(self, index: int) -> None:
        self._store.delete(index)
        self._rebuild_index()

    def _enforce_limits(self) -> None:
        """写入后治理：TTL 清理 + 每命名空间上限 LRU 淘汰（按 last_access_at 升序）。"""
        if self.ttl_days > 0:
            cutoff = time.time() - self.ttl_days * 86400
            for i in sorted(
                (i for i, m in enumerate(self._store.metadatas) if (m.get("created_at") or 0) < cutoff),
                reverse=True,
            ):
                self._drop(i)
        while self.max_per_namespace > 0 and len(self._store) > self.max_per_namespace:
            idx = min(
                range(len(self._store.metadatas)),
                key=lambda i: self._store.metadatas[i].get("last_access_at", 0),
            )
            self._drop(idx)

    def _find_dup(self, text: str) -> tuple[int, dict] | None:
        """语义去重：与已有记录相似度 ≥ dedup_threshold 时返回其 (索引, 元数据)。"""
        if not self._store.texts:
            return None
        hits = self._store.search(text, top_k=1, threshold=self.dedup_threshold)
        if not hits:
            return None
        rec_id = (hits[0].get("metadata") or {}).get("id")
        idx = self._index.get(rec_id) if rec_id else None
        if idx is None:
            return None
        return idx, self._store.metadatas[idx]

    # ---------- 写入 ----------

    def add(
        self,
        text: str,
        kind: str = "fact",
        importance: float = 0.5,
        source_session: str | None = None,
        ttl: int | None = None,
    ) -> dict[str, Any]:
        """写入一条记忆：语义去重（相似则更新而非追加，纠偏），写后落盘 + 治理。"""
        now = time.time()
        dup = self._find_dup(text)
        if dup is not None:
            idx, meta = dup
            meta = dict(meta)
            self._store.update(
                idx,
                text=text,
                metadata={
                    **meta,
                    "kind": kind,
                    "importance": importance,
                    "updated_at": now,
                },
            )
            self._persist()
            return {"id": meta.get("id"), "action": "update"}
        rec_id = uuid.uuid4().hex
        meta = {
            "id": rec_id,
            "kind": kind,
            "importance": importance,
            "created_at": now,
            "last_access_at": now,
            "access_count": 0,
            "source_session": source_session or self.session_id,
            "ttl": ttl,
        }
        self._store.add(text, meta)
        self._index[rec_id] = len(self._store) - 1
        self._enforce_limits()
        self._persist()
        return {"id": rec_id, "action": "add"}

    # ---------- 召回 ----------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """按语义召回：阈值过滤 + kind 过滤；命中即更新访问统计（供 LRU/老化）。"""
        top_k = top_k or self.top_k
        threshold = self.threshold if threshold is None else threshold
        hits = self._store.search(query, top_k=max(top_k * 2, 8), threshold=threshold)
        if kind:
            hits = [h for h in hits if (h.get("metadata") or {}).get("kind") == kind]
        now = time.time()
        changed = False
        for h in hits:
            rec_id = (h.get("metadata") or {}).get("id")
            idx = self._index.get(rec_id)
            if idx is None:
                continue
            meta = dict(self._store.metadatas[idx])
            meta["last_access_at"] = now
            meta["access_count"] = meta.get("access_count", 0) + 1
            self._store.metadatas[idx] = meta
            changed = True
        if changed:
            self._persist()
        return hits[:top_k]

    def constant_memories(self, top_k: int, min_importance: float = 0.7) -> list[tuple[str, dict]]:
        """常驻记忆：按 importance 降序取 top-k（注入 system 用，不依赖查询）。"""
        cands = [
            (text, meta)
            for text, meta in zip(self._store.texts, self._store.metadatas)
            if meta.get("importance", 0) >= min_importance
        ]
        cands.sort(key=lambda t: t[1].get("importance", 0), reverse=True)
        return cands[:top_k]

    # ---------- 管理 ----------

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for text, meta in zip(self._store.texts, self._store.metadatas):
            if kind and meta.get("kind") != kind:
                continue
            items.append({"id": meta.get("id"), "text": text, **meta})
        return items

    def delete(self, rec_id: str) -> bool:
        idx = self._index.get(rec_id)
        if idx is None:
            return False
        self._drop(idx)
        self._persist()
        return True

    def __len__(self) -> int:
        return len(self._store)

    def to_jsonl_snapshot(self) -> str:
        """返回当前全部记录的可读 JSON（管理面板展示用）。"""
        return json.dumps(self.list(), ensure_ascii=False, indent=2)
