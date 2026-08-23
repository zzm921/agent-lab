"""多后端融合存储：跨后端多路召回（Qdrant + Elasticsearch 双路查询后融合去重）。

实现 StoreBackend 接口，供 advanced 方案「不是只走一个后端」的多路召回：
- 入库：同时写入所有后端（同一语料、各自独立集合/索引）；
- 检索：`search` / `hybrid_search` 并行查询所有后端，按文本去重、取各后端最高分融合；
- 容错：单个后端异常（如 ES 未启动）自动跳过该路，不阻塞整条检索（对齐项目回退哲学）。
"""
from __future__ import annotations

import logging

from typing import Any

from app.memory.stores.base import StoreBackend

logger = logging.getLogger(__name__)


class MultiBackendStore(StoreBackend):
    """把多个 StoreBackend 聚合为一个逻辑后端，检索结果跨后端融合。"""

    name: str = "multi"

    def __init__(self, backends: list[StoreBackend]):
        if not backends:
            raise ValueError("MultiBackendStore 需要至少一个后端")
        self.backends = list(backends)

    @property
    def collection(self) -> str:
        return "+".join(b.collection for b in self.backends)

    def add(self, text: str, metadata: dict | None = None) -> None:
        for backend in self.backends:
            backend.add(text, metadata)

    def _query_all(self, method: str, query: str, top_k: int) -> list[list[dict[str, Any]]]:
        """并行查询所有后端；单个后端异常只告警跳过，保证一路失败不拖垮整体。"""
        results: list[list[dict[str, Any]]] = []
        for backend in self.backends:
            try:
                results.append(getattr(backend, method)(query, max(top_k * 2, 8)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("后端 %s 检索失败（%s），跳过该路", backend.name, exc)
        return results

    @staticmethod
    def _fuse(lists: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
        """按文本去重，保留各后端最高分，降序截取 top_k。"""
        best: dict[str, dict[str, Any]] = {}
        for hits in lists:
            for hit in hits:
                text = hit.get("text", "")
                if not text:
                    continue
                if text not in best or hit.get("score", 0.0) > best[text].get("score", 0.0):
                    best[text] = hit
        ranked = sorted(best.values(), key=lambda h: h.get("score", 0.0), reverse=True)
        return ranked[:top_k]

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self._fuse(self._query_all("search", query, top_k), top_k)

    def hybrid_search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self._fuse(self._query_all("hybrid_search", query, top_k), top_k)

    def all_texts(self) -> list[str]:
        seen: list[str] = []
        for backend in self.backends:
            for text in backend.all_texts():
                if text not in seen:
                    seen.append(text)
        return seen

    def clear(self) -> None:
        for backend in self.backends:
            backend.clear()

    def __len__(self) -> int:
        # 各后端入库同一语料，条数应一致；取最大避免某后端未写全时误判为空
        return max((len(b) for b in self.backends), default=0)
