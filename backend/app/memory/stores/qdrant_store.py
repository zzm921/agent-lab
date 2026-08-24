"""Qdrant 通用存储后端：稠密 + 可选稀疏（混合检索 RRF 融合）。

与具体 RAG 方案解耦：只负责「入库 / 稠密检索 / 混合检索 / 集合管理」，
方案层（app.rag）据此构建各自的独立集合。ES 后端后续实现同一 StoreBackend 接口即可替换。
"""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.errors import ConfigError
from app.memory.stores.base import StoreBackend


def _sparse_vec(sparse: dict | None):
    """把 {indices, values} 转换为 Qdrant SparseVector。"""
    if not sparse or not sparse.get("indices"):
        return None
    return models.SparseVector(indices=sparse["indices"], values=sparse["values"])


class QdrantStore(StoreBackend):
    """基于 Qdrant 的检索后端（可连云端 / Docker / 内嵌 :memory:）。"""

    name: str = "qdrant"

    def __init__(
        self,
        embeddings,
        collection: str,
        url: str = "",
        api_key: str = "",
        dim: int = 1024,
        sparse: bool = False,
        client: QdrantClient | None = None,
    ):
        self.embeddings = embeddings
        self._collection = collection
        self.dim = dim
        self.sparse = sparse
        if client is not None:  # 测试注入内嵌 client（QdrantClient(":memory:")）
            self.client = client
        else:
            if not url:
                raise ConfigError("未配置 QDRANT_URL，无法连接 Qdrant")
            self.client = QdrantClient(url=url, api_key=api_key or None)
        self._ensure_collection()

    @property
    def collection(self) -> str:
        return self._collection

    def _ensure_collection(self) -> None:
        """集合不存在则创建：命名稠密向量 dense（cosine）+ 可选命名稀疏向量 sparse。

        若集合已存在但结构不匹配（旧版未命名向量/维度不符/缺稀疏配置），删除重建以自愈。
        """
        try:
            if not self.client.collection_exists(self._collection):
                self._create_collection()
                return
            info = self.client.get_collection(self._collection)
            if self._needs_recreate(info):
                self.client.delete_collection(self._collection)
                self._create_collection()
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"Qdrant 连接/建集合失败：{exc}") from exc

    def _create_collection(self) -> None:
        self.client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()} if self.sparse else None,
        )

    def _needs_recreate(self, info) -> bool:
        """校验集合向量结构是否与当前约定一致（命名 dense + 可选命名 sparse + 维度）。"""
        vectors = info.config.params.vectors
        dense_ok = (
            isinstance(vectors, dict)
            and "dense" in vectors
            and vectors["dense"].size == self.dim
        )
        if not dense_ok:
            return True
        if not self.sparse:
            return False
        sparse = info.config.params.sparse_vectors
        return not (isinstance(sparse, dict) and "sparse" in sparse)

    def add(self, text: str, metadata: dict | None = None) -> None:
        """写入一条文本：稠密向量（+ 稀疏向量，若开启混合检索）。

        payload 结构与 ES 后端一致：text 与 metadata 分离（metadata 嵌套存放），
        保证跨后端（Qdrant + Elasticsearch）多路召回时元数据语义一致。
        """
        payload = {"text": text, "metadata": metadata or {}}
        vector: dict = {"dense": self.embeddings.embed_query(text)}
        if self.sparse:
            sparse = self.embeddings.embed_sparse_query(text)
            sv = _sparse_vec(sparse)
            if sv is not None:
                vector["sparse"] = sv
        self.client.upsert(
            collection_name=self._collection,
            points=[models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)],
        )

    def _map_hits(self, hits) -> list[dict[str, Any]]:
        return [
            {
                "text": hit.payload.get("text", ""),
                "score": round(float(hit.score), 4),
                "metadata": hit.payload.get("metadata") or {},
            }
            for hit in hits
        ]

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """稠密向量检索。"""
        qv = self.embeddings.embed_query(query)
        resp = self.client.query_points(
            collection_name=self._collection,
            query=qv,
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        return self._map_hits(resp.points)

    def all_texts(self) -> list[str]:
        """分页滚动读取全部已入库文本（语料指纹比对用）。"""
        texts: list[str] = []
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self._collection,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            texts.extend(p.payload.get("text", "") for p in points)
            if next_offset is None:
                break
            offset = next_offset
        return texts

    def clear(self) -> None:
        """清空集合全部数据（保留集合结构，便于语料变更后重建）。"""
        self.client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=models.Filter()),
        )

    def hybrid_search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """混合检索：稠密 + 稀疏双路召回，Qdrant 内置 RRF 融合。"""
        if not self.sparse:
            return self.search(query, top_k)
        dense = self.embeddings.embed_query(query)
        sparse = _sparse_vec(self.embeddings.embed_sparse_query(query))
        # 各路放宽召回（top_k×4）避免早期截断，再经 RRF 融合取 top_k
        prefetch = [models.Prefetch(query=dense, using="dense", limit=max(top_k * 4, 16))]
        if sparse is not None:
            prefetch.append(
                models.Prefetch(query=sparse, using="sparse", limit=max(top_k * 4, 16))
            )
        resp = self.client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return self._map_hits(resp.points)

    def __len__(self) -> int:
        return self.client.count(self._collection).count
