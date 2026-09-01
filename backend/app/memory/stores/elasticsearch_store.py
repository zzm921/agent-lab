"""Elasticsearch 存储后端：稠密向量 kNN 相似度检索 + 稠密&BM25 混合检索（RRF 融合）。

实现 StoreBackend 接口，与 Qdrant / 内存实现互为可替换后端。启动时探测服务端主版本自动选路：
- 现代路径（ES 8.x+）：`dense_vector` 字段存向量，在线 kNN 检索；混合检索用
  kNN 稠密 + `text` BM25 双路召回，经 ES 原生 RRF（`rank.rrf`，8.8+）融合；
- 兼容路径（ES < 8.x，如 6.8）：旧服务器没有 dense_vector / kNN / RRF，且多维
  数值字段会被 Lucene 按值排序（破坏向量维度对齐），无法做真实向量打分，因此
  退化为 `text` BM25 关键词检索——与 Qdrant 的稠密路在多路召回中互补。

未配置 ES 或连接失败时由 manager 回退内存存储（与 Qdrant 回退策略一致）。
"""
from __future__ import annotations

import logging
import uuid

from typing import Any

import httpx

from app.core.errors import ConfigError
from app.memory.stores.base import StoreBackend

logger = logging.getLogger(__name__)


class ElasticsearchStore(StoreBackend):
    """基于 Elasticsearch 的检索后端（现代 kNN/RRF 或旧版 BM25 兼容）。"""

    name: str = "elasticsearch"

    def __init__(
        self,
        embeddings,
        index: str,
        url: str = "",
        api_key: str = "",
        username: str = "",
        password: str = "",
        dim: int = 1024,
        hybrid: bool = False,
        client=None,
        http_client=None,
    ):
        self.embeddings = embeddings
        self._index = index
        self.dim = dim
        self.hybrid = hybrid
        if client is not None:  # 测试注入 fake client（不联网）：现代路径
            self.client = client
            self._legacy = False
        else:
            if not url:
                raise ConfigError("未配置 ES_URL，无法连接 Elasticsearch")
            self._legacy = self._detect_legacy(url, api_key, username, password)
            if self._legacy:
                # 旧版 ES：走原生 REST（httpx），兼容 6.x 协议与旧式 _doc 类型映射
                auth = (username, password) if username and password else None
                headers = {"Authorization": f"ApiKey {api_key}"} if api_key else {}
                self._http = http_client or httpx.Client(
                    base_url=url.rstrip("/"), auth=auth, headers=headers, timeout=15.0
                )
            else:
                try:
                    from elasticsearch import Elasticsearch
                except ImportError as exc:  # pragma: no cover - 依赖缺失提示
                    raise ConfigError("未安装 elasticsearch 客户端，请 `pip install elasticsearch`") from exc
                self.client = Elasticsearch(
                    hosts=[url],
                    api_key=api_key or None,
                    basic_auth=(username, password) if username and password else None,
                    request_timeout=10,
                )
        self._ensure_index()

    @property
    def collection(self) -> str:
        return self._index

    # ---- 版本探测 / 建索引 ----

    @staticmethod
    def _detect_legacy(url: str, api_key: str, username: str, password: str) -> bool:
        """探测 ES 服务端主版本：<8.x（如 6.8）返回 True 走兼容路径。

        连接/鉴权失败抛 ConfigError（由上层回退内存存储）。
        """
        auth = (username, password) if username and password else None
        headers = {"Authorization": f"ApiKey {api_key}"} if api_key else {}
        try:
            resp = httpx.get(url.rstrip("/") + "/", auth=auth, headers=headers, timeout=8)
            resp.raise_for_status()
            number = (resp.json() or {}).get("version", {}).get("number", "")
            major = int((number.split(".") or ["0"])[0] or 0)
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"ES 版本探测失败（无法连接或鉴权）：{exc}") from exc
        if major < 8:
            logger.warning(
                "ES %s 为旧版（<8.x），无 dense_vector/kNN/RRF，advanced 的 ES 路退化为 BM25 关键词召回",
                number,
            )
        return major < 8

    def _ensure_index(self) -> None:
        """索引不存在则创建；结构按现代/兼容路径分别处理。"""
        try:
            if self._legacy:
                self._ensure_legacy_index()
            elif self.client.indices.exists(index=self._index):
                return
            else:
                self.client.indices.create(
                    index=self._index,
                    mappings={
                        "properties": {
                            "text": {"type": "text"},
                            "dense_vector": {
                                "type": "dense_vector",
                                "dims": self.dim,
                                "index": True,
                                "similarity": "cosine",
                            },
                        }
                    },
                )
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"ES 连接/建索引失败：{exc}") from exc

    def _ensure_legacy_index(self) -> None:
        """旧版 ES 建索引：旧式 _doc 类型映射 + text 字段（优先 IK 中文分词）。

        text 用 `ik_max_word`/`ik_smart` 让中文 BM25 有意义；若该集群未装 IK
        分词插件则回退默认 standard 分析器。
        """
        resp = self._http.get(f"/{self._index}")
        if resp.status_code == 200:
            return  # 已存在
        if resp.status_code != 404:
            raise ConfigError(f"ES 索引检查失败：{resp.status_code} {resp.text[:200]}")
        for text_field in (
            {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
            {"type": "text"},
        ):
            resp = self._http.put(
                f"/{self._index}",
                json={"mappings": {"_doc": {"properties": {"text": text_field}}}},
            )
            if resp.status_code in (200, 201):
                return
        raise ConfigError(f"ES 建索引失败：{resp.status_code} {resp.text[:200]}")

    # ---- 数据写入 ----

    def add(self, text: str, metadata: dict | None = None) -> None:
        """写入一条文档：text（BM25）+ dense_vector（现代 kNN 路）+ 元数据。"""
        if self._legacy:
            # 旧版无向量字段：多维数值字段会被按值排序、无法对齐，故只存 text+元数据
            self._http.put(
                f"/{self._index}/_doc/{uuid.uuid4()}",
                json={"text": text, "metadata": metadata or {}},
            )
            return
        self.client.index(
            index=self._index,
            id=str(uuid.uuid4()),
            document={
                "text": text,
                "metadata": metadata or {},
                "dense_vector": self.embeddings.embed_query(text),
            },
        )

    def _map_hits(self, resp) -> list[dict[str, Any]]:
        hits = (resp or {}).get("hits", {}).get("hits", [])
        out = []
        for hit in hits:
            source = hit.get("_source") or {}
            out.append(
                {
                    "text": source.get("text", ""),
                    "score": round(float(hit.get("_score", 0.0)), 4),
                    "metadata": source.get("metadata") or {},
                }
            )
        return out

    # ---- 检索 ----

    def search(self, query: str, top_k: int = 3, volume_filter: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """稠密向量检索（现代 kNN）；旧版退化为 BM25 关键词检索。volume_filter 忽略。"""
        if self._legacy:
            return self._bm25_search(query, top_k)
        qv = self.embeddings.embed_query(query)
        resp = self.client.search(
            index=self._index,
            knn={
                "field": "dense_vector",
                "query_vector": qv,
                "k": top_k,
                "num_candidates": max(top_k * 4, 16),
            },
            size=top_k,
        )
        return self._map_hits(resp)

    def hybrid_search(self, query: str, top_k: int = 3, volume_filter: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """混合检索：现代 kNN+BM25 RRF 融合；旧版仅 BM25（稠密语义路由由 Qdrant 承担）。volume_filter 忽略。"""
        if self._legacy:
            return self._bm25_search(query, top_k)
        if not self.hybrid:
            return self.search(query, top_k)
        qv = self.embeddings.embed_query(query)
        body: dict[str, Any] = {
            "knn": {
                "field": "dense_vector",
                "query_vector": qv,
                "k": max(top_k * 4, 16),
                "num_candidates": max(top_k * 8, 32),
            },
            "size": top_k,
        }
        if query.strip():
            body["query"] = {"match": {"text": query}}
            body["rank"] = {"rrf": {}}  # ES 8.8+ 原生倒数排名融合
        resp = self.client.search(index=self._index, **body)
        return self._map_hits(resp)

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """旧版 ES 的检索实现：text 字段 BM25 关键词召回（空查询退化为 match_all）。"""
        body: dict[str, Any] = {"size": top_k}
        if query.strip():
            body["query"] = {"match": {"text": query}}
        resp = self._http.post(f"/{self._index}/_search", json=body)
        return self._map_hits(resp.json())

    # ---- 数据生命周期 ----

    def all_texts(self) -> list[str]:
        """读取全部已入库文本（语料指纹比对用）。"""
        if self._legacy:
            resp = self._http.post(
                f"/{self._index}/_search",
                json={"query": {"match_all": {}}, "_source": ["text"], "size": 10000},
            )
            return [
                h.get("_source", {}).get("text", "")
                for h in (resp.json() or {}).get("hits", {}).get("hits", [])
            ]
        resp = self.client.search(
            index=self._index,
            query={"match_all": {}},
            size=10000,
            source=["text"],
        )
        return [
            h.get("_source", {}).get("text", "")
            for h in (resp or {}).get("hits", {}).get("hits", [])
        ]

    def clear(self) -> None:
        """清空索引全部数据（保留索引结构，便于语料变更后重建）。"""
        if self._legacy:
            # 旧版 delete_by_query 的 refresh 是 URL 参数而非请求体字段
            self._http.post(
                f"/{self._index}/_delete_by_query",
                params={"refresh": "true"},
                json={"query": {"match_all": {}}},
            )
            return
        self.client.delete_by_query(
            index=self._index,
            query={"match_all": {}},
            refresh=True,
        )

    def delete_source(self, source: str) -> int:
        """按 metadata.source 过滤删除（增量更新：文档变更先删旧块）。"""
        if self._legacy:
            resp = self._http.post(
                f"/{self._index}/_delete_by_query",
                params={"refresh": "true"},
                json={"query": {"match": {"metadata.source": source}}},
            )
            return int((resp.json() or {}).get("deleted", 0))
        resp = self.client.delete_by_query(
            index=self._index,
            query={"match": {"metadata.source": source}},
            refresh=True,
        )
        return int((resp or {}).get("deleted", 0))

    def __len__(self) -> int:
        if self._legacy:
            resp = self._http.get(f"/{self._index}/_count")
            return int((resp.json() or {}).get("count", 0))
        resp = self.client.count(index=self._index)
        return int((resp or {}).get("count", 0))
