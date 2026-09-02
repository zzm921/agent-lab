"""RAG 方案管理器：按配置构建各方案（每方案一个独立 Qdrant 集合），统一入库与选择。

未配置 Qdrant 或连接失败时回退到内存存储（MemoryStore），保证离线/测试可用。
当前落地 naive（固定切块+纯稠密）、advanced（语义分块+混合检索+重写重排）、
modular（前置语义分类+按复杂度路由）方案；graph / agentic 后续在同一框架上扩展。
"""
from __future__ import annotations

import logging

from typing import Any

from app.config import Settings
from app.core.errors import ConfigError
from app.rag.cache import CachedEmbeddings
from app.rag.schemes.agentic import AgenticRagScheme
from app.memory.stores.base import StoreBackend
from app.memory.stores.elasticsearch_store import ElasticsearchStore
from app.memory.stores.memory_store import MemoryStore
from app.memory.stores.multi_backend_store import MultiBackendStore
from app.memory.stores.qdrant_store import QdrantStore
from app.rag.base import RagScheme
from app.rag.schemes.advanced import AdvancedRagScheme
from app.rag.schemes.modular import ModularRagScheme
from app.rag.schemes.naive import NaiveRagScheme

logger = logging.getLogger(__name__)

# 方案 id → 方案类（每方案一个独立 Qdrant 集合；后续扩展在此登记）
_SCHEME_REGISTRY: dict[str, type[RagScheme]] = {
    "naive": NaiveRagScheme,
    "advanced": AdvancedRagScheme,
    "modular": ModularRagScheme,
    "agentic": AgenticRagScheme,
}


class RagManager:
    """持有全部已注册 RAG 方案；每方案一个独立 Qdrant 集合（同一语料不同库）。"""

    def __init__(
        self,
        settings: Settings,
        embeddings,
        top_k: int = 3,
        stores: dict[str, StoreBackend] | None = None,
        scheme_ids: list[str] | None = None,
    ):
        self.settings = settings
        self.embeddings = embeddings
        # L2 嵌入缓存：包装后所有方案共享同一记忆层（query 文本 → 向量），避免重复 embedding 调用。
        # 语义等价（纯记忆），仅当开关开启时生效；语料重建不影响 query 向量有效性。
        if settings.rag_cache_enabled:
            self.embeddings = CachedEmbeddings(
                embeddings, max_entries=settings.rag_cache_max_entries * 8
            )
        self.top_k = top_k
        self.schemes: dict[str, RagScheme] = {}
        # 指定方案时只构建这些方案（离线建库按方案独立脚本用）；缺省用 settings.rag_schemes
        ids = scheme_ids if scheme_ids is not None else settings.rag_schemes
        for scheme_id in ids:
            if scheme_id not in _SCHEME_REGISTRY:
                raise ConfigError(f"未知 RAG 方案：{scheme_id}（支持 {list(_SCHEME_REGISTRY)}）")
            store = stores.get(scheme_id) if stores else self._build_store(scheme_id)
            scheme_cls = _SCHEME_REGISTRY[scheme_id]
            # advanced/modular 含 Query 重写/重排等可选配置项；naive 为最简构造。
            # 各方案内部的 LLM 阶段按命名场景从全局 LLMService 懒取，无需在此注入模型。
            if issubclass(scheme_cls, AdvancedRagScheme):
                kwargs: dict[str, Any] = dict(
                    rewrite_variants=self.settings.rag_rewrite_variants,
                    rerank_model=self.settings.rag_rerank_model,
                )
                if scheme_cls is ModularRagScheme:
                    kwargs["max_hops"] = self.settings.rag_max_hops
                    kwargs["fast_path_conf"] = self.settings.rag_fast_path_conf
                    kwargs["low_conf_threshold"] = self.settings.rag_low_conf_threshold
                    kwargs["cache_enabled"] = self.settings.rag_cache_enabled
                    kwargs["cache_max_entries"] = self.settings.rag_cache_max_entries
                    kwargs["cache_ttl_s"] = self.settings.rag_cache_ttl_s
                if scheme_cls is AgenticRagScheme:
                    kwargs["max_hops"] = self.settings.rag_max_hops
                    kwargs["max_steps"] = self.settings.rag_agent_max_steps
                    kwargs["correction_rounds"] = self.settings.rag_agent_correction_rounds
                    kwargs["timeout_s"] = self.settings.rag_agent_timeout_s
                    kwargs["token_budget"] = self.settings.rag_agent_token_budget
                    kwargs["call_cap"] = self.settings.rag_agent_tool_call_cap
                    kwargs["parallel"] = self.settings.rag_agent_parallel
                self.schemes[scheme_id] = scheme_cls(embeddings, store, top_k, **kwargs)
            else:
                self.schemes[scheme_id] = scheme_cls(embeddings, store, top_k)

    def _build_store(self, scheme_id: str) -> StoreBackend:
        """构建方案后端：主后端按 rag_store_backend 选择，multi_backend 方案叠加次后端。

        - 主后端：qdrant（默认）| elasticsearch | memory，未配置/连接失败回退内存；
        - 次后端（multi_backend 方案，如 advanced）：主后端为 qdrant/memory 时叠加 ES，
          主后端为 elasticsearch 时叠加 Qdrant——实现「跨后端多路召回」，不是只走一个；
        - 每个方案一个独立库/索引/集合（{prefix}_{scheme_id}）。
        """
        primary = self._build_primary_store(scheme_id)
        if not _SCHEME_REGISTRY[scheme_id].multi_backend:
            return primary
        secondary = self._build_secondary_store(scheme_id)
        if secondary is None or secondary is primary:
            return primary
        return MultiBackendStore([primary, secondary])

    def _build_primary_store(self, scheme_id: str) -> StoreBackend:
        """按 rag_store_backend 构建主后端（含未配置/连接失败回退内存）。"""
        hybrid = _SCHEME_REGISTRY[scheme_id].hybrid
        backend = self.settings.rag_store_backend
        if backend == "elasticsearch":
            return self._build_es(scheme_id, hybrid)
        if backend == "memory":
            return MemoryStore(
                self.embeddings,
                collection=f"{self.settings.es_index_prefix}_{scheme_id}",
            )
        return self._build_qdrant(scheme_id, hybrid)  # 默认 qdrant

    def _build_secondary_store(self, scheme_id: str) -> StoreBackend | None:
        """multi_backend 方案的次后端：与主后端互补（主 Qdrant→次 ES，主 ES→次 Qdrant）。

        未配置对应后端时返回 None（不叠加内存路），保持离线回退为单个内存后端。
        """
        hybrid = _SCHEME_REGISTRY[scheme_id].hybrid
        if self.settings.rag_store_backend == "elasticsearch":
            if not self.settings.qdrant_url:
                return None
            return self._build_qdrant(scheme_id, hybrid)
        if not self.settings.es_url:
            return None
        return self._build_es(scheme_id, hybrid)

    def _build_qdrant(self, scheme_id: str, hybrid: bool) -> StoreBackend:
        """构建 Qdrant 后端；未配置或连接失败回退内存。"""
        collection = f"{self.settings.qdrant_collection_prefix}_{scheme_id}"
        if not self.settings.qdrant_url:
            return MemoryStore(self.embeddings, collection=collection)
        try:
            return QdrantStore(
                self.embeddings,
                collection=collection,
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                dim=self.settings.qdrant_embedding_dim,
                sparse=hybrid,
            )
        except ConfigError as exc:
            logger.warning("Qdrant 不可用（%s），方案 %s 回退内存存储", exc, scheme_id)
            return MemoryStore(self.embeddings, collection=collection)

    def _build_es(self, scheme_id: str, hybrid: bool) -> StoreBackend:
        """构建 Elasticsearch 后端；未配置或连接失败回退内存。"""
        collection = f"{self.settings.es_index_prefix}_{scheme_id}"
        if not self.settings.es_url:
            return MemoryStore(self.embeddings, collection=collection)
        try:
            return ElasticsearchStore(
                self.embeddings,
                index=collection,
                url=self.settings.es_url,
                api_key=self.settings.es_api_key,
                username=self.settings.es_username,
                password=self.settings.es_password,
                dim=self.settings.es_embedding_dim,
                hybrid=hybrid,
            )
        except ConfigError as exc:
            logger.warning("ES 不可用（%s），方案 %s 回退内存存储", exc, scheme_id)
            return MemoryStore(self.embeddings, collection=collection)

    def ingest_all(self, texts: list[str]) -> None:
        """把同一份语料写入所有方案的独立集合（各方案幂等跳过）。"""
        for scheme in self.schemes.values():
            scheme.ingest(texts)

    def get(self, scheme_id: str) -> RagScheme | None:
        return self.schemes.get(scheme_id)

    def resolve(self, scheme_id: str | None) -> RagScheme:
        """解析方案 id；缺省/未知回退默认方案。

        注意不能用 `schemes.get(sid) or default`：RagScheme 定义了 __len__，
        空集合时 bool(方案) 为 False 会误回退，故必须用显式 None 判断。
        """
        sid = scheme_id or self.settings.rag_default_scheme
        scheme = self.schemes.get(sid)
        if scheme is None:
            scheme = self.schemes.get(self.settings.rag_default_scheme)
        return scheme

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "collection": s.collection,
                "count": len(s),
            }
            for s in self.schemes.values()
        ]
