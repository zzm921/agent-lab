"""RAG 方案管理器测试：内存回退 + 内嵌 Qdrant 注入，均离线。"""
import uuid

import pytest
from qdrant_client import QdrantClient

from app.core.errors import ConfigError
from app.llm.fake_model import FakeEmbeddings
from app.memory.stores.memory_store import MemoryStore
from app.memory.stores.qdrant_store import QdrantStore
from app.rag.manager import RagManager
from app.rag.naive import CHUNK_SIZE

CORPUS = [
    "LangGraph 基于 StateGraph 构建有状态、多步骤的 AI Agent。",
    "ReAct 模式由 思考-行动-观察 循环组成。",
    "MCP 把外部工具封装为 Server，Agent 可动态发现调用。",
]


def make_manager(settings, **kw) -> RagManager:
    return RagManager(settings, FakeEmbeddings(), top_k=settings.rag_top_k, **kw)


def test_builds_schemes_memory_fallback(settings):
    manager = make_manager(settings)  # 未配 Qdrant → 内存回退
    ids = {s.id for s in manager.schemes.values()}
    assert ids == {"naive", "advanced"}
    for scheme in manager.schemes.values():
        assert isinstance(scheme.store, MemoryStore)
        assert scheme.collection == f"{settings.qdrant_collection_prefix}_{scheme.id}"
        assert len(scheme) == 0


def test_unknown_scheme_raises(settings):
    settings.rag_schemes = ["graph"]
    with pytest.raises(ConfigError):
        make_manager(settings)


def test_scheme_ids_builds_only_selected(settings):
    """指定 scheme_ids 时只构建对应方案（离线建库按方案独立脚本用）。"""
    manager = RagManager(settings, FakeEmbeddings(), top_k=3, scheme_ids=["naive"])
    assert set(manager.schemes) == {"naive"}
    manager.ingest_all(CORPUS)
    assert len(manager.get("naive")) == len(CORPUS)
    # 缺省参数时仍按 settings.rag_schemes 构建全部方案
    manager = make_manager(settings)
    assert set(manager.schemes) == {"naive", "advanced"}


def test_ingest_all_idempotent(settings):
    manager = make_manager(settings)
    manager.ingest_all(CORPUS)
    assert len(manager.schemes["naive"]) == len(CORPUS)  # 短句不切块，一条一块
    # 幂等：再次入库不重复
    manager.ingest_all(CORPUS)
    assert len(manager.schemes["naive"]) == len(CORPUS)


def test_naive_fixed_chunking(settings):
    """固定切块：长文本被硬切为多块，演示固定切块切断长文本语义。"""
    manager = make_manager(settings)
    naive = manager.get("naive")
    naive.ingest(["长文本" * CHUNK_SIZE])  # CHUNK_SIZE × 3 字符
    assert len(naive) == 3


def test_ingest_rebuilds_on_corpus_change(settings):
    """语料更新后自动清空重建，避免旧语料残留。"""
    manager = make_manager(settings)
    manager.ingest_all(CORPUS)
    assert len(manager.get("naive")) == len(CORPUS)
    new_corpus = CORPUS + ["新增的制度条款：科创公司差旅餐补 100 元/天。"]
    manager.ingest_all(new_corpus)
    assert len(manager.get("naive")) == len(new_corpus)  # 重建后为新语料条数
    # 再次入库同一语料：幂等跳过
    manager.ingest_all(new_corpus)
    assert len(manager.get("naive")) == len(new_corpus)


def test_resolve(settings):
    manager = make_manager(settings)
    assert manager.resolve(None).id == settings.rag_default_scheme
    assert manager.resolve("naive").id == "naive"
    assert manager.resolve("不存在的方案").id == settings.rag_default_scheme
    assert manager.get("naive") is manager.schemes["naive"]
    assert manager.get("nope") is None


def test_naive_retrieval(settings):
    """naive 走纯稠密检索，可召回相关片段。"""
    manager = make_manager(settings)
    manager.ingest_all(CORPUS)
    naive = manager.get("naive")
    assert naive.retrieve("LangGraph StateGraph", top_k=2)


def test_list_entries(settings):
    manager = make_manager(settings)
    manager.ingest_all(CORPUS)
    entries = {e["id"]: e for e in manager.list()}
    assert set(entries) == {"naive", "advanced"}
    for e in entries.values():
        assert e["name"] and e["collection"] and e["count"] >= 1


def test_advanced_store_uses_sparse(settings, monkeypatch):
    """manager 为 advanced 构建稀疏集合（混合多路召回），naive 保持纯稠密。"""
    captured: dict[str, bool] = {}

    def fake_qdrant_store(embeddings, collection, url, api_key, dim, sparse=False, client=None):
        captured[collection] = sparse
        return MemoryStore(embeddings, collection=collection)

    monkeypatch.setattr("app.rag.manager.QdrantStore", fake_qdrant_store)
    settings.qdrant_url = "http://localhost:6333"
    RagManager(settings, FakeEmbeddings(), top_k=3)
    assert captured["knowledge_advanced"] is True
    assert captured["knowledge_naive"] is False


def test_with_qdrant_store_injected(settings):
    """注入内嵌 QdrantStore（独立集合），验证方案绑定独立库（advanced 启用稀疏）。"""
    suffix = uuid.uuid4().hex[:8]
    stores = {
        "naive": QdrantStore(FakeEmbeddings(), collection=f"n_{suffix}", dim=32, client=QdrantClient(":memory:")),
        "advanced": QdrantStore(FakeEmbeddings(), collection=f"a_{suffix}", dim=32, sparse=True, client=QdrantClient(":memory:")),
    }
    manager = RagManager(settings, FakeEmbeddings(), top_k=settings.rag_top_k, stores=stores)
    manager.ingest_all(CORPUS)
    assert manager.get("naive").store.collection == f"n_{suffix}"
    assert manager.get("advanced").store.collection == f"a_{suffix}"
    assert len(manager.get("naive")) == len(CORPUS)
    assert len(manager.get("advanced")) == len(CORPUS)  # 短句不拆块，一条一块
