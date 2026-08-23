"""Qdrant 存储后端测试：使用内嵌 :memory: 模式，无需云连接。"""
import uuid

from qdrant_client import QdrantClient

from app.llm.fake_model import FakeEmbeddings
from app.memory.stores.qdrant_store import QdrantStore


def make_store(sparse: bool = False) -> QdrantStore:
    """构建绑定内嵌 client 的 QdrantStore（FakeEmbeddings 维度 32）。"""
    return QdrantStore(
        FakeEmbeddings(),
        collection=f"test_{uuid.uuid4().hex[:8]}",
        dim=32,
        sparse=sparse,
        client=QdrantClient(":memory:"),
    )


def test_add_and_dense_search():
    store = make_store()
    store.add("LangGraph 是构建有状态 Agent 的框架", {"source": "builtin"})
    store.add("ReAct 模式由 思考-行动-观察 循环组成", {"source": "builtin"})
    assert len(store) == 2
    hits = store.search("LangGraph 是构建有状态 Agent 的框架", top_k=2)
    assert hits, "应检索到结果"
    assert all("text" in h and "score" in h and "metadata" in h for h in hits)
    # 查询与某条文本完全一致（FakeEmbeddings 基于字符位置）→ 该条必然最相关
    assert hits[0]["text"] == "LangGraph 是构建有状态 Agent 的框架"
    assert hits[0]["metadata"]["source"] == "builtin"
    assert hits[0]["score"] >= hits[1]["score"]


def test_search_empty_store():
    store = make_store()
    assert store.search("任何查询") == []


def test_hybrid_search_without_sparse_falls_back_to_dense():
    """未开启稀疏向量时 hybrid_search 退化为稠密检索。"""
    store = make_store(sparse=False)
    store.add("LangGraph 基于 StateGraph 构建 Agent")
    hits = store.hybrid_search("LangGraph", top_k=2)
    assert len(hits) == 1
    assert hits[0]["text"] == "LangGraph 基于 StateGraph 构建 Agent"


def test_hybrid_search_sparse():
    store = make_store(sparse=True)
    store.add("LangGraph 基于 StateGraph 构建 Agent")
    store.add("MCP 把外部工具封装为 Server")
    hits = store.hybrid_search("StateGraph 是什么", top_k=2)
    assert len(hits) == 2
    # 稀疏（关键词）信号应能召回含 StateGraph 的片段
    assert any("StateGraph" in h["text"] for h in hits)


def test_collection_name():
    store = make_store()
    assert store.name == "qdrant"
    assert store.collection.startswith("test_")


def test_all_texts_and_clear():
    """all_texts 可读回全部已入库文本；clear 清空集合（保留结构）。"""
    store = make_store()
    store.add("LangGraph 是构建有状态 Agent 的框架")
    store.add("ReAct 模式由 思考-行动-观察 循环组成")
    assert sorted(store.all_texts()) == sorted(
        ["LangGraph 是构建有状态 Agent 的框架", "ReAct 模式由 思考-行动-观察 循环组成"]
    )
    store.clear()
    assert len(store) == 0
    assert store.all_texts() == []
    # 清空后仍可继续入库（集合结构保留）
    store.add("MCP 把外部工具封装为 Server")
    assert len(store) == 1
