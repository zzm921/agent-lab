"""Elasticsearch 存储后端测试：全程离线（fake ES client + FakeEmbeddings）。

覆盖：稠密 kNN 相似度检索、混合检索（RRF 请求体）、数据生命周期、
以及 manager 按 rag_store_backend 分发与回退逻辑。
"""
import pytest

from app.core.errors import ConfigError
from app.llm.fake_model import FakeEmbeddings
from app.memory.stores.elasticsearch_store import ElasticsearchStore
from app.memory.stores.memory_store import MemoryStore
from app.memory.stores.multi_backend_store import MultiBackendStore
from app.rag.manager import RagManager

CORPUS = [
    "ReAct 模式由 思考-行动-观察 循环组成。",
    "LangGraph 基于 StateGraph 构建有状态、多步骤的 AI Agent。",
    "MCP 把外部工具封装为 Server，Agent 可动态发现调用。",
]


class FakeIndices:
    def __init__(self) -> None:
        self._exists: set[str] = set()

    def exists(self, index: str) -> bool:
        return index in self._exists

    def create(self, index: str, mappings=None) -> None:
        self._exists.add(index)


class FakeESClient:
    """内存版 ES 客户端：只实现 ElasticsearchStore 用到的 API。"""

    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.docs: dict[str, list[dict]] = {}
        self.calls: list[tuple[str, dict]] = []

    def index(self, index: str, id: str, document: dict) -> None:
        self.docs.setdefault(index, []).append({"_id": id, "_source": document})

    def search(self, index: str, **kwargs) -> dict:
        # 只记录实际传入的参数，便于断言「未传 query/rank」等行为
        self.calls.append({"index": index, **kwargs})
        docs = self.docs.get(index, [])
        # kNN 相似度检索的正确性属于 ES 本身，fake 只做确定性的适配层模拟：
        # 返回前 size 条（带 mock score），验证 store 正确构造请求体并映射响应。
        size = kwargs.get("size", 10)
        hits = [{"_source": d["_source"], "_score": 0.9} for d in docs[:size]]
        return {"hits": {"hits": hits}}

    def delete_by_query(self, index: str, query: dict, refresh=None) -> None:
        if query == {"match_all": {}}:
            self.docs[index] = []

    def count(self, index: str) -> dict:
        return {"count": len(self.docs.get(index, []))}


def make_store(client, hybrid: bool = False) -> ElasticsearchStore:
    return ElasticsearchStore(
        FakeEmbeddings(),
        index="knowledge_test",
        dim=32,
        hybrid=hybrid,
        client=client,
    )


def test_no_url_raises_config_error():
    with pytest.raises(ConfigError):
        ElasticsearchStore(FakeEmbeddings(), index="x", dim=32)  # 无 url 且无注入 client


def test_add_search_len_clear():
    client = FakeESClient()
    store = make_store(client)
    assert store.collection == "knowledge_test"
    assert len(store) == 0
    for text in CORPUS:
        store.add(text)
    assert len(store) == 3
    hits = store.search("ReAct", top_k=1)
    assert len(hits) == 1
    assert hits[0]["text"] == CORPUS[0]
    assert "score" in hits[0] and "metadata" in hits[0]
    store.clear()
    assert len(store) == 0


def test_search_uses_knn():
    client = FakeESClient()
    store = make_store(client)
    store.add("ReAct 模式")
    store.search("ReAct", top_k=2)
    call = client.calls[-1]
    assert call["knn"]["field"] == "dense_vector"
    assert call["knn"]["k"] == 2
    assert call["size"] == 2
    assert "query" not in call  # 纯稠密相似度检索不携带关键词查询


def test_hybrid_dense_only_when_not_hybrid():
    client = FakeESClient()
    store = make_store(client, hybrid=False)
    store.add("ReAct 模式")
    store.hybrid_search("ReAct", top_k=2)
    call = client.calls[-1]
    assert "query" not in call and "rank" not in call


def test_hybrid_rrf_when_hybrid():
    client = FakeESClient()
    store = make_store(client, hybrid=True)
    store.add("ReAct 模式")
    store.hybrid_search("ReAct", top_k=3)
    call = client.calls[-1]
    assert call["query"] == {"match": {"text": "ReAct"}}
    assert call["rank"] == {"rrf": {}}  # ES 原生 RRF 融合多路召回
    # 空查询退化为纯 kNN（无关键词路）
    store.hybrid_search("  ", top_k=3)
    call = client.calls[-1]
    assert "query" not in call and "rank" not in call


def test_all_texts():
    client = FakeESClient()
    store = make_store(client)
    for text in CORPUS:
        store.add(text)
    assert set(store.all_texts()) == set(CORPUS)


def test_manager_builds_elasticsearch_store(settings, monkeypatch):
    """rag_store_backend=elasticsearch：每个方案独立索引，advanced 开启混合检索。"""
    captured: dict[str, bool] = {}

    def fake_es_store(embeddings, index, url, api_key, username, password, dim, hybrid=False, client=None):
        captured[index] = hybrid
        return MemoryStore(embeddings, collection=index)

    monkeypatch.setattr("app.rag.manager.ElasticsearchStore", fake_es_store)
    settings.rag_store_backend = "elasticsearch"
    settings.es_url = "http://localhost:9200"
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    assert captured["knowledge_advanced"] is True
    assert captured["knowledge_naive"] is False


def test_manager_es_without_url_falls_back_memory(settings):
    settings.rag_store_backend = "elasticsearch"
    settings.es_url = ""
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    for scheme in manager.schemes.values():
        assert isinstance(scheme.store, MemoryStore)


def test_manager_es_failure_falls_back_memory(settings, monkeypatch):
    def boom(*args, **kwargs):
        raise ConfigError("ES 连接失败")

    monkeypatch.setattr("app.rag.manager.ElasticsearchStore", boom)
    settings.rag_store_backend = "elasticsearch"
    settings.es_url = "http://localhost:9200"
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    for scheme in manager.schemes.values():
        assert isinstance(scheme.store, MemoryStore)


def test_manager_memory_backend_forced(settings):
    """rag_store_backend=memory：即使配置了 Qdrant 也强制内存（离线/测试用）。"""
    settings.rag_store_backend = "memory"
    settings.qdrant_url = "http://localhost:6333"
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    for scheme in manager.schemes.values():
        assert isinstance(scheme.store, MemoryStore)


# ---------- 跨后端多路召回（advanced = Qdrant + ES） ----------


def test_multi_backend_add_len_all_texts_clear():
    store = MultiBackendStore([MemoryStore(FakeEmbeddings()), MemoryStore(FakeEmbeddings())])
    for text in CORPUS:
        store.add(text)
    assert len(store) == len(CORPUS)  # 各后端入库同一语料
    assert set(store.all_texts()) == set(CORPUS)  # 融合后端去重取并集
    store.clear()
    assert len(store) == 0


def test_multi_backend_fuse_dedupes_by_text():
    a = MemoryStore(FakeEmbeddings())
    b = MemoryStore(FakeEmbeddings())
    a.add("ReAct 模式由 思考-行动-观察 循环组成。")
    b.add("ReAct 模式由 思考-行动-观察 循环组成。")  # 与 a 重复
    b.add("MCP 把外部工具封装为 Server。")
    store = MultiBackendStore([a, b])
    hits = store.search("ReAct", top_k=5)
    texts = [h["text"] for h in hits]
    assert len(texts) == len(set(texts))  # 跨后端按文本去重
    assert len(hits) == 2


def test_multi_backend_skips_failing_backend():
    class Boom:
        name = "boom"
        collection = "boom"
        def add(self, text, metadata=None):
            pass
        def search(self, query, top_k=3):
            raise RuntimeError("ES 挂了")
        def hybrid_search(self, query, top_k=3):
            raise RuntimeError("ES 挂了")
        def all_texts(self):
            return []
        def clear(self):
            pass
        def __len__(self):
            return 0

    good = MemoryStore(FakeEmbeddings())
    good.add("ReAct 模式由 思考-行动-观察 循环组成。")
    store = MultiBackendStore([Boom(), good])
    hits = store.search("ReAct", top_k=3)
    assert len(hits) == 1  # 一路失败不影响另一路召回


def test_advanced_multi_backend_qdrant_plus_es(settings, monkeypatch):
    """主 Qdrant + ES 已配置：advanced 跨后端多路召回（naive 仍单后端）。"""
    created: dict[str, str] = {}

    def fake_qdrant(*args, **kwargs):
        created.setdefault("qdrant", []).append(kwargs["collection"])
        return MemoryStore(FakeEmbeddings(), collection=kwargs["collection"])

    def fake_es(*args, **kwargs):
        created.setdefault("es", []).append(kwargs["index"])
        return MemoryStore(FakeEmbeddings(), collection=kwargs["index"])

    monkeypatch.setattr("app.rag.manager.QdrantStore", fake_qdrant)
    monkeypatch.setattr("app.rag.manager.ElasticsearchStore", fake_es)
    settings.qdrant_url = "http://localhost:6333"
    settings.es_url = "http://localhost:9200"
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    naive = manager.get("naive")
    advanced = manager.get("advanced")
    assert isinstance(naive.store, MemoryStore)  # naive 单后端
    assert isinstance(advanced.store, MultiBackendStore)  # advanced 多路召回
    assert len(advanced.store.backends) == 2
    assert {b.collection for b in advanced.store.backends} == {
        "knowledge_advanced",
        "knowledge_advanced",
    }


def test_advanced_single_backend_without_es(settings, monkeypatch):
    """只配 Qdrant、未配 ES：advanced 退化为单后端（Qdrant），不叠加内存路。"""
    def fake_qdrant(*args, **kwargs):
        return MemoryStore(FakeEmbeddings(), collection=kwargs["collection"])

    monkeypatch.setattr("app.rag.manager.QdrantStore", fake_qdrant)
    settings.qdrant_url = "http://localhost:6333"
    settings.es_url = ""
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    assert not isinstance(manager.get("advanced").store, MultiBackendStore)


def test_advanced_multi_es_plus_qdrant(settings, monkeypatch):
    """主 ES + 配了 Qdrant：advanced 主 ES、叠加 Qdrant 多路召回。"""
    created: dict[str, str] = {}

    def fake_qdrant(*args, **kwargs):
        created["qdrant"] = kwargs["collection"]
        return MemoryStore(FakeEmbeddings(), collection=kwargs["collection"])

    def fake_es(*args, **kwargs):
        created["es"] = kwargs["index"]
        return MemoryStore(FakeEmbeddings(), collection=kwargs["index"])

    monkeypatch.setattr("app.rag.manager.QdrantStore", fake_qdrant)
    monkeypatch.setattr("app.rag.manager.ElasticsearchStore", fake_es)
    settings.rag_store_backend = "elasticsearch"
    settings.es_url = "http://localhost:9200"
    settings.qdrant_url = "http://localhost:6333"
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    advanced = manager.get("advanced")
    assert isinstance(advanced.store, MultiBackendStore)
    assert created["es"] == "knowledge_advanced"
    assert created["qdrant"] == "knowledge_advanced"


# ---------- ES < 8.x 兼容路径（旧版无 dense_vector/kNN/RRF → BM25 关键词） ----------


class _Resp:
    """模拟 httpx.Response 的最小对象。"""

    def __init__(self, status_code: int = 200, data: dict | None = None):
        self.status_code = status_code
        self._data = data or {}
        self.text = ""

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeLegacyHTTP:
    """内存版旧版 ES 的 HTTP 层：只实现兼容路径用到的 REST API。"""

    def __init__(self) -> None:
        self.docs: list[dict] = []
        self.calls: list[tuple[str, str]] = []
        self._exists = False

    def get(self, path: str, **kw) -> _Resp:
        self.calls.append(("get", path))
        if path.startswith("/count") or path.endswith("/_count"):
            return _Resp(data={"count": len(self.docs)})
        if self._exists:
            return _Resp()
        return _Resp(status_code=404)

    def put(self, path: str, json=None, **kw) -> _Resp:
        self.calls.append(("put", path))
        if path.count("/") == 1:  # 建索引
            self._exists = True
            return _Resp(status_code=200)
        self.docs.append(json or {})  # 写文档
        return _Resp(status_code=201)

    def post(self, path: str, json=None, **kw) -> _Resp:
        self.calls.append(("post", path))
        if path.endswith("/_delete_by_query"):
            self.docs.clear()
            return _Resp()
        body = json or {}
        size = body.get("size", 10)
        query = body.get("query", {})
        hits = []
        for d in self.docs:
            matched = query.get("match_all") is not None
            if not matched:
                term = query.get("match", {}).get("text", "")
                matched = term and term in d.get("text", "")
            if matched:
                hits.append({"_source": d, "_score": 0.9})
        return _Resp(data={"hits": {"hits": hits[:size]}})

    def delete(self, path: str, **kw) -> _Resp:
        self.calls.append(("delete", path))
        return _Resp()


def _make_legacy_store(http) -> ElasticsearchStore:
    store = ElasticsearchStore(
        FakeEmbeddings(),
        index="knowledge_test",
        url="http://es:9200",
        dim=32,
        hybrid=True,
        http_client=http,
    )
    assert store._legacy is True  # _detect_legacy 由真实版本决定；此处注入 http 后仍走兼容
    return store


def test_legacy_bm25_lifecycle(monkeypatch):
    """ES <8.x 兼容路径：BM25 增删查生命周期（不联网，fake HTTP 层）。"""
    monkeypatch.setattr(ElasticsearchStore, "_detect_legacy", staticmethod(lambda *a, **k: True))
    http = _FakeLegacyHTTP()
    store = _make_legacy_store(http)
    assert store.collection == "knowledge_test"
    assert len(store) == 0
    for text in CORPUS:
        store.add(text)
    assert len(store) == 3
    # BM25 关键词检索（兼容路径无向量字段）
    hits = store.search("MCP", top_k=2)
    assert len(hits) == 1 and hits[0]["text"] == CORPUS[2]
    hits = store.hybrid_search("MCP", top_k=2)
    assert len(hits) == 1
    assert set(store.all_texts()) == set(CORPUS)
    store.clear()
    assert len(store) == 0
    # 建索引用的是旧式 _doc 类型映射（single_type=false 集群）
    assert ("put", "/knowledge_test") in http.calls


def test_legacy_detects_old_version(monkeypatch):
    """版本探测：<8.x 返回 True（走兼容路径）。"""
    resp = _Resp(data={"version": {"number": "6.8.1"}})
    monkeypatch.setattr("app.memory.stores.elasticsearch_store.httpx.get", lambda *a, **k: resp)
    assert ElasticsearchStore._detect_legacy("http://es:9200", "", "", "") is True
    resp2 = _Resp(data={"version": {"number": "8.15.0"}})
    monkeypatch.setattr("app.memory.stores.elasticsearch_store.httpx.get", lambda *a, **k: resp2)
    assert ElasticsearchStore._detect_legacy("http://es:9200", "", "", "") is False
