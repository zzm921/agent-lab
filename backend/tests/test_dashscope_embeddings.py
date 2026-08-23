"""DashScope Embedding 适配测试：桩掉 TextEmbedding.call，不联网。"""
from types import SimpleNamespace

import pytest

from app.core.errors import ConfigError
from app.llm.dashscope_embeddings import DashScopeEmbeddings


def _resp(status_code: int, embeddings: list[dict]):
    # 新版 dashscope SDK：resp.output 为 dict（{"embeddings": [...]}）
    return SimpleNamespace(status_code=status_code, output={"embeddings": embeddings})


def test_embed_query(monkeypatch):
    calls = []
    model = "text-embedding-v3"

    def fake_call(**kwargs):
        calls.append(kwargs)
        return _resp(200, [{"embedding": [0.1, 0.2, 0.3]}])

    monkeypatch.setattr("app.llm.dashscope_embeddings.TextEmbedding.call", staticmethod(fake_call))
    emb = DashScopeEmbeddings("test-key")
    vec = emb.embed_query("你好")
    assert vec == [0.1, 0.2, 0.3]
    assert calls[0]["model"] == model
    assert calls[0]["input"] == ["你好"]
    assert calls[0]["text_type"] == "query"


def test_embed_documents(monkeypatch):
    def fake_call(**kwargs):
        return _resp(200, [{"embedding": [1.0]}, {"embedding": [2.0]}])

    monkeypatch.setattr("app.llm.dashscope_embeddings.TextEmbedding.call", staticmethod(fake_call))
    emb = DashScopeEmbeddings("test-key")
    assert emb.embed_documents(["a", "b"]) == [[1.0], [2.0]]
    assert emb.embed_documents([]) == []


def test_embed_sparse(monkeypatch):
    def fake_call(**kwargs):
        assert kwargs["model"] == "text-sparse-embedding-v1"
        return _resp(200, [{"token_ids": [3, 7], "embeddings": [0.6, 0.4]}])

    monkeypatch.setattr("app.llm.dashscope_embeddings.TextEmbedding.call", staticmethod(fake_call))
    emb = DashScopeEmbeddings("test-key")
    assert emb.embed_sparse_query("关键词") == {"indices": [3, 7], "values": [0.6, 0.4]}


def test_api_error_raises_config_error(monkeypatch):
    def fake_call(**kwargs):
        return _resp(400, [])

    monkeypatch.setattr("app.llm.dashscope_embeddings.TextEmbedding.call", staticmethod(fake_call))
    emb = DashScopeEmbeddings("test-key")
    with pytest.raises(ConfigError):
        emb.embed_query("触发错误")


def test_sparse_falls_back_to_local_when_model_unavailable(monkeypatch):
    """稀疏模型不可用（400）时不抛错，回退本地 n-gram 稀疏向量，保证混合检索可用。"""
    def fake_call(**kwargs):
        return _resp(400, [])

    monkeypatch.setattr("app.llm.dashscope_embeddings.TextEmbedding.call", staticmethod(fake_call))
    emb = DashScopeEmbeddings("test-key")
    sparse = emb.embed_sparse_query("ReAct 模式调用工具")
    assert set(sparse) == {"indices", "values"}
    assert len(sparse["indices"]) > 0
    assert len(sparse["indices"]) == len(sparse["values"])


def test_local_sparse_is_deterministic():
    from app.llm.dashscope_embeddings import local_sparse

    a = local_sparse("ReAct 模式 思考-行动-观察")
    b = local_sparse("ReAct 模式 思考-行动-观察")
    assert a == b
    # 不同文本信号有差异
    c = local_sparse("MCP 协议 封装工具")
    assert a != c
