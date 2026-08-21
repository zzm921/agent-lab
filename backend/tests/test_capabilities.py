"""能力注册表：内置能力可用性判断（未配 Embedding → rag/memory 不适配）。"""
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.memory.session_store import SessionStore


def test_builtin_available(settings, sessions, corpus, embeddings):
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), corpus, embeddings)
    caps = {c["id"]: c for c in registry.list()}
    assert caps["calculator"]["availability"] == "available"
    assert caps["time_now"]["availability"] == "available"
    assert caps["rag"]["availability"] == "available"
    assert caps["memory"]["availability"] == "available"


def test_rag_memory_unavailable_without_embedding(settings, sessions):
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), None, None)
    caps = {c["id"]: c for c in registry.list()}
    assert caps["calculator"]["availability"] == "available"
    assert caps["rag"]["availability"] == "unavailable"
    assert caps["memory"]["availability"] == "unavailable"
    assert "Embedding" in caps["rag"]["unavailable_reason"]


def test_tool_for_unavailable_returns_none(settings, sessions):
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), None, None)
    registry.list()
    assert registry.tool_for("rag", "s1") is None
    assert registry.tool_for("calculator", "s1") is not None
