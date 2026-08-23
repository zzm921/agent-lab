"""能力注册表：内置能力可用性判断（未配 Embedding → rag/memory 不适配）。"""
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.memory.session_store import SessionStore


def test_builtin_available(settings, sessions, rag_manager, embeddings):
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), rag_manager, embeddings)
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


def test_rag_is_not_a_tool(settings, sessions, rag_manager, embeddings):
    """RAG 是独立检索阶段（runner 前置检索 + 上下文注入），不映射为 LangChain 工具。"""
    rag_manager.ingest_all(["LangGraph 基于 StateGraph 构建 Agent。"])
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), rag_manager, embeddings)
    assert registry.tool_for("rag", "s1", lambda e: None) is None
    assert registry.tool_for("calculator", "s1") is not None


def test_rag_schemes_directory(settings, sessions, rag_manager, embeddings):
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), rag_manager, embeddings)
    schemes = {s["id"]: s for s in registry.rag_schemes()}
    assert set(schemes) == {"naive", "advanced"}
    assert schemes["naive"]["name"] == "朴素 RAG"
    assert schemes["naive"]["collection"].endswith("_naive")
    assert schemes["advanced"]["name"] == "高级 RAG"
    # 未注入 rag_manager 时为空目录
    empty = CapabilityRegistry(settings, sessions, McpManager("{}"), None, None)
    assert empty.rag_schemes() == []
