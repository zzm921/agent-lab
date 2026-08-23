"""pytest 共享夹具：全部使用 Fake 模型，不联网、不依赖 Key。"""
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from langchain_core.messages import AIMessage  # noqa: E402

from app.agents.runner import AgentRunner  # noqa: E402
from app.capabilities.mcp import McpManager  # noqa: E402
from app.capabilities.registry import CapabilityRegistry  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm.fake_model import FakeChatModel, FakeEmbeddings  # noqa: E402
from app.memory.session_store import SessionStore  # noqa: E402
from app.memory.vector_store import VectorStore  # noqa: E402
from app.rag.manager import RagManager  # noqa: E402


def make_settings(**kw) -> Settings:
    defaults = {
        "llm_api_key": "test-key",
        "embedding_api_key": "test-key",
        "mcp_servers": "{}",
        # 测试强制离线：显式清空 Qdrant / ES 配置，避免读到开发机 .env 的真实实例
        "qdrant_url": "",
        "qdrant_api_key": "",
        "es_url": "",
        "es_api_key": "",
        "es_username": "",
        "es_password": "",
        # 与 .env 解耦：测试固定启用 naive + advanced 两方案
        "rag_schemes": ["naive", "advanced"],
    }
    defaults.update(kw)
    return Settings(**defaults)


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.fixture
def sessions():
    return SessionStore()


@pytest.fixture
def corpus(embeddings):
    vs = VectorStore(embeddings, name="knowledge")
    vs.add("LangGraph 基于 StateGraph 构建有状态、多步骤的 AI Agent。")
    vs.add("ReAct 模式由 思考-行动-观察 循环组成。")
    return vs


@pytest.fixture
def rag_manager(settings, embeddings):
    """多 RAG 方案管理器（未配 Qdrant → 内存存储回退）。"""
    return RagManager(settings, embeddings, top_k=settings.rag_top_k)


@pytest.fixture
def registry(settings, sessions, rag_manager, embeddings):
    mcp = McpManager("{}")
    return CapabilityRegistry(settings, sessions, mcp, rag_manager, embeddings)


@pytest.fixture
def runner(settings, registry, sessions):
    return AgentRunner(settings, FakeChatModel(), registry, sessions)


def ai_with_tool(content: str, name: str = "calculator", args: dict | None = None, cid: str = "call_1"):
    """构造带 tool_calls 的 AI 消息。"""
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args or {}, "id": cid, "type": "tool_call"}],
    )


async def collect_stream(runner, **kwargs):
    """运行 runner.stream 并收集全部事件。"""
    defaults = {
        "session_id": "s1",
        "message": "测试任务",
        "mode": "react",
        "enabled": ["calculator"],
        "prompt_strategy": "standard",
        "approval_policy": "never",
    }
    defaults.update(kwargs)
    events = []
    async for ev in runner.stream(**defaults):
        events.append(ev)
    return events
