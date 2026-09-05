"""MCP 开关 API 测试：默认关闭，POST /api/mcp 点选开启/关闭；开关仅控制能力是否入目录。"""
import pytest
from fastapi.testclient import TestClient

import app.api.chat as chat
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.main import app

client = TestClient(app)


@pytest.fixture
def mcp_runtime(registry):
    """注入带空 McpManager 的 registry（不联网），用后还原。"""
    chat.set_runtime(sessions=registry.sessions, registry=registry, runner=None)
    yield registry
    chat.set_runtime(sessions=None, registry=None, runner=None)


def test_mcp_default_disabled(mcp_runtime):
    resp = client.get("/api/mcp")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False
    assert data["capabilities"] == []


def test_mcp_toggle_on_then_off(mcp_runtime):
    resp = client.post("/api/mcp", json={"enabled": True})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    assert client.get("/api/mcp").json()["enabled"] is True

    resp = client.post("/api/mcp", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False


def test_registry_filters_mcp_caps_by_enabled(settings, sessions, rag_manager, embeddings):
    """开关仅控制 MCP 能力是否进入能力目录；已建立连接（_discovered）不受开关影响。"""
    mcp = McpManager("{}")
    mcp._discovered = True  # 模拟服务启动时已连接并发现工具
    mcp.capabilities.append(
        {
            "id": "mcp-info:now",
            "name": "mcp-info · now",
            "source": "mcp",
            "server": "mcp-info",
            "requires": None,
            "availability": "available",
            "unavailable_reason": None,
            "desc": "now",
            "example": "",
            "code_key": "mcp",
        }
    )
    registry = CapabilityRegistry(settings, sessions, mcp, rag_manager, embeddings)
    ids = {c["id"] for c in registry.list()}
    assert "mcp-info:now" not in ids  # 默认未启用 → 能力目录不含 MCP
    mcp.enabled = True
    ids = {c["id"] for c in registry.list()}
    assert "mcp-info:now" in ids  # 开启后进入能力目录
    assert mcp._discovered is True  # 开关切换不改变连接状态


def test_mcp_disable_keeps_connection_and_re_enable_no_reconnect(settings, sessions, rag_manager, embeddings):
    """关闭仅隐藏能力、不重置连接；再次开启直接恢复，无需重连。"""
    import asyncio

    mcp = McpManager("{}")
    mcp._discovered = True  # 模拟已连接
    mcp.enabled = True
    CapabilityRegistry(settings, sessions, mcp, rag_manager, embeddings)
    mcp.disable()
    assert mcp.enabled is False
    assert mcp._discovered is True  # 连接保持
    # enable() 为 async；_discovered 已为 True 时不再触发 discover（不会重新连接）
    asyncio.run(mcp.enable())
    assert mcp.enabled is True
    assert mcp._discovered is True
