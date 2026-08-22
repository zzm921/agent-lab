"""MCP 开关 API 测试：默认关闭，POST /api/mcp 点选开启/关闭。"""
import pytest
from fastapi.testclient import TestClient

import app.api.chat as chat
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
