"""长期记忆管理 API 测试：GET 列表 / POST 写入 / DELETE 删除（注入 Fake 运行时）。"""
import pytest
from fastapi.testclient import TestClient

from app.api import chat
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.main import app
from app.memory.session_store import SessionStore


@pytest.fixture
def mem_registry(settings, embeddings):
    sessions = SessionStore(memory_dir=None)  # 内存态，不落盘
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), None, embeddings)
    chat.set_runtime(sessions=sessions, registry=registry)
    yield sessions
    chat.set_runtime(sessions=None, registry=None, runner=None)


def test_memory_api_write_list_delete(settings, embeddings, mem_registry):
    client = TestClient(app)
    # POST 手动写入（全局常驻库）
    resp = client.post(
        "/api/memory",
        json={"text": "用户喜欢深色主题", "kind": "preference", "importance": 0.9, "scope": "global"},
    )
    assert resp.status_code == 200
    mem_id = resp.json()["id"]

    # GET 列表
    resp = client.get("/api/memory?scope=global")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(it["text"] == "用户喜欢深色主题" and it["scope"] == "global" for it in items)

    # DELETE 删除后列表消失
    resp = client.delete(f"/api/memory/{mem_id}?scope=global")
    assert resp.status_code == 200
    resp = client.get("/api/memory?scope=global")
    assert not any(it["id"] == mem_id for it in resp.json()["items"])

    # 删除不存在 → 404
    resp = client.delete("/api/memory/not-exist?scope=global")
    assert resp.status_code == 404


def test_memory_api_kind_filter(settings, embeddings, mem_registry):
    client = TestClient(app)
    client.post("/api/memory", json={"text": "用户喜欢深色主题", "kind": "preference", "importance": 0.9})
    client.post("/api/memory", json={"text": "用户生日是 1995-08-20", "kind": "fact", "importance": 0.7})

    resp = client.get("/api/memory?kind=preference")
    items = resp.json()["items"]
    assert all(it["kind"] == "preference" for it in items)
    assert any("深色" in it["text"] for it in items)


def test_memory_api_global_isolated_by_client(settings, embeddings, mem_registry):
    """常驻记忆按设备指纹（X-Client-Id）隔离：device-a 写入，device-b 的 global 列表不可见。"""
    client = TestClient(app)
    client.post(
        "/api/memory",
        json={"text": "A 的私密偏好", "kind": "preference", "importance": 0.9, "scope": "global"},
        headers={"X-Client-Id": "device-a"},
    )

    # A 能看到自己的常驻记忆
    resp = client.get("/api/memory?scope=global", headers={"X-Client-Id": "device-a"})
    assert any("A 的私密偏好" in it["text"] for it in resp.json()["items"])

    # B 看不到 A 的常驻记忆（各自独立库）
    resp = client.get("/api/memory?scope=global", headers={"X-Client-Id": "device-b"})
    assert not any("A 的私密偏好" in it["text"] for it in resp.json()["items"])
