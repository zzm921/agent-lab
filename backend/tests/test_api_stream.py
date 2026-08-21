"""API 层测试：能力目录、SSE 流式对话、审批、源码展示、健康检查（注入 Fake 运行时）。"""
import json

from fastapi.testclient import TestClient

from app.api import chat
from app.main import app


class FakeRegistry:
    def __init__(self, caps):
        self._caps = caps

    async def refresh(self):
        return None

    def list(self):
        return self._caps


class FakeRunner:
    def __init__(self, events):
        self._events = events

    async def stream(self, *args, **kwargs):
        for ev in self._events:
            yield ev

    async def resume(self, *args, **kwargs):
        for ev in self._events:
            yield ev


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                events.append(json.loads(payload))
    return events


def test_health():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_capabilities_with_fake_registry():
    chat.set_runtime(registry=FakeRegistry([{"id": "calculator", "availability": "available"}]))
    client = TestClient(app)
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    assert resp.json()["capabilities"][0]["id"] == "calculator"


def test_stream_sse_with_fake_runner():
    events = [
        {"type": "meta", "session_id": "s1", "mode": "react", "capabilities": ["calculator"]},
        {"type": "message", "delta": "你好"},
        {"type": "done", "summary": "完成", "stats": {"tool_calls": 0}},
    ]
    chat.set_runtime(runner=FakeRunner(events))
    client = TestClient(app)
    resp = client.post("/api/stream", json={"session_id": "s1", "message": "你好", "mode": "react"})
    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert parsed[0]["type"] == "meta"
    assert parsed[-1]["type"] == "done"


def test_approve_sse():
    events = [{"type": "tool_end", "tool": "calculator", "success": True, "result": "2"}, {"type": "done"}]
    chat.set_runtime(runner=FakeRunner(events))
    client = TestClient(app)
    resp = client.post(
        "/api/approve", json={"approval_id": "a1", "decision": "approve", "modified_args": None}
    )
    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert parsed[-1]["type"] == "done"


def test_source_returns_code():
    client = TestClient(app)
    resp = client.get("/api/source/react")
    assert resp.status_code == 200
    assert "build_react_agent" in resp.json()["content"]


def test_source_unknown_module():
    client = TestClient(app)
    resp = client.get("/api/source/not_a_module")
    assert resp.status_code == 200
    assert resp.json()["content"] == ""
