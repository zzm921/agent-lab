"""API 层测试：能力目录、SSE 流式对话、审批、故障注入、源码展示、健康检查、沙箱文件（注入 Fake 运行时）。"""
import json

from fastapi.testclient import TestClient

from app.agents.harness import AgentHarness
from app.api import chat, sandbox
from app.config import Settings
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
        self.harness = AgentHarness(
            Settings(llm_api_key="test-key", embedding_api_key="test-key", mcp_servers="{}")
        )

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


def test_fault_set_list_and_clear():
    chat.set_runtime(runner=FakeRunner([]))
    client = TestClient(app)
    resp = client.post("/api/fault", json={"tool": "calculator", "mode": "error"})
    assert resp.status_code == 200
    assert resp.json()["faults"] == {"calculator": "error"}
    resp = client.get("/api/faults")
    assert resp.status_code == 200
    assert resp.json()["faults"] == {"calculator": "error"}
    resp = client.post("/api/fault", json={"tool": "calculator", "mode": "off"})
    assert resp.status_code == 200
    assert resp.json()["faults"] == {}


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


def test_sandbox_files_list_and_download(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir(parents=True)
    (work / "a.txt").write_text("hello", encoding="utf-8")
    (work / "sub").mkdir()
    (work / "sub" / "b.log").write_text("log", encoding="utf-8")
    monkeypatch.setattr(sandbox, "_work_dir", lambda: work)

    client = TestClient(app)
    resp = client.get("/api/sandbox/files")
    assert resp.status_code == 200
    paths = [f["path"] for f in resp.json()["files"]]
    assert paths == ["a.txt", "sub/b.log"]

    dl = client.get("/api/sandbox/files/download", params={"path": "sub/b.log"})
    assert dl.status_code == 200
    assert dl.content == b"log"


def test_sandbox_files_download_traversal_blocked(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir(parents=True)
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(sandbox, "_work_dir", lambda: work)

    client = TestClient(app)
    resp = client.get("/api/sandbox/files/download", params={"path": "../secret.txt"})
    assert resp.status_code == 400
    resp = client.get(
        "/api/sandbox/files/download",
        params={"path": str(tmp_path / "secret.txt")},
    )
    assert resp.status_code == 400


def test_sandbox_files_download_not_found(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(sandbox, "_work_dir", lambda: work)

    client = TestClient(app)
    resp = client.get("/api/sandbox/files/download", params={"path": "missing.txt"})
    assert resp.status_code == 404
