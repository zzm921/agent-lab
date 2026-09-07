"""会话（多会话）测试：元数据 CRUD、客户端隔离、文件持久化恢复、SQLite 检查点、TTL 清理、API 层。

全部离线：SessionStore 用 tmp 目录，不碰真实 data/；API 测试注入独立的 SessionStore 到运行时。
"""
import asyncio
import os
import time

import pytest
from fastapi.testclient import TestClient

from app.api import chat
from app.main import app
from app.memory.session_store import SessionStore


def _store(tmp_path, ttl=7, checkpoint=True):
    return SessionStore(
        memory_dir=str(tmp_path / "memory"),
        checkpoint_dir=str(tmp_path / "checkpoints") if checkpoint else None,
        sessions_meta_path=str(tmp_path / "sessions.jsonl"),
        checkpoint_ttl_days=ttl,
        events_dir=str(tmp_path / "events"),
    )


# ---- 元数据 CRUD ----

def test_create_list_get(tmp_path):
    store = _store(tmp_path)
    sid = store.create(client_key="cid:device-a")
    rec = store.get(sid)
    assert rec is not None
    assert rec["client_key"] == "cid:device-a"
    assert rec["title"] == ""
    # 列表按最后活跃倒序；同客户端可见
    listed = store.list(client_key="cid:device-a")
    assert [r["session_id"] for r in listed] == [sid]


def test_client_isolation(tmp_path):
    store = _store(tmp_path)
    a = store.create(client_key="cid:device-a")
    store.create(client_key="cid:device-b")
    assert [r["session_id"] for r in store.list("cid:device-a")] == [a]
    assert len(store.list("cid:device-b")) == 1


def test_touch_and_title(tmp_path):
    store = _store(tmp_path)
    sid = store.create(client_key="cid:device-a")
    store.title_if_empty(sid, "cid:device-a", "请计算 137 除以 3 的结果")
    assert store.get(sid)["title"] == "请计算 137 除以 3 的结果"
    # 已有标题不再覆盖
    store.title_if_empty(sid, "cid:device-a", "再问一次")
    assert store.get(sid)["title"] == "请计算 137 除以 3 的结果"
    # touch 递增消息数、刷新活跃时间
    old = store.get(sid)["last_active"]
    store.touch(sid, "cid:device-a")
    assert store.get(sid)["message_count"] == 1
    assert store.get(sid)["last_active"] >= old


def test_rename(tmp_path):
    store = _store(tmp_path)
    sid = store.create(client_key="cid:device-a")
    store.rename(sid, "手动标题")
    assert store.get(sid)["title"] == "手动标题"


@pytest.mark.asyncio
async def test_delete_removes_meta_and_memory_file(tmp_path):
    store = _store(tmp_path)
    sid = store.create(client_key="cid:device-a")
    # 会话记忆文件：写入后随删除清理
    mem_path = f"{tmp_path / 'memory'}/{sid}.jsonl"
    os.makedirs(tmp_path / "memory", exist_ok=True)
    with open(mem_path, "w", encoding="utf-8") as f:
        f.write('{"text":"x"}\n')
    assert await store.delete(sid) is True
    assert store.get(sid) is None
    assert not os.path.exists(mem_path)
    # 重复删除返回 False
    assert await store.delete(sid) is False


# ---- 文件持久化恢复 ----

def test_meta_persists_across_instances(tmp_path):
    store1 = _store(tmp_path)
    sid = store1.create(client_key="cid:device-a")
    store1.title_if_empty(sid, "cid:device-a", "持久化测试会话")
    store1.touch(sid, "cid:device-a")

    # 重启：新实例从 sessions.jsonl 恢复全部元数据
    store2 = _store(tmp_path)
    rec = store2.get(sid)
    assert rec is not None
    assert rec["title"] == "持久化测试会话"
    assert rec["client_key"] == "cid:device-a"
    assert rec["message_count"] == 1


@pytest.mark.asyncio
async def test_checkpoint_persists_across_instances(tmp_path):
    """SQLite 检查点落盘：新实例初始化后不再是内存 saver，且旧会话线程可恢复。"""
    from langgraph.checkpoint.memory import MemorySaver

    store1 = _store(tmp_path)
    assert isinstance(store1.checkpointer, MemorySaver)
    await store1.init_checkpointer()
    from langgraph.checkpoint.base import BaseCheckpointSaver

    assert isinstance(store1.checkpointer, BaseCheckpointSaver)
    assert not isinstance(store1.checkpointer, MemorySaver)
    assert os.path.exists(tmp_path / "checkpoints" / "checkpoints.sqlite")

    store2 = _store(tmp_path)
    await store2.init_checkpointer()
    assert not isinstance(store2.checkpointer, MemorySaver)


@pytest.mark.asyncio
async def test_delete_removes_checkpoint_thread(tmp_path):
    """删除会话联动清理 checkpoint 线程（SQLite saver 的 adelete_thread）。"""
    store = _store(tmp_path)
    await store.init_checkpointer()
    sid = store.create(client_key="cid:device-a")
    await store.delete(sid)
    # 线程不存在也不报错（adelete_thread 幂等），元数据已移除
    assert store.get(sid) is None


# ---- TTL 清理 ----

@pytest.mark.asyncio
async def test_gc_ttl_removes_expired(tmp_path):
    store = _store(tmp_path, ttl=7)
    fresh = store.create(client_key="cid:device-a")
    stale = store.create(client_key="cid:device-a")
    # 把 stale 的最后活跃时间改到 8 天前
    store.get(stale)["last_active"] = time.time() - 8 * 86400

    deleted = await store.gc_ttl()
    assert deleted == 1
    assert store.get(fresh) is not None
    assert store.get(stale) is None


@pytest.mark.asyncio
async def test_gc_ttl_disabled_when_ttl_zero(tmp_path):
    store = _store(tmp_path, ttl=0)
    sid = store.create(client_key="cid:device-a")
    store.get(sid)["last_active"] = time.time() - 365 * 86400
    assert await store.gc_ttl() == 0
    assert store.get(sid) is not None


# ---- 会话事件流（历史回放）----

@pytest.mark.asyncio
async def test_record_events_tees_and_persists(tmp_path):
    """record_events 边透传边落盘：消费方拿到的顺序不变，文件可被 load_events 读回。"""
    store = _store(tmp_path)
    sid = store.create(client_key="cid:device-a")

    async def fake_events():
        yield {"type": "meta", "session_id": sid, "mode": "react"}
        for i in range(5):
            yield {"type": "thinking", "delta": f"d{i}"}
        yield {"type": "done", "summary": "ok", "stats": {}}

    got = [ev async for ev in store.record_events(sid, fake_events())]
    assert got[0]["type"] == "meta"
    assert [ev["delta"] for ev in got if ev["type"] == "thinking"] == ["d0", "d1", "d2", "d3", "d4"]

    loaded = store.load_events(sid)
    assert [ev["type"] for ev in loaded] == [ev["type"] for ev in got]
    assert loaded[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_record_events_flushes_on_generator_exit(tmp_path):
    """客户端断连（GeneratorExit）：已产出的部分事件仍要落盘，保证历史尽量完整。"""
    store = _store(tmp_path)
    sid = store.create(client_key="cid:device-a")

    async def fake_events():
        yield {"type": "meta", "session_id": sid}
        for i in range(3):
            yield {"type": "message", "delta": f"m{i}"}
        raise GeneratorExit()

    gen = store.record_events(sid, fake_events())
    with pytest.raises(GeneratorExit):
        async for _ev in gen:
            pass

    loaded = store.load_events(sid)
    assert len(loaded) == 4  # meta + 3 条增量已落盘


@pytest.mark.asyncio
async def test_delete_removes_events_file(tmp_path):
    store = _store(tmp_path)
    sid = store.create(client_key="cid:device-a")

    async def fake_events():
        yield {"type": "meta", "session_id": sid}
        yield {"type": "done", "summary": "ok", "stats": {}}

    _ = [ev async for ev in store.record_events(sid, fake_events())]
    assert len(store.load_events(sid)) == 2
    await store.delete(sid)
    assert store.load_events(sid) == []
    assert not os.path.exists(f"{tmp_path / 'events'}/{sid}.jsonl")


# ---- API 层 ----

@pytest.fixture
def sessions_runtime(tmp_path):
    """注入独立 SessionStore 到运行时，测试后还原，避免污染其他用例。"""
    store = _store(tmp_path)
    chat.set_runtime(sessions=store)
    yield store
    chat._RUNTIME["sessions"] = None


def _headers(client="device-a"):
    return {"X-Client-Id": client}


def test_api_create_and_list(sessions_runtime, tmp_path):
    client = TestClient(app)
    resp = client.post("/api/sessions", headers=_headers("device-a"))
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    assert sid

    listed = client.get("/api/sessions", headers=_headers("device-a")).json()["sessions"]
    assert any(s["session_id"] == sid for s in listed)

    # 客户端隔离：device-b 看不到 device-a 的会话
    listed_b = client.get("/api/sessions", headers=_headers("device-b")).json()["sessions"]
    assert all(s["session_id"] != sid for s in listed_b)


def test_api_rename_and_permission(sessions_runtime, tmp_path):
    client = TestClient(app)
    sid = client.post("/api/sessions", headers=_headers("device-a")).json()["session_id"]

    # 他人改名 → 403
    resp = client.patch(
        f"/api/sessions/{sid}",
        json={"title": "篡改"},
        headers=_headers("device-b"),
    )
    assert resp.status_code == 403

    # 本人改名 → 200 且生效
    resp = client.patch(
        f"/api/sessions/{sid}",
        json={"title": "正式标题"},
        headers=_headers("device-a"),
    )
    assert resp.status_code == 200
    assert resp.json()["session"]["title"] == "正式标题"

    # 不存在的会话 → 404
    resp = client.patch("/api/sessions/not-exist", json={"title": "x"}, headers=_headers("device-a"))
    assert resp.status_code == 404


def test_api_delete_and_permission(sessions_runtime, tmp_path):
    client = TestClient(app)
    sid = client.post("/api/sessions", headers=_headers("device-a")).json()["session_id"]

    assert client.delete(f"/api/sessions/{sid}", headers=_headers("device-b")).status_code == 403
    assert client.delete(f"/api/sessions/{sid}", headers=_headers("device-a")).status_code == 200

    listed = client.get("/api/sessions", headers=_headers("device-a")).json()["sessions"]
    assert all(s["session_id"] != sid for s in listed)


def test_api_events_replay(sessions_runtime, tmp_path):
    """历史回放接口：返回会话已落盘的事件流；他人会话 403、不存在 404。"""
    store = sessions_runtime
    client = TestClient(app)
    sid = client.post("/api/sessions", headers=_headers("device-a")).json()["session_id"]

    async def fake_events():
        yield {"type": "meta", "session_id": sid, "mode": "react"}
        yield {"type": "message", "delta": "你好"}
        yield {"type": "done", "summary": "ok", "stats": {}}

    asyncio.run(_consume(store, sid, fake_events()))

    resp = client.get(f"/api/sessions/{sid}/events", headers=_headers("device-a"))
    assert resp.status_code == 200
    assert [e["type"] for e in resp.json()["events"]] == ["meta", "message", "done"]

    assert client.get(f"/api/sessions/{sid}/events", headers=_headers("device-b")).status_code == 403
    assert client.get("/api/sessions/not-exist/events", headers=_headers("device-a")).status_code == 404


async def _consume(store, sid, events):
    _ = [ev async for ev in store.record_events(sid, events)]
