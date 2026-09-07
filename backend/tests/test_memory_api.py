"""长期记忆管理 API 测试：GET 列表 / POST 写入 / DELETE 删除 / 记忆梦游（注入 Fake 运行时）。"""
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph

from app.api import chat
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.llm.fake_model import FakeChatModel
from app.main import app
from app.memory.session_store import SessionStore


@pytest.fixture
def mem_registry(settings, embeddings, tmp_path):
    sessions = SessionStore(memory_dir=str(tmp_path))  # 落盘到临时目录（隔离 + 支持审计）
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), None, embeddings)
    chat.set_runtime(sessions=sessions, registry=registry)
    yield sessions
    chat.set_runtime(sessions=None, registry=None, runner=None)


class FakeDreamRunner:
    """dream 端点仅用到 runner._scenario_llm 与 runner.settings：注入脚本化 Fake 模型。"""

    def __init__(self, llm, settings):
        self.llm = llm
        self.settings = settings

    def _scenario_llm(self, scenario):
        return self.llm


async def _seed_session(sessions, session_id: str) -> None:
    """用最小 StateGraph 往 checkpointer 种入一条多轮对话，供 dream 端点读取。"""
    graph = StateGraph(MessagesState)

    async def _node(state):
        return {"messages": [AIMessage(content="好的，已记住。")]}

    graph.add_node("seed", _node)
    graph.set_entry_point("seed")
    graph.set_finish_point("seed")
    app_graph = graph.compile(checkpointer=sessions.checkpointer)
    await app_graph.ainvoke(
        {"messages": [HumanMessage(content="我喜欢深色主题，主色是紫色")]},
        {"configurable": {"thread_id": session_id}},
    )


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


def test_memory_api_audit(settings, embeddings, mem_registry):
    """审计端点：写入/删除产生流水，按时间倒序返回。"""
    client = TestClient(app)
    client.post("/api/memory", json={"text": "用户喜欢深色主题", "kind": "preference", "importance": 0.9})
    client.post("/api/memory", json={"text": "用户生日是 1995-08-20", "kind": "fact", "importance": 0.7})

    resp = client.get("/api/memory/audit")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 2
    assert items[0]["action"] == "add"
    assert items[0]["scope"] == "session"
    assert {i["kind"] for i in items} == {"preference", "fact"}

    # scope 过滤：global 无记录（上面写的都是 session）
    resp = client.get("/api/memory/audit?scope=global")
    assert resp.json()["items"] == []


def test_memory_api_dream(settings, embeddings, mem_registry):
    """记忆梦游：无会话记录 → 400；有会话 → 返回结构化报告且真实落库。"""
    client = TestClient(app)
    # 无会话记录 → 400
    resp = client.post("/api/memory/dream?session_id=no_such_session")
    assert resp.status_code == 400

    # 种入一条对话并注入脚本化提取模型
    import asyncio

    asyncio.run(_seed_session(mem_registry, "s_dream"))
    llm = FakeChatModel(
        script=[
            AIMessage(
                content=(
                    '[{"text": "用户喜欢深色主题，主色是紫色", "type": "preference", '
                    '"importance": 0.9}]'
                )
            )
        ]
    )
    chat.set_runtime(runner=FakeDreamRunner(llm, settings))

    resp = client.post("/api/memory/dream?session_id=s_dream")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "s_dream"
    # 提取 1 条、写入 1 条、无过滤
    assert data["summary"]["extracted"] == 1
    assert data["summary"]["written"] == 1
    assert data["summary"]["dropped"] == 0
    assert data["written"][0]["action"] == "add"
    assert data["written"][0]["text"] == "用户喜欢深色主题，主色是紫色"
    # 会话库 +1
    assert data["summary"]["before"]["session"] == 0
    assert data["summary"]["after"]["session"] == 1

    # 真实落库：再次 GET 可见
    resp = client.get("/api/memory?scope=session&session_id=s_dream")
    assert any("主色是紫色" in it["text"] for it in resp.json()["items"])


def test_memory_api_dream_tidy(settings, embeddings, mem_registry):
    """记忆梦游 · 整理模式：读现有记忆 → LLM 建议直接落库 → 前后对比。"""
    client = TestClient(app)
    # 先写两条记忆（避免 add 自动合并）
    store = mem_registry.long_memory("s_tidy", embeddings)
    r1 = store.add_judged("用户喜欢深色主题", kind="preference", importance=0.8, decision="add")
    r2 = store.add_judged("项目使用 TypeScript 编写", kind="preference", importance=0.7, decision="add")
    llm = FakeChatModel(
        script=[
            AIMessage(
                content=(
                    '[{"action": "merge", "scope": "session", "ids": ["%s", "%s"], '
                    '"text": "用户喜欢深色主题（已归档）", "kind": "preference", '
                    '"importance": 0.85, "reason": "测试合并"}]' % (r1["id"], r2["id"])
                )
            )
        ]
    )
    chat.set_runtime(runner=FakeDreamRunner(llm, settings))

    resp = client.post("/api/memory/dream?session_id=s_tidy&mode=tidy")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "tidy"
    assert data["summary"]["before"] == {"session": 2, "global": 0}
    assert len(data["actions"]) == 1
    assert data["actions"][0]["action"] == "merge"
    assert data["actions"][0]["executed"] is True
