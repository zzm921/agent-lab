"""运行记录（telemetry）测试：sink 聚合、store 落盘/隔离/治理、resume 续写、LLM 埋点、runner 集成。"""
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.llm.fake_model import FakeChatModel
from app.llm.service import LoggedChatModel
from app.main import app
from app.telemetry.sink import ACTIVE_SINK
from app.telemetry.store import RunStore, set_run_store
from tests.conftest import ai_with_tool, collect_stream


@pytest.fixture
def store(tmp_path, monkeypatch):
    """开启 telemetry 并注入 tmp 目录的 RunStore（含单价，验证成本统计）。"""
    from app.config import settings

    monkeypatch.setattr(settings, "telemetry_enabled", True)
    s = RunStore(
        str(tmp_path),
        ttl_days=7,
        max_runs=10,
        price_input_per_1m=0.3,
        price_output_per_1m=0.6,
    )
    set_run_store(s)
    yield s
    set_run_store(None)


# ---- sink：增量合并 + 聚合统计 ----
@pytest.mark.asyncio
async def test_sink_merges_delta_and_stats(store):
    sink = store.new_run("s1", "cid:a", {"message": "你好"})
    sink.observe({"type": "meta", "session_id": "s1"})
    sink.observe({"type": "thinking", "delta": "先"})
    sink.observe({"type": "thinking", "delta": "思考"})
    sink.observe({"type": "message", "delta": "你"})
    sink.observe({"type": "message", "delta": "好"})
    sink.observe({"type": "tool_start", "tool": "calculator"})
    sink.observe({"type": "tool_end", "tool": "calculator", "success": False})
    sink.observe({"type": "tool_retry", "tool": "calculator"})
    sink.observe({"type": "retrieve", "hits": [{"x": 1}, {"x": 2}]})
    sink.observe({"type": "done", "summary": "完成", "stats": {"tool_calls": 1}})
    meta = sink.close()

    # 增量文本合并为完整事件，且顺序保持在首次出现位置
    texts = [ev["text"] for ev in sink.events if ev["type"] in ("thinking", "message")]
    assert "先思考" in texts
    assert "你好" in texts
    assert meta["status"] == "done"
    assert meta["summary"] == "完成"
    assert meta["stats"]["tool_calls"] == {"calculator": 1}
    assert meta["stats"]["tool_failures"] == {"calculator": 1}
    assert meta["stats"]["retries"] == 1
    assert meta["stats"]["rag_retrieves"] == 1
    assert meta["stats"]["rag_hits"] == 2
    assert meta["message"] == "你好"
    # 落盘 + 客户端隔离：cid:a 可见，cid:b 不可见
    assert store.list("cid:a")[0]["run_id"] == meta["run_id"]
    assert store.list("cid:b") == []


def test_llm_call_records_tokens_and_cost(store):
    sink = store.new_run("s1", "cid:a", {})
    sink.record_llm(
        {
            "scenario": "chat",
            "model": "m",
            "latency_ms": 10,
            "success": True,
            "tokens": {"input": 100, "output": 200, "total": 300},
        }
    )
    sink.record_llm({"scenario": "chat", "model": "m", "latency_ms": 5, "success": False, "error": "boom"})
    meta = sink.close()
    assert meta["stats"]["llm_calls"] == 2
    assert meta["stats"]["tokens"]["total"] == 300
    # 100 input * 0.3 + 200 output * 0.6（元/百万 token）
    assert meta["stats"]["cost_yuan"] == round(100 / 1e6 * 0.3 + 200 / 1e6 * 0.6, 4)


# ---- store：TTL / 数量上限治理 ----
def test_store_prunes_max_runs(store):
    for i in range(12):
        s = store.new_run(f"s{i}", "cid:a", {})
        s.close(status="done")
    assert store.count() == 10  # 上限 10，删最旧
    assert len(store.list("cid:a")) == 10


def test_store_ttl_purges(tmp_path):
    st = RunStore(str(tmp_path), ttl_days=1, max_runs=10)
    s = st.new_run("s1", "cid:a", {})
    s.close(status="done")
    path = Path(st._dir) / f"{s.run_id}.jsonl"
    old = time.time() - 2 * 86400
    os.utime(path, (old, old))
    assert st.list("cid:a") == []  # list 触发 _prune


# ---- pending / resume 续写同一 run ----
def test_pending_and_resume_same_run(store):
    sink = store.new_run("s1", "cid:a", {"message": "q"})
    sink.observe({"type": "tool_start", "tool": "calc"})
    sink.observe({"type": "approval_request", "approval_id": "a1", "tool_calls": []})
    sink.close(status="pending")
    assert store.list("cid:a")[0]["status"] == "pending"

    resumed = store.resume_run("s1", "cid:a")
    assert resumed is not None
    assert resumed.run_id == sink.run_id
    resumed.observe({"type": "tool_end", "tool": "calc", "success": True})
    resumed.observe({"type": "done", "summary": "ok", "stats": {}})
    resumed.close()

    # 只应有一条记录，事件完整、状态收敛为 done
    runs = store.list("cid:a")
    assert len(runs) == 1
    assert runs[0]["status"] == "done"
    doc = store.get(resumed.run_id, "cid:a")
    types = [e["type"] for e in doc["events"]]
    assert types == ["tool_start", "approval_request", "tool_end", "done"]


def test_resume_run_none_without_pending(store):
    assert store.resume_run("s1", "cid:a") is None


# ---- LLM 埋点：成功/失败都记录 ----
@pytest.mark.asyncio
async def test_logged_model_records_success_and_failure(store):
    model = LoggedChatModel(inner=FakeChatModel(), scenario="chat")
    sink = store.new_run("s1", "cid:a", {})
    token = ACTIVE_SINK.set(sink)
    try:
        await model._agenerate([HumanMessage(content="hi")])
        calls = [e for e in sink.events if e["type"] == "llm_call"]
        assert len(calls) == 1
        assert calls[0]["success"] is True
        assert calls[0]["scenario"] == "chat"

        class Boom(FakeChatModel):
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                raise RuntimeError("boom")

        bad = LoggedChatModel(inner=Boom(), scenario="chat")
        with pytest.raises(Exception):
            await bad._agenerate([HumanMessage(content="hi")])
        calls = [e for e in sink.events if e["type"] == "llm_call"]
        assert len(calls) == 2
        assert calls[-1]["success"] is False
        assert "boom" in calls[-1]["error"]
    finally:
        ACTIVE_SINK.reset(token)


# ---- token 计算：从 usage_metadata 提取（修复 llm_output 为空导致记账全 0） ----
def test_usage_tokens_reads_usage_metadata():
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    from app.llm.service import _usage_tokens

    # 项目标准路径：非流式结果的消息带 usage_metadata（input/output/total_tokens）
    msg = AIMessage(content="ok")
    msg.usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    result = ChatResult(generations=[ChatGeneration(message=msg)])
    assert _usage_tokens(result) == {"input": 10, "output": 20, "total": 30}

    # 流式末块：ChatGenerationChunk 携带 usage_metadata
    from langchain_core.messages import AIMessageChunk

    chunk = AIMessageChunk(content="x", usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
    assert _usage_tokens(ChatGenerationChunk(message=chunk)) == {"input": 1, "output": 2, "total": 3}

    # 兼容回退：response_metadata.usage（prompt/completion/total_tokens）
    fallback = AIMessage(content="f")
    fallback.response_metadata = {"usage": {"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15}}
    assert _usage_tokens(fallback) == {"input": 7, "output": 8, "total": 15}

    # 无用量：返回 None（调用仍记录，只是不带 tokens 字段）
    assert _usage_tokens(ChatResult(generations=[ChatGeneration(message=AIMessage(content="n"))])) is None
    assert _usage_tokens(None) is None


@pytest.mark.asyncio
async def test_logged_model_records_usage_metadata_tokens(store):
    """DashScope 实际路径（usage 写在 usage_metadata）：_agenerate 应据此记账 token 与成本。"""
    from langchain_core.outputs import ChatGeneration, ChatResult

    class WithUsage(FakeChatModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            msg = AIMessage(content="hi")
            msg.usage_metadata = {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}
            return ChatResult(generations=[ChatGeneration(message=msg)])

    model = LoggedChatModel(inner=WithUsage(), scenario="chat")
    sink = store.new_run("s1", "cid:a", {})
    token = ACTIVE_SINK.set(sink)
    try:
        await model._agenerate([HumanMessage(content="hi")])
        calls = [e for e in sink.events if e["type"] == "llm_call"]
        assert calls[0]["tokens"] == {"input": 100, "output": 200, "total": 300}
        meta = sink.close()
        assert meta["stats"]["tokens"]["total"] == 300
        # 100 input * 0.3 + 200 output * 0.6（元/百万 token）
        assert meta["stats"]["cost_yuan"] == round(100 / 1e6 * 0.3 + 200 / 1e6 * 0.6, 4)
    finally:
        ACTIVE_SINK.reset(token)


# ---- runner 集成：stream 落盘完整 run ----
@pytest.mark.asyncio
async def test_runner_stream_records_run(settings, registry, sessions, runner, store):
    events = await collect_stream(runner, enabled=["calculator"])
    assert events[-1]["type"] == "done"
    runs = store.list("default")  # collect_stream 未传 client_key → "default"
    assert len(runs) == 1
    assert runs[0]["status"] == "done"
    assert runs[0]["message"] == "测试任务"
    doc = store.get(runs[0]["run_id"], "default")
    assert len(doc["events"]) > 0


@pytest.mark.asyncio
async def test_runner_resume_continues_same_run(settings, registry, sessions, runner, store):
    """审批暂停 → resume 续写同一 run_id（一轮对话一条完整记录）。"""
    runner.llm.script = [
        ai_with_tool("需要计算", args={"expression": "1+1"}),
        AIMessage(content="最终结果 2"),
    ]
    # memory_enabled=False：避免 L2 主动召回 selector 消费 Fake 脚本导致首个工具调用被吞
    events = await collect_stream(runner, approval_policy="always", enabled=["calculator"], memory_enabled=False)
    request = next(e for e in events if e["type"] == "approval_request")
    assert "done" not in [e["type"] for e in events]
    assert store.list("default")[0]["status"] == "pending"

    resumed = []
    async for ev in runner.resume(request["approval_id"], "approve", {}, client_key="default"):
        resumed.append(ev)
    assert resumed[-1]["type"] == "done"

    runs = store.list("default")
    assert len(runs) == 1  # 仍是一条
    assert runs[0]["status"] == "done"
    types = [e["type"] for e in store.get(runs[0]["run_id"], "default")["events"]]
    assert "approval_request" in types
    assert "done" in types


# ---- API：列表 + 详情 + 客户端隔离 ----
def test_telemetry_api_list_and_detail(store):
    """列表 + 详情 + 客户端隔离（按 X-Client-Id 判定 client_key）。"""
    sink = store.new_run("s1", "cid:device-a", {"message": "你好", "mode": "react"})
    sink.observe({"type": "meta", "session_id": "s1"})
    sink.observe({"type": "message", "delta": "你好"})
    sink.observe({"type": "done", "summary": "ok", "stats": {}})
    sink.close()

    client = TestClient(app)
    headers = {"X-Client-Id": "device-a"}
    resp = client.get("/api/telemetry/runs", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert len(data["runs"]) >= 1
    assert data["runs"][0]["message"] == "你好"
    run_id = data["runs"][0]["run_id"]

    detail = client.get(f"/api/telemetry/runs/{run_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["events"][-1]["type"] == "done"

    # 其他客户端不可见
    other = client.get(f"/api/telemetry/runs/{run_id}", headers={"X-Client-Id": "device-b"})
    assert other.status_code == 404


def test_telemetry_api_disabled_returns_empty():
    from app.api import chat as chat_api

    chat_api.set_runtime(runner=None)
    client = TestClient(app)
    resp = client.get("/api/telemetry/runs")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
