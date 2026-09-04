"""常驻记忆注入测试（T3）：首轮 system 含记忆块 + memory_constant 事件；低重要度过滤。"""
import pytest

from tests.conftest import collect_stream


def _seed_constant(sessions, embeddings):
    constant = sessions.constant_memory(embeddings)
    constant.add("用户喜欢深色主题，主色是紫色", kind="preference", importance=0.9)
    constant.add("临时低价值记忆", kind="fact", importance=0.2)
    return constant


def test_constant_memory_block_filtering(settings, registry, sessions, runner, embeddings):
    """常驻块只取 importance ≥ memory_constant_min_importance 的 top-k。"""
    _seed_constant(sessions, embeddings)
    block, count = runner._constant_memory_block()
    assert count == 1
    assert block is not None
    assert "用户记忆（来自历史会话，仅供参考）" in block
    assert "[preference]" in block
    assert "深色主题" in block
    assert "临时低价值记忆" not in block


def test_constant_memory_disabled(settings, registry, sessions, runner, embeddings):
    _seed_constant(sessions, embeddings)
    settings.memory_constant_enabled = False
    block, count = runner._constant_memory_block()
    assert block is None
    assert count == 0


def test_constant_memory_isolated_by_client_key(settings, registry, sessions, runner, embeddings):
    """常驻记忆按客户端隔离：device-a 写入的常驻记忆，device-b 不可见、不注入。"""
    a = sessions.constant_memory(embeddings, "cid:device-a")
    b = sessions.constant_memory(embeddings, "cid:device-b")
    a.add("A 的私有偏好", kind="preference", importance=0.9)
    assert len(a) == 1
    assert len(b) == 0  # B 的常驻库独立，不受 A 写入影响

    block_a, count_a = runner._constant_memory_block(client_key="cid:device-a")
    block_b, count_b = runner._constant_memory_block(client_key="cid:device-b")
    assert count_a == 1
    assert count_b == 0
    assert "A 的私有偏好" in block_a
    assert block_b is None


@pytest.mark.asyncio
async def test_constant_memory_event_in_stream(settings, registry, sessions, runner, embeddings):
    """首轮流式运行产出 memory_constant 事件（注入条数），供前端卡片。"""
    _seed_constant(sessions, embeddings)
    events = await collect_stream(runner, enabled=["calculator"])
    types = [e["type"] for e in events]
    assert "memory_constant" in types
    ev = next(e for e in events if e["type"] == "memory_constant")
    assert ev["count"] == 1


@pytest.mark.asyncio
async def test_constant_memory_injected_once(settings, registry, sessions, runner, embeddings):
    """同一会话首轮注入 system；第二轮（已有历史）不再重复注入。"""
    from app.agents.tools_builder import build_tools

    _seed_constant(sessions, embeddings)
    tools = build_tools(registry, ["calculator"], "s1")
    graph = runner._build_graph("react", tools, lambda e: None)
    config = runner._config("s1", "never", "standard")

    inputs1 = await runner._make_inputs(graph, config, "你好", "standard")
    system1 = inputs1["messages"][0]
    assert "用户记忆（来自历史会话，仅供参考）" in system1.content

    # 模拟第一轮已写入 checkpointer：第二轮 _make_inputs 读到历史 → 不重建 system
    await graph.ainvoke(inputs1, config)
    inputs2 = await runner._make_inputs(graph, config, "再说一遍", "standard")
    # 第二轮沿用历史中的 system 消息（不重复注入、条数不增长）
    system2 = inputs2["messages"][0]
    assert system2.content.count("用户记忆（来自历史会话") == 1
