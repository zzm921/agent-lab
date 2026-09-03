"""上下文管理与压缩管线测试：snip-compact / micro-compact / auto-compact / 大文件落盘。

全部使用本地 Fake 模型，确定性、不联网、不依赖 Key。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.context_manage import (
    ContextManager,
    _drop_exact_duplicates,
    maybe_offload,
    micro_compact,
    snip_compact,
)
from app.agents.modes.react import build_react_agent
from app.agents.runner import AgentRunner
from app.llm.fake_model import FakeChatModel
from app.tools.big_output import big_output
from tests.conftest import make_settings


def _tool_ai(cid: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "1"}, "id": cid, "type": "tool_call"}],
    )


class FailingModel(FakeChatModel):
    """模拟 LLM 调用失败的模型（auto-compact 熔断测试）。"""

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("llm unavailable")


# ---------- snip-compact ----------

def test_snip_below_threshold_noop():
    msgs = [HumanMessage(content=f"u{i}") for i in range(5)]
    out, metrics = snip_compact(msgs, max_messages=10, keep_head=2, keep_tail=3)
    assert out is msgs  # 未触发：零改动、返回原列表
    assert metrics is None


def test_snip_keep_head_and_tail():
    msgs = [HumanMessage(content=f"u{i}") for i in range(12)]
    out, metrics = snip_compact(msgs, max_messages=6, keep_head=2, keep_tail=3)
    assert metrics == {"original": 12, "kept": 5, "dropped": 7}
    assert [m.content for m in out[:2]] == ["u0", "u1"]  # 开头保留
    assert [m.content for m in out[-3:]] == ["u9", "u10", "u11"]  # 结尾保留
    assert "u4" not in [m.content for m in out]  # 中间裁剪


def test_snip_boundary_protection():
    # 构造裁剪边界（start）落在 ToolMessage 上：必须向前并入其 tool_use AIMessage，保证配对完整
    msgs = [SystemMessage(content="sys"), HumanMessage(content="u0")]
    for i in range(5):
        cid = f"c{i + 1}"
        msgs.append(_tool_ai(cid))
        msgs.append(ToolMessage(content=f"结果{i + 1}", tool_call_id=cid))
    # 12 条：start = max(2, 12-3) = 9 → msgs[9] 是 ToolMessage(c4)，需并入 msgs[8] AI(c4)
    out, metrics = snip_compact(msgs, max_messages=6, keep_head=2, keep_tail=3)
    assert metrics["dropped"] > 0
    # 尾部 AI(tool_use) 与其 ToolMessage 配对不拆散
    tail_ai = out[-4]
    tail_tool = out[-3]
    assert tail_ai.type == "ai" and tail_tool.type == "tool"
    assert tail_tool.tool_call_id == tail_ai.tool_calls[0]["id"]


def test_snip_light_cleanup():
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="u0"),
        HumanMessage(content="u0"),  # 完全重复 → 去重
        AIMessage(content="a1"),
        AIMessage(content="a1"),  # 连续同内容助手消息 → 合并
        ToolMessage(content="x" * 300, tool_call_id="c1"),  # 超长但 snip 不截断（职责在 micro）
    ]
    out, metrics = snip_compact(msgs, max_messages=3, keep_head=2, keep_tail=3)
    assert metrics == {"original": 6, "kept": 4, "dropped": 2}
    assert len([m for m in out if m.type == "human" and m.content == "u0"]) == 1
    assert len([m for m in out if m.type == "ai"]) == 1
    tool = next(m for m in out if m.type == "tool")
    assert tool.content == "x" * 300  # 工具输出原文保留（不重复压缩：截断交给 micro_compact）


def test_drop_duplicates_keeps_interleaved_history():
    # 跨轮次相同内容（用户重复提问、模型重复回答）是合法对话，不得删除，保持 user/assistant 交替
    msgs = [
        HumanMessage(content="你好"),
        AIMessage(content="hello"),
        HumanMessage(content="哈哈哈"),
        AIMessage(content="hello"),  # 与更早一条同内容，但不相邻 → 必须保留
        HumanMessage(content="xxx"),
    ]
    out = _drop_exact_duplicates(msgs)
    assert len(out) == len(msgs)  # 全部保留
    # 交替结构不被破坏：任意两条相邻消息 type 不同
    for a, b in zip(out, out[1:]):
        assert a.type != b.type


def test_drop_duplicates_same_tool_output_keeps_pairing():
    # 两条 ToolMessage 内容相同但 tool_call_id 不同 → 按调用 id 判同，不得误删，配对完整
    msgs = [
        _tool_ai("c1"), ToolMessage(content="same", tool_call_id="c1"),
        _tool_ai("c2"), ToolMessage(content="same", tool_call_id="c2"),
    ]
    out = _drop_exact_duplicates(msgs)
    assert len(out) == 4
    ai_ids = [tc["id"] for m in out if m.type == "ai" for tc in m.tool_calls]
    tool_ids = [m.tool_call_id for m in out if m.type == "tool"]
    assert tool_ids == ai_ids  # 每个 AI tool_call 都有对应 ToolMessage


# ---------- micro-compact ----------

def test_micro_truncates_old_long_results_keeps_recent():
    msgs = [
        _tool_ai("c1"), ToolMessage(content="x" * 500 + "尾1", tool_call_id="c1"),
        _tool_ai("c2"), ToolMessage(content="y" * 500 + "尾2", tool_call_id="c2"),
        _tool_ai("c3"), ToolMessage(content="新结果3", tool_call_id="c3"),
    ]
    out, metrics = micro_compact(msgs, keep_recent=1, truncate_chars=300)
    assert metrics == {"original": 3, "truncated": 2, "kept": 1}
    c1 = next(m for m in out if m.type == "tool" and m.tool_call_id == "c1")
    c2 = next(m for m in out if m.type == "tool" and m.tool_call_id == "c2")
    c3 = next(m for m in out if m.type == "tool" and m.tool_call_id == "c3")
    assert c1.content.startswith("x" * 300) and "（工具结果已压缩" in c1.content  # 截断保头部
    assert c2.content.startswith("y" * 300) and "（工具结果已压缩" in c2.content
    assert "尾1" not in c1.content and "尾2" not in c2.content  # 尾部丢弃
    assert c3.content == "新结果3"  # 最近 1 条保留原文
    assert [m.tool_call_id for m in out if m.type == "tool"] == ["c1", "c2", "c3"]  # 配对保持


def test_micro_skips_offload_pointer_and_short_text():
    # 落盘指针含路径（受保护）与短文本均不被截断：避免同一内容被重复压缩
    ptr = "[工具输出已落盘] 完整输出共 5000 字符，已保存至 data/offload/xxx.txt。开头预览：……"
    msgs = [
        _tool_ai("c1"), ToolMessage(content=ptr, tool_call_id="c1"),
        _tool_ai("c2"), ToolMessage(content="短文本", tool_call_id="c2"),
        _tool_ai("c3"), ToolMessage(content="新结果3", tool_call_id="c3"),
    ]
    out, metrics = micro_compact(msgs, keep_recent=1, truncate_chars=10)  # 极小阈值也不截断
    assert metrics is None  # 无可截断项 → 不触发
    assert out is msgs


# ---------- 大文件落盘 ----------

def test_offload_writes_file_and_pointer(tmp_path, settings):
    settings = make_settings(
        context_offload_enabled=True,
        context_offload_threshold=10,
        context_offload_dir=str(tmp_path),
        context_offload_preview=5,
        context_offload_max_per_session=50,
    )
    text = "x" * 100
    ptr, info = maybe_offload(text, session_id="session-123456", tool_name="run_command", settings=settings)
    assert info == {"chars": 100, "file": info["file"]}
    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == text
    assert ptr.startswith("[工具输出已落盘]") and "已保存至" in ptr and "开头预览" in ptr


def test_offload_below_threshold_noop(tmp_path, settings):
    settings = make_settings(
        context_offload_enabled=True,
        context_offload_threshold=1000,
        context_offload_dir=str(tmp_path),
    )
    text = "short"
    ptr, info = maybe_offload(text, session_id="s", tool_name="t", settings=settings)
    assert ptr == text
    assert info is None
    assert list(tmp_path.iterdir()) == []


def test_offload_max_per_session_prunes(tmp_path, settings):
    settings = make_settings(
        context_offload_enabled=True,
        context_offload_threshold=1,
        context_offload_dir=str(tmp_path),
        context_offload_max_per_session=2,
    )
    for i in range(5):
        maybe_offload(f"内容{i}" * 10, session_id="sess-aaa", tool_name="t", settings=settings)
    files = list(tmp_path.glob("sess-aaa_*.txt"))
    assert len(files) <= 2


# ---------- auto-compact（LLM 摘要） ----------

async def test_auto_compact_summarizes_and_memoizes(settings):
    settings = make_settings(
        context_auto_compact_enabled=True,
        context_auto_compact_threshold=5,
        context_auto_compact_keep_recent=2,
    )
    cm = ContextManager(settings)
    llm = FakeChatModel()
    # 第二次应复用缓存摘要，不会消费第二条脚本
    llm.script = [AIMessage(content='{"goal":"目标"}'), AIMessage(content="不应被消费")]
    messages = [HumanMessage(content=f"第{i}轮", id=f"m{i}") for i in range(6)]
    msgs1, events1 = await cm.build(messages, llm=llm, session_id="s1")
    assert events1 and events1[0]["kind"] == "auto_compact"
    assert len(msgs1) == 3  # 摘要 + 最近 2 条
    msgs2, events2 = await cm.build(messages, llm=llm, session_id="s1")
    assert len(llm.script) == 1  # 复用缓存，未重复调用 LLM
    assert msgs2[0].content == msgs1[0].content


async def test_auto_compact_circuit_breaker(settings):
    settings = make_settings(
        context_auto_compact_enabled=True,
        context_auto_compact_threshold=5,
        context_auto_compact_keep_recent=2,
    )
    cm = ContextManager(settings)
    llm = FailingModel()
    messages = [HumanMessage(content=f"第{i}轮", id=f"m{i}") for i in range(6)]
    for _ in range(3):
        msgs, events = await cm.build(messages, llm=llm, session_id="s1")
        assert events == []  # 每次失败跳过，不阻断流程
    assert "s1" in cm._disabled  # 连续 3 次失败 → 熔断禁用
    msgs, events = await cm.build(messages, llm=llm, session_id="s1")  # 禁用后不再尝试
    assert msgs == messages and events == []


# ---------- runner 集成 ----------

async def _seed_and_compress(settings, registry, sessions, history, message="本轮问题"):
    """构建 react 图 → 种子历史 → 执行 _make_inputs，返回 (inputs, context 事件)。"""
    runner = AgentRunner(settings, FakeChatModel(), registry, sessions)
    events: list = []
    emit = events.append
    graph = build_react_agent(runner._scenario_llm("chat"), [], emit, settings, sessions.checkpointer, runner.harness)
    config = runner._config("s1", "never", "standard")
    await graph.aupdate_state(config, {"messages": list(history)})
    inputs = await runner._make_inputs(graph, config, message, "standard", emit=emit)
    return inputs, events


async def test_runner_snip_compact_bounds_messages(settings, registry, sessions):
    settings = make_settings(
        context_mgmt_enabled=True,
        context_snip_enabled=True,
        context_snip_max_messages=6,
        context_snip_keep_head=2,
        context_snip_keep_tail=3,
        context_micro_keep_recent=6,
    )
    history = [
        SystemMessage(content="sys"),
        *[m for i in range(4) for m in (HumanMessage(content=f"u{i}"), AIMessage(content=f"a{i}"))],
    ]
    # 9 条历史 > 阈值 6 → snip 触发：保留 head(2) + tail(3) = 5
    inputs, events = await _seed_and_compress(settings, registry, sessions, history)
    ctx = [e for e in events if e["type"] == "context"]
    assert any(e["kind"] == "snip_compact" for e in ctx)
    snip = next(e for e in ctx if e["kind"] == "snip_compact")
    assert snip["metrics"]["kept"] <= 6
    # 模型收到消息数有界：压缩后历史 + 本轮 HumanMessage
    assert len(inputs["messages"]) <= 7


async def test_runner_context_mgmt_disabled_passthrough(settings, registry, sessions):
    settings = make_settings(
        context_mgmt_enabled=False,
        context_snip_max_messages=6,
        context_snip_keep_head=2,
        context_snip_keep_tail=3,
    )
    history = [
        SystemMessage(content="sys"),
        *[m for i in range(4) for m in (HumanMessage(content=f"u{i}"), AIMessage(content=f"a{i}"))],
    ]
    inputs, events = await _seed_and_compress(settings, registry, sessions, history)
    assert not any(e["type"] == "context" for e in events)
    assert len(inputs["messages"]) == len(history) + 1  # 原样透传 + 本轮消息


# ---------- 「每轮压缩」演示（keep_rounds 保留最近 N 轮） ----------

async def test_runner_keep_rounds_triggers_snip(settings, registry, sessions):
    """keep_rounds=2：保留量 = head(3) + 2*2 = 7 条，9 条历史超量即触发 snip，保留最近 2 轮。"""
    settings = make_settings(
        context_mgmt_enabled=True,
        context_snip_enabled=True,
        context_micro_enabled=True,
        context_micro_truncate_chars=50,
    )
    history = [
        SystemMessage(content="sys"),
        *[m for i in range(4) for m in (HumanMessage(content=f"u{i}", id=f"h{i}"), AIMessage(content=f"a{i}", id=f"a{i}"))],
    ]  # 9 条
    runner = AgentRunner(settings, FakeChatModel(), registry, sessions)
    events: list = []
    emit = events.append
    graph = build_react_agent(runner._scenario_llm("chat"), [], emit, settings, sessions.checkpointer, runner.harness)
    config = runner._config("s1", "never", "standard")
    await graph.aupdate_state(config, {"messages": list(history)})
    inputs = await runner._make_inputs(graph, config, "本轮问题", "standard", emit=emit, keep_rounds=2)
    ctx = [e for e in events if e["type"] == "context"]
    snip = next(e for e in ctx if e["kind"] == "snip_compact")
    assert snip["keep_rounds"] == 2  # 事件带出保留轮数
    assert snip["metrics"]["kept"] == 7  # head(3) + 最近 4 条（2 轮）
    assert snip["metrics"]["dropped"] == 2  # 中间 1 轮（第 1 轮）被裁
    # 掐头去尾：开头 3 条（sys+初始轮）与最近 2 轮（u2,a2,u3,a3）保留，中间 u1,a1 被裁剪
    kept_texts = [m.content for m in inputs["messages"]]
    assert "u3" in kept_texts and "a3" in kept_texts and "u2" in kept_texts and "a2" in kept_texts
    assert "u1" not in kept_texts and "a1" not in kept_texts


async def test_runner_keep_rounds_zero_uses_default_threshold(settings, registry, sessions):
    """keep_rounds=0：走系统默认阈值（50），9 条历史不触发，保持原行为。"""
    settings = make_settings(
        context_mgmt_enabled=True,
        context_snip_enabled=True,
        context_snip_max_messages=50,
        context_micro_enabled=True,
    )
    history = [
        SystemMessage(content="sys"),
        *[m for i in range(4) for m in (HumanMessage(content=f"u{i}", id=f"h{i}"), AIMessage(content=f"a{i}", id=f"a{i}"))],
    ]
    runner = AgentRunner(settings, FakeChatModel(), registry, sessions)
    events: list = []
    emit = events.append
    graph = build_react_agent(runner._scenario_llm("chat"), [], emit, settings, sessions.checkpointer, runner.harness)
    config = runner._config("s1", "never", "standard")
    await graph.aupdate_state(config, {"messages": list(history)})
    inputs = await runner._make_inputs(graph, config, "本轮问题", "standard", emit=emit, keep_rounds=0)
    assert not any(e["type"] == "context" for e in events)
    assert len(inputs["messages"]) == len(history) + 1  # 原样透传 + 本轮消息


# ---------- 大输出落盘演示工具 ----------

def test_big_output_exceeds_offload_threshold():
    """big_output 固定输出 5000+ 字符，超默认落盘阈值（3000）→ 天然触发 maybe_offload。"""
    out = big_output.invoke({"topic": "演示"})
    assert len(out) > 3000
    assert "演示" in out[:50]

