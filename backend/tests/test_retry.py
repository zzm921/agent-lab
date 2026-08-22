"""两层重试机制测试：工具层透明重试 + Agent 层重试上限 + 错误分类/格式化/退避。

覆盖：
- retry.py 单元：瞬时错误分类、状态码判定、退避延迟边界、结构化错误文案；
- make_tools_node 集成：工具层透明重试（模型无感知）、重试耗尽返回结构化错误、
  非瞬时错误不直接重试、Agent 层上限「改用其它工具」提示；
- harness：同工具连续失败计数与上限判定、成功后清零。
"""
import httpx
import pytest
from langchain_core.messages import AIMessage

from app.agents.harness import AgentHarness
from app.core.errors import RetryableToolError
from app.tools import retry as retry_mod
from app.tools.retry import backoff_delay, exp_backoff_value, format_tool_error, invoke_with_retry, is_retryable_exception, is_retryable_status
from app.tools.runner import make_tools_node
from tests.conftest import make_settings, ai_with_tool


# ---------- 单元：错误分类 ----------

def test_is_retryable_exception_marks_transient():
    assert is_retryable_exception(RetryableToolError("超时"))
    assert is_retryable_exception(TimeoutError("read timeout"))
    assert is_retryable_exception(ConnectionResetError("reset"))
    assert is_retryable_exception(ConnectionError("refused"))
    assert is_retryable_exception(httpx.ConnectTimeout("connect timeout"))
    assert is_retryable_exception(httpx.ConnectError("no route"))
    assert is_retryable_exception(httpx.HTTPStatusError("503", request=httpx.Request("GET", "http://x"), response=httpx.Response(503)))


def test_is_retryable_exception_marks_permanent():
    assert not is_retryable_exception(ValueError("参数错误"))
    assert not is_retryable_exception(KeyError("missing"))
    assert not is_retryable_exception(FileNotFoundError("no file"))  # OSError 但确定性错误
    assert not is_retryable_exception(httpx.HTTPStatusError("400", request=httpx.Request("GET", "http://x"), response=httpx.Response(400)))


def test_is_retryable_status():
    assert is_retryable_status(429)
    assert is_retryable_status(500)
    assert is_retryable_status(502)
    assert is_retryable_status(503)
    assert not is_retryable_status(200)
    assert not is_retryable_status(400)
    assert not is_retryable_status(404)
    assert not is_retryable_status(None)


# ---------- 单元：退避延迟 ----------

def test_backoff_delay_within_bounds():
    base, cap = 1.0, 4.0
    for attempt in (1, 2, 3, 4):
        d = backoff_delay(attempt, base, cap)
        assert 0.5 <= d <= cap  # 抖动下限 base*0.5，上限封顶 cap
    # 封顶：attempt 很大时仍不超过 cap
    assert backoff_delay(10, base, cap) <= cap


def test_exp_backoff_value_exponential_curve():
    """纯指数退避值：base * 2^(attempt-1)，封顶 cap（不含抖动，供前端画退避曲线）。"""
    base, cap = 1.0, 10.0
    assert [exp_backoff_value(i, base, cap) for i in (1, 2, 3, 4, 5)] == [1.0, 2.0, 4.0, 8.0, 10.0]
    # 封顶后不再增长
    assert exp_backoff_value(10, base, cap) == 10.0
    # cap 小于 base 时以 base 为下限
    assert exp_backoff_value(1, 0.5, 0.1) == 0.5
    # 抖动后的实际值介于纯指数值的 0.5~1.0 倍之间
    assert 0.5 * exp_backoff_value(2, base, cap) <= backoff_delay(2, base, cap) <= exp_backoff_value(2, base, cap)


# ---------- 单元：结构化错误文案 ----------

def test_format_tool_error_retryable_exhausted():
    msg = format_tool_error("web_search", RetryableToolError("连接超时"), retried=2)
    assert "工具 web_search 执行失败" in msg
    assert "错误类型：瞬时错误" in msg
    assert "连接超时" in msg
    assert "已用相同参数自动重试 2 次" in msg
    assert "建议" in msg


def test_format_tool_error_permanent():
    msg = format_tool_error("web_search", ValueError("query 格式错误"), retried=0)
    assert "错误类型：参数或策略错误" in msg
    assert "query 格式错误" in msg
    assert "修正参数" in msg


# ---------- 单元：invoke_with_retry ----------

async def test_invoke_with_retry_succeeds_after_transient_failures():
    calls = []

    async def run():
        calls.append(1)
        if len(calls) < 3:
            raise RetryableToolError("模拟瞬时超时")
        return "最终成功"

    events = []
    result, success, error, retries = await invoke_with_retry(
        run, "flaky", make_settings(tool_retry_base_delay=0.001, tool_retry_max_delay=0.002), events.append
    )
    assert success is True
    assert result == "最终成功"
    assert error is None
    assert retries == 2  # 1 次原始 + 2 次直接重试
    assert len(calls) == 3
    assert len(events) == 2  # 两次 tool_retry 事件
    assert events[0]["type"] == "tool_retry"
    assert events[0]["attempt"] == 1
    assert events[0]["max"] == 3
    assert events[0]["tool"] == "flaky"
    # 事件含实际抖动睡眠 delay 与纯指数退避 base_delay（供前端画退避曲线）
    assert events[0]["base_delay"] == 0.001  # attempt=1：base * 2^0
    assert events[1]["base_delay"] == 0.002  # attempt=2：base * 2^1
    assert events[0]["delay"] >= 0.001 * 0.5  # 抖动下限
    assert events[1]["delay"] >= 0.002 * 0.5


async def test_invoke_with_retry_exhausted():
    async def run():
        raise RetryableToolError("持续超时")

    events = []
    settings = make_settings(tool_retry_max=3, tool_retry_base_delay=0.001, tool_retry_max_delay=0.002)
    result, success, error, retries = await invoke_with_retry(run, "flaky", settings, events.append)
    assert success is False
    assert result is None
    assert isinstance(error, RetryableToolError)
    assert retries == 3  # 第一次调用不算重试：原始 1 次 + 失败后自动重试 3 次 = 共 4 次尝试
    assert len(events) == 3


async def test_invoke_with_retry_skips_permanent():
    async def run():
        raise ValueError("参数错误")

    events = []
    result, success, error, retries = await invoke_with_retry(run, "flaky", make_settings(), events.append)
    assert success is False
    assert isinstance(error, ValueError)
    assert retries == 0
    assert events == []  # 非瞬时错误不重试、不发 tool_retry


# ---------- 集成：make_tools_node（plan_execute 工具节点路径） ----------

class _FlakyTool:
    """工具层重试场景：前 N 次抛瞬时错误，之后成功。"""

    name = "flaky"

    def __init__(self, fail_before_ok: int):
        self.fail_before_ok = fail_before_ok
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        if self.calls <= self.fail_before_ok:
            raise RetryableToolError("模拟瞬时网络超时")
        return "最终成功"


class _BrokenTool:
    """Agent 层上限场景：持续抛确定性错误。"""

    name = "broken"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        raise ValueError("业务逻辑错误：余额不足")


async def _run_node(tools, session_id="s_retry", cid="c1", name="flaky"):
    settings = make_settings(tool_retry_max=3, tool_retry_base_delay=0.001, tool_retry_max_delay=0.002)
    harness = AgentHarness(settings)
    events = []
    node = make_tools_node(tools, events.append, harness=harness)
    state = {"messages": [ai_with_tool("执行", name=name, args={}, cid=cid)]}
    config = {"configurable": {"thread_id": session_id, "approval_policy": "never"}}
    out = await node(state, config)
    return out, events, harness


async def test_tool_layer_retry_transparent():
    """瞬时错误：工具层直接重试，最终成功，模型无感知（只有一条成功 ToolMessage）。"""
    tool = _FlakyTool(fail_before_ok=2)
    out, events, _ = await _run_node([tool])
    retry_events = [e for e in events if e["type"] == "tool_retry"]
    assert len(retry_events) == 2
    assert retry_events[0]["attempt"] == 1 and retry_events[1]["attempt"] == 2
    assert tool.calls == 3  # 原始 1 + 重试 2
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["success"] is True
    assert end["result"] == "最终成功"
    # 透明：给模型的结果只有一条成功消息，不含任何失败痕迹
    assert len(out["messages"]) == 1
    assert out["messages"][0].content == "最终成功"
    assert out["step_failed"] is False


async def test_tool_layer_retry_exhausted_returns_structured_error():
    """重试耗尽：返回结构化错误（含错误类型/详情/建议）给模型。"""
    tool = _FlakyTool(fail_before_ok=99)
    out, events, _ = await _run_node([tool])
    assert len([e for e in events if e["type"] == "tool_retry"]) == 3  # 失败后自动重试 3 次（共 4 次尝试）后耗尽
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["success"] is False
    msg = out["messages"][0].content
    assert "错误类型：瞬时错误" in msg
    assert "已用相同参数自动重试 3 次" in msg
    assert out["step_failed"] is True


async def test_permanent_error_no_direct_retry():
    """确定性错误（参数/业务逻辑）：不直接重试，结构化错误直接给模型。"""
    tool = _BrokenTool()
    out, events, _ = await _run_node([tool], name="broken")
    assert not any(e["type"] == "tool_retry" for e in events)
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["success"] is False
    msg = out["messages"][0].content
    assert "错误类型：参数或策略错误" in msg
    assert "余额不足" in msg
    assert tool.calls == 1  # 仅执行一次


async def test_agent_layer_retry_cap_gives_up():
    """Agent 层上限：同一工具连续失败达到 agent_retry_max 后，直接提示模型改用其它工具（不执行）。"""
    settings = make_settings(tool_retry_max=3, tool_retry_base_delay=0.001, tool_retry_max_delay=0.002, agent_retry_max=3)
    harness = AgentHarness(settings)
    events = []
    node = make_tools_node([_BrokenTool()], events.append, harness=harness)
    session_id = "s_cap"
    config = {"configurable": {"thread_id": session_id, "approval_policy": "never"}}
    # 前 3 次连续失败（换参数/换调用，均被确定性错误拒绝）
    for i in range(3):
        await node({"messages": [ai_with_marker(i)]}, config)
    # 第 4 次：已达上限，直接提示改用其它工具；工具不再执行
    last_start = len([e for e in events if e["type"] == "tool_start"])
    out = await node({"messages": [ai_with_marker(3)]}, config)
    ends = [e for e in events if e["type"] == "tool_end"]
    assert "请停止使用该工具" in ends[-1]["result"]
    # 每次执行都会发 tool_start（含第 4 次的短路提示），确认第 4 次仍发了提示事件
    assert len([e for e in events if e["type"] == "tool_start"]) == last_start + 1
    assert out["messages"][0].content == ends[-1]["result"]


def ai_with_marker(n: int):
    """构造带 tool_calls 的 AI 消息（调用 broken 工具，cid 各不相同）。"""
    return AIMessage(
        content=f"第{n}次调用",
        tool_calls=[{"name": "broken", "args": {"n": n}, "id": f"call_{n}", "type": "tool_call"}],
    )


# ---------- 故障注入类型目录与分类 ----------

def test_fault_catalog_classification():
    modes = AgentHarness.available_fault_modes()
    # 瞬时错误 → 工具层直接重试
    for m in ("timeout", "conn_reset", "dns", "http_429", "http_500", "http_502", "http_503"):
        assert modes[m] == "retryable", m
    # 参数/业务错误 → 返回给模型思考后重试
    for m in ("error", "business", "http_400", "http_401", "http_403", "http_404"):
        assert modes[m] == "permanent", m


def test_fault_spec_classification():
    harness = AgentHarness(make_settings())
    harness.set_fault("calc", "http_500")
    spec = harness.fault_spec("calc")
    assert spec["retryable"] is True
    assert "故障注入" in spec["message"]
    harness.set_fault("calc", "business")
    assert harness.fault_spec("calc")["retryable"] is False
    harness.set_fault("calc", "off")
    assert harness.fault_spec("calc") is None


def test_fault_unknown_type_rejected():
    harness = AgentHarness(make_settings())
    with pytest.raises(ValueError):
        harness.set_fault("calc", "bogus")


# ---------- harness：连续失败计数与上限 ----------

def test_harness_agent_retry_cap_and_reset(settings):
    harness = AgentHarness(make_settings(agent_retry_max=3))
    sid = "s_harness"
    assert harness.agent_retry_limit() == 3
    assert harness.tool_exhausted(sid, "calc") is False
    harness.record_tool_failure(sid, "calc", {"a": 1})
    harness.record_tool_failure(sid, "calc", {"a": 2})
    assert harness.tool_exhausted(sid, "calc") is False  # 2 < 3
    harness.record_tool_failure(sid, "calc", {"a": 3})
    assert harness.tool_exhausted(sid, "calc") is True  # 3 次达上限
    harness.record_tool_success(sid, "calc", {"a": 4})
    assert harness.tool_exhausted(sid, "calc") is False  # 成功后清零
