"""LLM 流式 token 记账回归测试（修复 chat astream 记 0t 的问题）。

覆盖两个修复点：
- service.py 聚合式记账：流式 usage 附着在「非末块」（不同模型/供应商位置不同）时，
  不再只取最后一块；取「最后一次出现的有效用量」，并忽略全 0 占位；
- dashscope_chat.py 兜底：放宽 total_tokens>0 判定（个别模型只回 input_tokens），
  且当「已见 usage 但最后产出块未携带」时补发独立尾块，保证上层能记账。
全部离线确定性：stub 模型 / mock Generation.call，不依赖 Key 与网络。
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from app.llm import dashscope_chat as ds_mod
from app.llm.dashscope_chat import DashScopeChatModel
from app.llm.service import LoggedChatModel

_USAGE = {"input_tokens": 278, "output_tokens": 25, "total_tokens": 303}


class _ChunkStub:
    """模拟 inner：stream/astream 产出消息块序列（usage 附着在中间块，末块无 usage）。"""

    model_name = "stub"

    def stream(self, messages, **kwargs):
        yield AIMessageChunk(content="a", usage_metadata=_USAGE)
        yield AIMessageChunk(content="b")  # 末块无 usage → 旧实现记 0t

    async def astream(self, messages, **kwargs):
        for chunk in self.stream(messages, **kwargs):
            yield chunk


def _make_recorder(records: list[dict]):
    def recorder(self, method, latency_ms, success, tokens=None, error=""):
        records.append({"method": method, "success": success, "tokens": tokens, "error": error})

    return recorder


# ---------------------------------------------------------------------------
# 方案 A：service.py 聚合式记账
# ---------------------------------------------------------------------------
def test_stream_aggregates_usage_from_middle_chunk(monkeypatch):
    """usage 在中间块、末块无 usage → 记账中间块的值（不再记 0t）。"""
    llm = LoggedChatModel(inner=_ChunkStub(), scenario="chat")
    records: list[dict] = []
    monkeypatch.setattr(LoggedChatModel, "_record", _make_recorder(records))

    chunks = list(llm._stream([HumanMessage(content="hi")]))
    assert [c.message.content for c in chunks] == ["a", "b"]
    assert records[0]["success"] is True
    assert records[0]["tokens"] == {"input": 278, "output": 25, "total": 303}


@pytest.mark.asyncio
async def test_astream_aggregates_usage_from_middle_chunk(monkeypatch):
    """异步路径（chat 场景真实路径）同修复：usage 在中间块也能记账。"""
    llm = LoggedChatModel(inner=_ChunkStub(), scenario="chat")
    records: list[dict] = []
    monkeypatch.setattr(LoggedChatModel, "_record", _make_recorder(records))

    chunks = [c async for c in llm._astream([HumanMessage(content="hi")])]
    assert [c.message.content for c in chunks] == ["a", "b"]
    assert records[0]["success"] is True
    assert records[0]["tokens"] == {"input": 278, "output": 25, "total": 303}


def test_stream_ignores_all_zero_usage(monkeypatch):
    """全 0 的 usage 占位块不覆盖有效记账（宁缺毋滥）。"""

    class _ZeroThenValid(_ChunkStub):
        def stream(self, messages, **kwargs):
            yield AIMessageChunk(
                content="z",
                usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            yield AIMessageChunk(content="v", usage_metadata=_USAGE)

    llm = LoggedChatModel(inner=_ZeroThenValid(), scenario="chat")
    records: list[dict] = []
    monkeypatch.setattr(LoggedChatModel, "_record", _make_recorder(records))

    list(llm._stream([HumanMessage(content="hi")]))
    assert records[0]["tokens"] == {"input": 278, "output": 25, "total": 303}


# ---------------------------------------------------------------------------
# 方案 B：dashscope_chat.py 兜底（放宽判定 + 补发独立尾块）
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, choices, usage=None):
        self.status_code = 200
        self.output = {"choices": choices} if choices is not None else None
        self.usage = usage


def _fake_call_factory(blocks: list[dict]):
    def fake_call(**payload):
        return [_FakeResp(b.get("choices"), usage=b.get("usage")) for b in blocks]

    return fake_call


def _stream_chunks(monkeypatch, blocks: list[dict]):
    """mock Generation.call 后迭代 DashScopeChatModel._stream，返回全部 chunk。"""
    monkeypatch.setattr(ds_mod.Generation, "call", staticmethod(_fake_call_factory(blocks)))
    model = DashScopeChatModel(model_name="qwen3.5-flash", api_key="test-key")
    return list(model._stream([HumanMessage(content="hi")]))


def test_dashscope_usage_on_middle_content_block(monkeypatch):
    """usage 附着在中间内容块、后续还有内容块 → 补发独立尾块，末块必带 usage。"""
    chunks = _stream_chunks(
        monkeypatch,
        [
            {"choices": [{"message": {"content": "a", "reasoning_content": ""}}], "usage": _USAGE},
            {"choices": [{"message": {"content": "b", "reasoning_content": ""}}]},
        ],
    )
    assert [c.message.content for c in chunks] == ["a", "b", ""]  # 尾块为空内容，仅携带用量
    last = chunks[-1]
    assert last.message.usage_metadata == {"input_tokens": 278, "output_tokens": 25, "total_tokens": 303}


def test_dashscope_usage_without_total_tokens(monkeypatch):
    """usage 块只有 input_tokens（无 total_tokens）→ 不再被丢弃，正常记账。"""
    chunks = _stream_chunks(
        monkeypatch,
        [
            {
                "choices": [{"message": {"content": "a", "reasoning_content": ""}}],
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
            {"choices": [{"message": {"content": "b", "reasoning_content": ""}}]},
        ],
    )
    last = chunks[-1]
    assert last.message.usage_metadata == {"input_tokens": 100, "output_tokens": 0, "total_tokens": 0}


def test_dashscope_standalone_usage_tail_block(monkeypatch):
    """独立尾块（无内容）携带 usage → 保持原有行为，末块即 usage 块。"""
    chunks = _stream_chunks(
        monkeypatch,
        [
            {"choices": [{"message": {"content": "a", "reasoning_content": ""}}]},
            {"choices": None, "usage": _USAGE},  # 独立尾块
        ],
    )
    assert [c.message.content for c in chunks] == ["a", ""]
    assert chunks[-1].message.usage_metadata == {"input_tokens": 278, "output_tokens": 25, "total_tokens": 303}


def test_dashscope_no_usage_no_tail_block(monkeypatch):
    """流式完全无 usage 块 → 不补发尾块，行为与现状一致（不引入多余空块）。"""
    chunks = _stream_chunks(
        monkeypatch,
        [{"choices": [{"message": {"content": "a", "reasoning_content": ""}}]}],
    )
    assert [c.message.content for c in chunks] == ["a"]
    assert getattr(chunks[-1].message, "usage_metadata", None) is None
