"""轮末自动提取巩固测试：Fake LLM 返回事实 → 记忆库增长；低重要度过滤；异常不阻断。"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.llm.fake_model import FakeChatModel
from app.memory.consolidate import maybe_consolidate
from app.memory.long_memory import LongMemoryStore


def _store(embeddings, tmp_path):
    return LongMemoryStore("s1", embeddings, str(tmp_path / "m.jsonl"))


def _constant(embeddings, tmp_path):
    return LongMemoryStore("_global:default", embeddings, str(tmp_path / "c.jsonl"))


def _history():
    return [
        HumanMessage(content="我喜欢深色主题，主色用紫色"),
        AIMessage(content="好的，已记录你的配色偏好。"),
    ]


@pytest.mark.asyncio
async def test_consolidate_extracts_and_writes(embeddings, tmp_path, settings):
    store = _store(embeddings, tmp_path)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(
        script=[
            AIMessage(
                content=(
                    '[{"text": "用户喜欢深色主题，主色是紫色", "type": "preference", '
                    '"importance": 0.9}, {"text": "临时状态不用记", "type": "fact", '
                    '"importance": 0.2}]'
                )
            )
        ]
    )
    written = await maybe_consolidate(store, constant, _history(), llm, settings, "s1")
    # 只写入 importance ≥ 0.5 的一条；scope 缺失默认会话库
    assert len(written) == 1
    assert written[0]["text"] == "用户喜欢深色主题，主色是紫色"
    assert written[0]["kind"] == "preference"
    assert len(store) == 1
    assert len(constant) == 0


@pytest.mark.asyncio
async def test_consolidate_scope_global_writes_constant(embeddings, tmp_path, settings):
    """scope=global 的长期偏好/约束写入常驻库（跨会话生效），不进会话库。"""
    store = _store(embeddings, tmp_path)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(
        script=[
            AIMessage(
                content=(
                    '[{"text": "以后所有项目都用 nodejs 写", "type": "preference", '
                    '"importance": 0.9, "scope": "global"}, '
                    '{"text": "本轮对比了两个方案", "type": "episodic", '
                    '"importance": 0.6, "scope": "session"}]'
                )
            )
        ]
    )
    written = await maybe_consolidate(store, constant, _history(), llm, settings, "s1")
    assert len(written) == 2
    # 长期偏好 → 常驻库
    assert constant.list()[0]["text"] == "以后所有项目都用 nodejs 写"
    assert store.list()[0]["text"] == "本轮对比了两个方案"


@pytest.mark.asyncio
async def test_consolidate_disabled(embeddings, tmp_path, settings):
    settings.memory_consolidate_enabled = False
    store = _store(embeddings, tmp_path)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(script=[AIMessage(content='[{"text": "x", "type": "fact", "importance": 0.9}]')])
    written = await maybe_consolidate(store, constant, _history(), llm, settings, "s1")
    assert written == []
    assert len(store) == 0


@pytest.mark.asyncio
async def test_consolidate_llm_error_swallowed(embeddings, tmp_path, settings):
    """LLM 异常被吞掉：返回空列表，不影响主链路。"""
    store = _store(embeddings, tmp_path)
    constant = _constant(embeddings, tmp_path)

    class BoomLLM:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("llm down")

    written = await maybe_consolidate(store, constant, _history(), BoomLLM(), settings, "s1")
    assert written == []
    assert len(store) == 0
