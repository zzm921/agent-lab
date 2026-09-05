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


# ---- 模糊带 LLM 批量裁决（提取→匹配→合并三段式） ----

# 已有记忆与新提取事实相似度落在模糊带 [0.6, 0.92)：0.7741 / 0.7969 / 0.6542
_EXTRACT_DARK_PURPLE = (
    '[{"text": "用户喜欢深色主题，主色是紫色，强调色为金色", "type": "preference", '
    '"importance": 0.9}]'
)
_EXTRACT_SWITCH_BLUE = (
    '[{"text": "用户改用了蓝色主题", "type": "preference", "importance": 0.9}]'
)
_EXTRACT_BIRTHDAY = (
    '[{"text": "用户生日是十月十号", "type": "fact", "importance": 0.8}]'
)


@pytest.mark.asyncio
async def test_consolidate_ambiguous_llm_merge(embeddings, tmp_path, settings):
    """模糊带 + LLM 判 merge：补充/更新 → 合并进同一条，旧值入 history 归档。"""
    store = _store(embeddings, tmp_path)
    store.add("用户喜欢深色主题，主色是紫色", kind="preference", importance=0.8)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(
        script=[
            AIMessage(content=_EXTRACT_DARK_PURPLE),
            AIMessage(
                content=(
                    '[{"index": 0, "action": "merge", "reason": "补充强调色", '
                    '"text": "用户喜欢深色主题，主色是紫色，强调色为金色"}]'
                )
            ),
        ]
    )
    written = await maybe_consolidate(store, constant, _history(), llm, settings, "s1")
    assert len(written) == 1
    recs = store.list()
    assert len(recs) == 1  # 合并而非新增
    assert recs[0]["text"] == "用户喜欢深色主题，主色是紫色，强调色为金色"
    assert recs[0]["merge_count"] == 1
    assert len(recs[0]["history"]) == 1  # 旧值已归档
    assert recs[0]["history"][0]["text"] == "用户喜欢深色主题，主色是紫色"
    assert "merge" in [i["action"] for i in store.list_audit()]


@pytest.mark.asyncio
async def test_consolidate_ambiguous_llm_conflict(embeddings, tmp_path, settings):
    """模糊带 + LLM 判 conflict：用户改口 → 新表述作当前值、旧值归档不再召回。"""
    store = _store(embeddings, tmp_path)
    store.add("用户喜欢深色主题，主色是紫色", kind="preference", importance=0.8)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(
        script=[
            AIMessage(content=_EXTRACT_SWITCH_BLUE),
            AIMessage(
                content=(
                    '[{"index": 0, "action": "conflict", "reason": "用户改口", '
                    '"text": "用户改用了蓝色主题"}]'
                )
            ),
        ]
    )
    await maybe_consolidate(store, constant, _history(), llm, settings, "s1")
    recs = store.list()
    assert len(recs) == 1
    assert recs[0]["text"] == "用户改用了蓝色主题"
    assert recs[0]["history"][0]["text"] == "用户喜欢深色主题，主色是紫色"
    assert "conflict" in [i["action"] for i in store.list_audit()]


@pytest.mark.asyncio
async def test_consolidate_ambiguous_llm_add(embeddings, tmp_path, settings):
    """模糊带 + LLM 判 add：不同事实 → 另存一条（不覆盖旧生日）。"""
    store = _store(embeddings, tmp_path)
    store.add("用户生日是1995年8月20日", kind="fact", importance=0.8)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(
        script=[
            AIMessage(content=_EXTRACT_BIRTHDAY),
            AIMessage(
                content=(
                    '[{"index": 0, "action": "add", "reason": "不同事实", "text": ""}]'
                )
            ),
        ]
    )
    await maybe_consolidate(store, constant, _history(), llm, settings, "s1")
    texts = {r["text"] for r in store.list()}
    assert len(texts) == 2  # 旧生日保留，新生日另存
    assert "用户生日是1995年8月20日" in texts
    assert "用户生日是十月十号" in texts


@pytest.mark.asyncio
async def test_consolidate_ambiguous_judge_fail_conflict_fallback(embeddings, tmp_path, settings):
    """裁决失败（默认回答非法 JSON）→ 规则回退：含改口触发词按 conflict 合并。"""
    store = _store(embeddings, tmp_path)
    store.add("用户喜欢深色主题，主色是紫色", kind="preference", importance=0.8)
    constant = _constant(embeddings, tmp_path)
    # 只有提取响应；裁决轮用尽脚本 → 默认回答（非 JSON）→ 走规则回退
    llm = FakeChatModel(script=[AIMessage(content=_EXTRACT_SWITCH_BLUE)])
    await maybe_consolidate(store, constant, _history(), llm, settings, "s1")
    recs = store.list()
    assert len(recs) == 1
    assert recs[0]["text"] == "用户改用了蓝色主题"  # 改用 ∈ 冲突触发词 → 合并
    assert recs[0]["merge_count"] == 1


@pytest.mark.asyncio
async def test_consolidate_ambiguous_judge_fail_add_fallback(embeddings, tmp_path, settings):
    """裁决失败 → 规则回退：无改口触发词 → 保守新增（宁重不漏，两条并存）。"""
    store = _store(embeddings, tmp_path)
    store.add("用户喜欢深色主题，主色是紫色", kind="preference", importance=0.8)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(script=[AIMessage(content=_EXTRACT_DARK_PURPLE)])  # 无触发词
    await maybe_consolidate(store, constant, _history(), llm, settings, "s1")
    assert len(store.list()) == 2
