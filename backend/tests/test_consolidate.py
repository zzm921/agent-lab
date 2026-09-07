"""轮末自动提取巩固测试：Fake LLM 返回事实 → 记忆库增长；低重要度过滤；异常不阻断。"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.llm.fake_model import FakeChatModel
from app.memory.consolidate import dream_consolidate, dream_tidy, maybe_consolidate
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


# ---- 记忆梦游（手动显式巩固）：返回结构化中间结果报告，展示每步 ----

@pytest.mark.asyncio
async def test_dream_returns_structured_report(embeddings, tmp_path, settings):
    """梦游报告包含提取/过滤/匹配/落库明细与前后数量对比。"""
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
    report = await dream_consolidate(store, constant, _history(), llm, settings, "s1")
    # 提取阶段：2 条候选全部进入报告；过滤 1 条低重要度
    assert report["summary"]["extracted"] == 2
    assert len(report["extracted"]) == 2
    assert report["summary"]["dropped"] == 1
    assert len(report["filtered"]) == 1
    assert "阈值" in report["filtered"][0]["reason"]
    # 匹配阶段：1 条无匹配（zone=none）
    assert len(report["matches"]) == 1
    assert report["matches"][0]["zone"] == "none"
    # 落库明细：写入 1 条，动作 add
    assert report["summary"]["written"] == 1
    assert report["summary"]["added"] == 1
    assert report["written"][0]["action"] == "add"
    assert report["written"][0]["text"] == "用户喜欢深色主题，主色是紫色"
    # 前后数量对比
    assert report["summary"]["before"] == {"session": 0, "global": 0}
    assert report["summary"]["after"] == {"session": 1, "global": 0}
    assert len(report["steps"]) >= 2


@pytest.mark.asyncio
async def test_dream_ambiguous_judged_reported(embeddings, tmp_path, settings):
    """模糊带裁决在报告中可见：judgments 记录动作与理由，written 落 merge。"""
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
    report = await dream_consolidate(store, constant, _history(), llm, settings, "s1")
    assert len(report["judgments"]) == 1
    assert report["judgments"][0]["action"] == "merge"
    assert report["judgments"][0]["reason"] == "补充强调色"
    assert len(report["matches"]) == 1
    assert report["matches"][0]["zone"] == "ambiguous"
    assert report["matches"][0]["sim"] >= 0.6
    assert report["written"][0]["action"] == "merge"
    assert report["summary"]["merged"] == 1
    # 合并而非新增：库内仍 1 条
    assert len(store.list()) == 1
    assert report["summary"]["after"] == {"session": 1, "global": 0}


@pytest.mark.asyncio
async def test_dream_llm_error_returns_empty_report(embeddings, tmp_path, settings):
    """LLM 提取异常被吞掉：返回零值报告，不抛错。"""
    store = _store(embeddings, tmp_path)
    constant = _constant(embeddings, tmp_path)

    class BoomLLM:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("llm down")

    report = await dream_consolidate(store, constant, _history(), BoomLLM(), settings, "s1")
    assert report["summary"]["extracted"] == 0
    assert report["summary"]["written"] == 0
    assert len(store) == 0


# ---- 记忆梦游 · 整理现有记忆（对齐 Claude Dreaming）----

@pytest.mark.asyncio
async def test_dream_tidy_merges_and_replaces(embeddings, tmp_path, settings):
    """tidy：LLM 给出 merge/replace/pattern 建议 → 直接落库，报告含动作与前后对比。"""
    store = _store(embeddings, tmp_path)
    constant = _constant(embeddings, tmp_path)
    # 用 add_judged(add) 强制新建三条（避免 FakeEmbeddings 下 add() 自动语义合并干扰测试前提）
    r1 = store.add_judged("用户喜欢深色主题", kind="preference", importance=0.8, decision="add")
    r2 = store.add_judged("项目使用 TypeScript 编写", kind="preference", importance=0.7, decision="add")
    r3 = store.add_judged("项目旧方案是使用 Flask", kind="fact", importance=0.6, decision="add")
    llm = FakeChatModel(
        script=[
            AIMessage(
                content=(
                    '[{"action": "merge", "scope": "session", "ids": ["%(a)s", "%(b)s"], '
                    '"text": "用户喜欢深色主题，主色是紫色", "kind": "preference", '
                    '"importance": 0.85, "reason": "两条主题偏好重复"}, '
                    '{"action": "replace", "scope": "session", "ids": ["%(c)s"], '
                    '"text": "项目后端改用 FastAPI", "kind": "fact", '
                    '"importance": 0.6, "reason": "旧方案已被替换"}]'
                    % {"a": r1["id"], "b": r2["id"], "c": r3["id"]}
                )
            )
        ]
    )
    report = await dream_tidy(store, constant, llm, settings, "s1")
    # 落库执行：merge 1、replace 1
    assert report["summary"]["merge"] == 1
    assert report["summary"]["replace"] == 1
    assert report["summary"]["pattern"] == 0
    assert len(report["actions"]) == 2
    # merge 后 r2 被删除、r1 保留合并表述；replace 后 r3 换成新表述
    items = store.list()
    assert len(items) == 2
    assert any(it["id"] == r1["id"] and "主色是紫色" in it["text"] for it in items)
    assert all(it["id"] != r2["id"] for it in items)
    assert any(it["id"] == r3["id"] and "FastAPI" in it["text"] for it in items)
    # 前后对比：4 条 → 2 条
    assert report["summary"]["before"] == {"session": 3, "global": 0}
    assert report["summary"]["after"] == {"session": 2, "global": 0}
    assert len(report["steps"]) >= 2


@pytest.mark.asyncio
async def test_dream_tidy_empty_store(embeddings, tmp_path, settings):
    """tidy：记忆库为空 → 不调 LLM，报告提示无需整理。"""
    store = _store(embeddings, tmp_path)
    constant = _constant(embeddings, tmp_path)

    class ShouldNotCall:
        async def ainvoke(self, *a, **k):
            raise AssertionError("空库不应调用 LLM")

    report = await dream_tidy(store, constant, ShouldNotCall(), settings, "s1")
    assert report["summary"]["merge"] == 0
    assert report["actions"] == []
    assert report["summary"]["before"] == {"session": 0, "global": 0}


@pytest.mark.asyncio
async def test_dream_tidy_scope_guard(embeddings, tmp_path, settings):
    """tidy：跨 scope 建议被忽略（session 库建议不能作用于 global 条目）。"""
    store = _store(embeddings, tmp_path)
    constant = _constant(embeddings, tmp_path)
    r1 = store.add("用户喜欢深色主题", kind="preference", importance=0.8)
    r2 = constant.add("全局偏好：使用 TypeScript", kind="preference", importance=0.8)
    llm = FakeChatModel(
        script=[
            AIMessage(
                content=(
                    '[{"action": "merge", "scope": "session", "ids": ["%(a)s", "%(b)s"], '
                    '"text": "用户偏好深色与 TypeScript", "kind": "preference", '
                    '"importance": 0.8, "reason": "跨库合并"}]'
                    % {"a": r1["id"], "b": r2["id"]}
                )
            )
        ]
    )
    report = await dream_tidy(store, constant, llm, settings, "s1")
    # 建议因 scope 不匹配被忽略：无执行动作、两库各保持 1 条
    assert report["summary"]["merge"] == 0
    assert report["actions"] == []
    assert len(store) == 1
    assert len(constant) == 1


@pytest.mark.asyncio
async def test_dream_tidy_pattern_adds_new(embeddings, tmp_path, settings):
    """tidy：pattern 动作新增一条模式记忆，原条目保留。"""
    store = _store(embeddings, tmp_path)
    constant = _constant(embeddings, tmp_path)
    store.add("用户在 React 项目用 hooks", kind="procedural", importance=0.6)
    store.add("用户在 Vue 项目用组合式 API", kind="procedural", importance=0.6)
    llm = FakeChatModel(
        script=[
            AIMessage(
                content=(
                    '[{"action": "pattern", "scope": "session", "ids": [], '
                    '"text": "用户偏好现代框架新写法（hooks/组合式 API）", "kind": "preference", '
                    '"importance": 0.7, "reason": "跨条目归纳出的共性偏好"}]'
                )
            )
        ]
    )
    report = await dream_tidy(store, constant, llm, settings, "s1")
    assert report["summary"]["pattern"] == 1
    assert len(report["actions"]) == 1
    # 原 2 条保留 + 新模式 1 条 = 3 条
    assert len(store) == 3
