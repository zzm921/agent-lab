"""L2 主动语义召回测试：selector 触发判断 / 写指令跳过 / 合并召回去重 / 已见去重 / 预算封顶 / 容错。"""
import pytest
from langchain_core.messages import AIMessage

from app.agents.runner import AgentRunner
from app.llm.fake_model import FakeChatModel
from app.memory.long_memory import LongMemoryStore
from app.memory.proactive import is_write_intent, maybe_recall
from tests.conftest import collect_stream


def _store(embeddings, tmp_path, ns="s1"):
    return LongMemoryStore(ns, embeddings, str(tmp_path / f"{ns}.jsonl"))


def _constant(embeddings, tmp_path):
    return LongMemoryStore("_global:dev-a", embeddings, str(tmp_path / "c.jsonl"))


def _events():
    out = []

    def emit(ev):
        out.append(ev)

    return out, emit


@pytest.mark.asyncio
async def test_proactive_selector_skip(embeddings, tmp_path):
    """selector 判定 need=false：跳过召回，返回 (None, [], True)，事件带 need=false。"""
    store = _store(embeddings, tmp_path)
    store.add("用户喜欢深色主题", kind="preference", importance=0.9)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(script=[AIMessage(content='{"need": false, "reason": "通用技术问题，无需背景"}')])
    events, emit = _events()
    block, hits, skipped = await maybe_recall(
        llm, store, constant, "React 和 Vue 有什么区别？",
        top_k=3, threshold=0.0, max_chars=400, injected_ids=set(), emit=emit,
    )
    assert skipped is True
    assert block is None and hits == []
    assert events[0]["type"] == "memory_read"
    assert events[0]["need"] is False
    assert events[0]["reason"] == "通用技术问题，无需背景"


@pytest.mark.parametrize(
    "query",
    [
        "记住我的生日是 十月十号",
        "帮我记住，以后所有项目都用 nodejs",
        "请记住我喜欢深色主题",
        "记一下我的邮箱是 a@b.com",
        "忘掉我之前说的生日",
        "保存这条：用户技术栈是 Python",
    ],
)
def test_is_write_intent_true(query):
    assert is_write_intent(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "你还记得上次让我记住的偏好吗？",
        "我的生日是哪天？",
        "我之前说过什么配色偏好？",
        "React 和 Vue 有什么区别？",
        "记住的东西有哪些？",
    ],
)
def test_is_write_intent_false(query):
    """读指令 / 通用问题不应被写指令规则误伤。"""
    assert is_write_intent(query) is False


@pytest.mark.asyncio
async def test_proactive_personal_query_force_recall(embeddings, tmp_path):
    """个人化归属查询（我的主管是谁）确定性召回：跳过 selector，直接召回身份记忆。"""
    store = _store(embeddings, tmp_path)
    store.add("用户名为张三，任职于研发部", kind="fact", importance=0.9)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel()  # script 为空：护栏直接判定，selector 不应被调用
    events, emit = _events()
    block, hits, skipped = await maybe_recall(
        llm, store, constant, "我的主管是谁？",
        top_k=3, threshold=0.0, max_chars=400, injected_ids=set(), emit=emit,
    )
    assert skipped is False
    assert block is not None and "张三" in block
    assert events[0]["type"] == "memory_read"
    assert events[0]["need"] is True
    assert events[0]["reason"] == "个人化归属查询，需召回用户身份记忆"
    assert any("张三" in h["text"] for h in events[0]["hits"])


@pytest.mark.asyncio
async def test_proactive_write_intent_skip(embeddings, tmp_path):
    """写指令（记住 X）确定性跳过：不发 selector、不召回，事件 need=false，reason 说明写指令。"""
    store = _store(embeddings, tmp_path)
    store.add("用户偏好深色主题", kind="preference", importance=0.9)
    constant = _constant(embeddings, tmp_path)
    constant.add("用户要求所有项目用 nodejs", kind="preference", importance=0.8)
    llm = FakeChatModel()  # script 为空：若被调用会返回默认回答，但不该被调用
    events, emit = _events()
    block, hits, skipped = await maybe_recall(
        llm, store, constant, "记住我的生日是 十月十号",
        top_k=3, threshold=0.0, max_chars=400, injected_ids=set(), emit=emit,
    )
    assert skipped is True
    assert block is None and hits == []
    assert events[0]["type"] == "memory_read"
    assert events[0]["need"] is False
    assert events[0]["source"] == "proactive"
    assert "写指令" in events[0]["reason"]


@pytest.mark.asyncio
async def test_proactive_read_not_misjudged_as_write(embeddings, tmp_path):
    """读指令（你还记得…吗）不被写指令规则误伤：照常召回相关记忆。"""
    store = _store(embeddings, tmp_path)
    store.add("用户偏好深色主题", kind="preference", importance=0.9)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(script=[AIMessage(content='{"need": true, "reason": ""}')])
    block, hits, skipped = await maybe_recall(
        llm, store, constant, "你还记得上次让我记住的偏好吗？",
        top_k=3, threshold=0.0, max_chars=400, injected_ids=set(),
    )
    assert skipped is False
    assert block is not None and "深色主题" in block


@pytest.mark.asyncio
async def test_proactive_recall_merges_both_stores(embeddings, tmp_path):
    """need=true：会话库 + 常驻库合并召回，注入块含两库记忆，命中即记录已见 id。"""
    store = _store(embeddings, tmp_path)
    store.add("用户偏好深色主题", kind="preference", importance=0.9)
    constant = _constant(embeddings, tmp_path)
    constant.add("用户要求所有项目用 nodejs 开发", kind="preference", importance=0.8)
    llm = FakeChatModel(script=[AIMessage(content='{"need": true, "reason": "用户个人偏好相关"}')])
    injected = set()
    events, emit = _events()
    block, hits, skipped = await maybe_recall(
        llm, store, constant, "我的偏好是什么？",
        top_k=3, threshold=0.0, max_chars=800, injected_ids=injected, emit=emit,
    )
    assert skipped is False
    assert block is not None
    assert "深色主题" in block and "nodejs" in block
    assert len(hits) == 2
    assert len(injected) == 2  # 命中 id 已记入会话级已见集合
    assert events[0]["source"] == "proactive"


@pytest.mark.asyncio
async def test_proactive_dedup_injected_across_calls(embeddings, tmp_path):
    """已见去重：同一会话已注入过的记忆，第二轮同 query 不重复注入。"""
    store = _store(embeddings, tmp_path)
    store.add("用户偏好深色主题", kind="preference", importance=0.9)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(script=[AIMessage(content='{"need": true, "reason": ""}')] * 2)
    injected = set()
    block1, hits1, _ = await maybe_recall(
        llm, store, constant, "我的偏好？", top_k=3, threshold=0.0, max_chars=800,
        injected_ids=injected,
    )
    assert block1 is not None and len(hits1) == 1
    block2, hits2, _ = await maybe_recall(
        llm, store, constant, "我的偏好？", top_k=3, threshold=0.0, max_chars=800,
        injected_ids=injected,
    )
    assert block2 is None and hits2 == []  # 已见 → 不再注入


@pytest.mark.asyncio
async def test_proactive_budget_cap(embeddings, tmp_path):
    """预算封顶：超 max_chars 的命中被截断（保留能放下的条数）。"""
    store = _store(embeddings, tmp_path)
    store.add("用户偏好深色主题" * 20, kind="preference", importance=0.9)
    constant = _constant(embeddings, tmp_path)
    constant.add("用户要求所有项目用 nodejs" * 20, kind="preference", importance=0.8)
    llm = FakeChatModel(script=[AIMessage(content='{"need": true, "reason": ""}')])
    block, hits, _ = await maybe_recall(
        llm, store, constant, "我的偏好？", top_k=3, threshold=0.0, max_chars=400,
        injected_ids=set(),
    )
    assert block is not None
    assert len(hits) == 1  # 只放得下 1 条（第 2 条超预算截断）
    assert len(block) < 500  # 固定头 + 单条命中；若两条都塞入将超过 500


@pytest.mark.asyncio
async def test_proactive_selector_parse_fail_conservative(embeddings, tmp_path):
    """selector 输出非法 JSON：保守 need=true，照常召回（宁多召，交给阈值/预算兜底）。"""
    store = _store(embeddings, tmp_path)
    store.add("用户偏好深色主题", kind="preference", importance=0.9)
    constant = _constant(embeddings, tmp_path)
    llm = FakeChatModel(script=[AIMessage(content="不是 JSON")])
    block, hits, skipped = await maybe_recall(
        llm, store, constant, "我的偏好？", top_k=3, threshold=0.0, max_chars=400,
        injected_ids=set(),
    )
    assert skipped is False
    assert block is not None and "深色主题" in block


@pytest.mark.asyncio
async def test_proactive_selector_llm_error_conservative(embeddings, tmp_path):
    """selector LLM 调用失败：保守召回，不阻断主链路。"""
    store = _store(embeddings, tmp_path)
    store.add("用户偏好深色主题", kind="preference", importance=0.9)
    constant = _constant(embeddings, tmp_path)

    class BoomLLM:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("selector down")

    block, hits, skipped = await maybe_recall(
        BoomLLM(), store, constant, "我的偏好？", top_k=3, threshold=0.0,
        max_chars=400, injected_ids=set(),
    )
    assert skipped is False
    assert block is not None and "深色主题" in block


@pytest.mark.asyncio
async def test_runner_seeds_constant_ids_for_l2_dedup(settings, registry, sessions):
    """L1/L2 去重打通：首轮 seed 后，L2 主动召回不再重复注入常驻记忆（只召回会话库新增）。"""
    settings.memory_proactive_threshold = 0.0
    settings.memory_consolidate_enabled = False  # 关闭轮末巩固，避免后台写库干扰断言
    runner = AgentRunner(settings, FakeChatModel(), registry, sessions)
    # 常驻库放两条高重要度记忆（会被 L1 注入 system）
    constant = sessions.constant_memory(registry.embeddings, "dev-a")
    constant.add("用户偏好深色主题", kind="preference", importance=0.9)
    constant.add("用户技术栈为 Node.js 和 Python", kind="preference", importance=0.8)
    # 会话库放一条未注入过的（应被 L2 召回到并注入）
    sessions.long_memory("s1", registry.embeddings).add(
        "用户生日是十月十号", kind="fact", importance=0.9
    )
    events = await collect_stream(
        runner,
        session_id="s1",
        message="我的生日是哪天？",
        enabled=["calculator", "memory"],
        rag_enabled=False,
        memory_enabled=True,
        client_key="dev-a",
    )
    # L1 常驻注入了 2 条（system）
    constant_evs = [ev for ev in events if ev["type"] == "memory_constant"]
    assert constant_evs and constant_evs[0]["count"] == 2
    # L2 主动召回：只注入会话库的新记忆，常驻记忆（已 L1 覆盖）不再重复注入
    proactive_evs = [ev for ev in events if ev["type"] == "memory_read" and ev.get("source") == "proactive"]
    assert proactive_evs, "应产出 proactive memory_read 事件"
    ev = proactive_evs[0]
    assert ev["need"] is True
    texts = [h["text"] for h in ev["hits"]]
    assert any("生日" in t for t in texts)
    assert not any("深色主题" in t or "Node" in t for t in texts)


@pytest.mark.asyncio
async def test_runner_write_intent_no_proactive(settings, registry, sessions):
    """写指令（记住 X）：L2 主动召回整轮跳过，产出 need=false 事件且无命中。"""
    settings.memory_proactive_threshold = 0.0
    settings.memory_consolidate_enabled = False
    runner = AgentRunner(settings, FakeChatModel(), registry, sessions)
    sessions.long_memory("s1", registry.embeddings).add(
        "用户生日是十月十号", kind="fact", importance=0.9
    )
    events = await collect_stream(
        runner,
        session_id="s1",
        message="记住我的生日是 十月十号",
        enabled=["calculator", "memory"],
        rag_enabled=False,
        memory_enabled=True,
        client_key="dev-a",
    )
    proactive_evs = [ev for ev in events if ev["type"] == "memory_read" and ev.get("source") == "proactive"]
    assert proactive_evs, "应产出 proactive memory_read 事件"
    ev = proactive_evs[0]
    assert ev["need"] is False
    assert ev["hits"] == []


@pytest.mark.asyncio
async def test_proactive_memory_precedes_rag_stage(settings, registry, sessions):
    """记忆注入在 RAG 之前：事件顺序 memory_read → rag retrieve（记忆先行原则）。"""
    settings.memory_proactive_threshold = 0.0
    settings.memory_consolidate_enabled = False
    runner = AgentRunner(settings, FakeChatModel(), registry, sessions)
    sessions.long_memory("s1", registry.embeddings).add(
        "用户生日是十月十号", kind="fact", importance=0.9
    )
    events = await collect_stream(
        runner,
        session_id="s1",
        message="我的生日是哪天？",
        enabled=["calculator", "memory"],
        rag_enabled=True,
        rag_scheme="naive",
        memory_enabled=True,
        client_key="dev-a",
    )
    memory_idx = next(i for i, e in enumerate(events) if e["type"] == "memory_read")
    rag_idx = next(i for i, e in enumerate(events) if e["type"] == "retrieve")
    assert memory_idx < rag_idx, "memory_read 事件必须先于 RAG retrieve 事件"
    # 主动召回仍注入了会话库记忆（未被 RAG 阶段抢占）
    proactive_evs = [e for e in events if e["type"] == "memory_read" and e.get("source") == "proactive"]
    assert proactive_evs and proactive_evs[0]["hits"]
