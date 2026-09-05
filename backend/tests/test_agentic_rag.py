"""Agentic RAG（企业级）测试：多角色状态机 + 工具治理 + 预算治理（全程离线确定性）。

使用 FakeEmbeddings / FakeChatModel（脚本化角色决策 JSON）与各角色规则回退，
不联网、不依赖 Key。覆盖：
- state：轨迹结构 / token 记账 / 预算与超时判定；
- ToolRegistry：单工具上限、重复去重、非法卷名降级、未知工具拦截、并行波次；
- roles：五角色（Router/Planner/Grader/Corrector/Verifier）LLM JSON 解析与规则回退；
- orchestrator：状态机回环（CRAG 纠错 / Self-RAG 校验）、步数 / token / 超时 /
  纠错轮数预算、角色熔断；
- 方案与接入：AgenticRagScheme 端到端（MemoryStore）、astream 事件序列、
  指代消解 rewrite 事件、跨轮 seed 闸门、manager 预算配置透传、LLM 场景配置。
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.runner import AgentRunner
from app.llm.fake_model import FakeChatModel, FakeEmbeddings
from app.llm.service import DEFAULT_PROFILES
from app.memory.stores.memory_store import MemoryStore
from app.rag.agentic.orchestrator import AgenticOrchestrator, OrchestratorBudgets
from app.rag.agentic.roles import (
    CorrectorAgent,
    GraderAgent,
    PlannerAgent,
    RouterAgent,
    VerifierAgent,
)
from app.rag.agentic.state import REC_ANSWER, REC_CLARIFY, AgentState, ToolCallSpec
from app.rag.agentic.tools import (
    ACTION_HYBRID,
    ACTION_MULTI_HOP,
    ACTION_SEARCH,
    ACTION_VOLUME,
    ToolRegistry,
    cross_turn_seed,
    default_registry_specs,
    volume_catalog,
)
from app.rag.retrieval.context_compress import ExtractiveContextCompressor
from app.rag.retrieval.iterative_retrieval import PlanExecuteRetriever
from app.rag.retrieval.planner import RuleMultiHopPlanner
from app.rag.retrieval.reranker import LexicalReranker
from app.rag.retrieval.verifier import RuleMultiHopVerifier
from app.rag.routing.deictic_resolver import RuleDeicticResolver
from app.rag.routing.query_rewrite import RuleQueryRewriter
from app.rag.schemes.agentic import AgenticRagScheme
from app.rag.schemes.modular import _TARGET_VOLUME_FILTERS
from app.rag.task.decomposer import SCENARIO_DECOMPOSE, TaskDecomposer
from app.rag.task.executor import TaskExecutor
from app.rag.task.gap_center import GapClassifier, GapDecision, GapStrategyCenter
from app.rag.task.graph import NS_RESOLVED, TC_CLARIFIED, TC_COMPLETE, TC_PARTIAL, SessionLedger, TaskBudgets, TaskNode
from app.rag.task.rag_task_tool import make_knowledge_task_tool
from app.tools.rag_tool import _MAX_BLOCK_CHARS, make_knowledge_retrieve_tool, rag_block_payload

# 语料：与查询「报销发票」共现 ≥2 个 2 字词，规则评审/校验确定性通过
DOC_REIMBURSE = "员工报销需要附上发票，随报销单一并提交给财务。"
QUERY_REIMBURSE = "报销发票"


# ---- 测试桩与夹具 ----

@pytest.fixture
def no_role_llm(monkeypatch):
    """角色层全部规则回退（模拟无 Key 离线环境）。"""
    monkeypatch.setattr("app.rag.agentic.roles.get_chat_model", lambda scenario: None)


def role_msg(payload: dict, usage: tuple[int, int] | None = None) -> AIMessage:
    """构造角色决策 JSON 消息（usage=(prompt, completion) 时附带 token 用量）。"""
    kw: dict = {"content": json.dumps(payload, ensure_ascii=False)}
    if usage:
        kw["usage_metadata"] = {
            "input_tokens": usage[0],
            "output_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        }
    return AIMessage(**kw)


ROUTE_MSG = role_msg(
    {"retrieval_need": True, "generation_mode": "citation", "target": "none", "reason": "需要检索"}
)
PLAN_MSG = role_msg(
    {
        "facts": ["发票报销时限"],
        "calls": [{"action": "hybrid", "query": "发票 报销 时限", "reason": "主路混合检索"}],
        "reason": "单事实单路",
    }
)
GRADE_MSG = role_msg({"relevant": [0], "missing_facts": [], "reason": "首条相关"})
VERIFY_OK = role_msg({"answerable": True, "missing_facts": [], "reason": "事实有证据支持"})


def make_store(texts: list[str] | None = None) -> MemoryStore:
    store = MemoryStore(FakeEmbeddings(), collection="agentic_test")
    for t in texts or []:
        store.add(t)
    return store


def make_scheme(store=None, **kw) -> AgenticRagScheme:
    """构造 agentic 方案：显式注入规则实现，保证离线确定性。"""
    if store is None:
        store = make_store()
    kw.setdefault("rewriter", RuleQueryRewriter())
    kw.setdefault("reranker", LexicalReranker())
    kw.setdefault("deictic", RuleDeicticResolver())
    kw.setdefault("compressor", ExtractiveContextCompressor())
    kw.setdefault("multi_hop", PlanExecuteRetriever(RuleMultiHopPlanner(), RuleMultiHopVerifier()))
    return AgenticRagScheme(FakeEmbeddings(), store, top_k=kw.pop("top_k", 3), **kw)


def make_orchestrator(store, **budget_kw) -> AgenticOrchestrator:
    return AgenticOrchestrator(
        store,
        FakeEmbeddings(),
        LexicalReranker(),
        ExtractiveContextCompressor(),
        parent_resolver=lambda hits: hits,
        multi_hop=None,
        budgets=OrchestratorBudgets(**budget_kw),
    )


class RecordingStore(MemoryStore):
    """记录 search 调用的 MemoryStore，供 hybrid 内隐式 HyDE 一路断言。"""

    def __init__(self, texts: list[str] | None = None):
        super().__init__(FakeEmbeddings(), collection="agentic_test")
        self.search_calls: list[str] = []
        for t in texts or []:
            self.add(t)

    def search(self, query: str, top_k: int = 3, volume_filter: tuple[str, ...] | None = None):
        self.search_calls.append(query)
        return super().search(query, top_k)


class StubHyde:
    """测试桩：固定假想文档（与查询不同），模拟 LLM HyDE 输出。"""

    def __init__(self, doc: str = "发票报销提交时限规定"):
        self.doc = doc

    def expand(self, query: str):
        return self.doc


def scripted_llm(**by_scenario) -> callable:
    """按场景注入不同脚本模型（未注入的场景返回 None → 规则回退）。"""
    return lambda scenario: by_scenario.get(scenario)


# ---- state：状态与轨迹 ----

def test_state_seq_tokens_and_trace():
    state = AgentState(query="q")
    assert state.next_seq() == 1
    state.add_tokens({"prompt": 10, "completion": 5})
    state.add_tokens({"prompt": 1, "completion": 0})
    assert state.tokens == {"prompt": 11, "completion": 5}
    assert not state.over_budget(token_budget=100)
    assert state.over_budget(token_budget=16) and state.over_budget(token_budget=11)
    assert not state.over_budget(token_budget=0), "0 表示不限 token"
    assert not state.timed_out(), "deadline=0 表示未设超时"


def test_state_trace_structure():
    state = AgentState(query="q")
    state.role_llm_calls["route"] = 1
    state.tool_calls["hybrid"] = 2
    state.total_tool_exec = 2
    trace = state.trace(corrections=1)
    assert trace["total_events"] == 0
    assert trace["role_llm_calls"] == {"route": 1}
    assert trace["tool_calls"] == {"hybrid": 2}
    assert trace["total_tool_exec"] == 2
    assert trace["corrections"] == 1
    assert trace["tokens"] == {"prompt": 0, "completion": 0}
    assert trace["steps"] == []


# ---- ToolRegistry：工具治理 ----

def test_registry_call_cap_intercepts():
    """单工具调用超 call_cap → 拦截（note 说明、不执行、仍消耗预算计数）。"""
    registry = ToolRegistry(make_store([DOC_REIMBURSE]), call_cap=1)
    state = AgentState(query="测试")
    wave = registry.execute_wave(
        [ToolCallSpec(ACTION_SEARCH, "报销"), ToolCallSpec(ACTION_SEARCH, "发票")], k=3, recall_k=9, state=state
    )
    assert not wave[0].note and wave[0].hits, "首个调用正常执行"
    assert "上限" in wave[1].note and wave[1].hits == []
    assert state.tool_calls == {ACTION_SEARCH: 2}, "被拦截调用也计入预算"


def test_registry_duplicate_skipped():
    """跨波重复调用去重：同 (action, query) 在后续波次被拦截（同波不去重，允许首发多路撞车）。"""
    registry = ToolRegistry(make_store([DOC_REIMBURSE]), call_cap=3)
    state = AgentState(query="测试")
    first = registry.execute_wave([ToolCallSpec(ACTION_SEARCH, "报销发票")], k=3, recall_k=9, state=state)
    assert not first[0].note and first[0].hits
    second = registry.execute_wave([ToolCallSpec(ACTION_SEARCH, "报销发票")], k=3, recall_k=9, state=state)
    assert "重复" in second[0].note and second[0].hits == []


def test_registry_illegal_volume_degrades():
    """volume_search 卷名不在目录 → 降级全库检索（保留意图、正常返回命中）。"""
    registry = ToolRegistry(make_store([DOC_REIMBURSE]), call_cap=3)
    state = AgentState(query="测试")
    wave = registry.execute_wave(
        [ToolCallSpec(ACTION_VOLUME, "报销发票", volume="不存在的卷")], k=3, recall_k=9, state=state
    )
    assert wave[0].note == "" and wave[0].hits, "降级后应正常执行并返回命中"
    assert wave[0].call.action == ACTION_VOLUME


def test_registry_unknown_action_intercepted():
    """注册表白名单外动作 → 直接拦截（不执行）。"""
    registry = ToolRegistry(make_store([DOC_REIMBURSE]), call_cap=3)
    state = AgentState(query="测试")
    wave = registry.execute_wave([ToolCallSpec("web_search", "外部检索")], k=3, recall_k=9, state=state)
    assert "不在注册表内" in wave[0].note and wave[0].hits == []


def test_registry_parallel_wave_executes_all():
    """一波多路调用（不同工具）并行执行，互不影响。"""
    registry = ToolRegistry(make_store([DOC_REIMBURSE]), call_cap=3, parallel=2)
    state = AgentState(query="测试")
    wave = registry.execute_wave(
        [ToolCallSpec(ACTION_SEARCH, "报销发票"), ToolCallSpec(ACTION_HYBRID, "发票 报销")],
        k=3, recall_k=9, state=state,
    )
    assert all(not wr.note and wr.hits for wr in wave)
    assert state.tool_calls == {ACTION_SEARCH: 1, ACTION_HYBRID: 1}


def test_registry_hybrid_implicit_hyde_fires():
    """hybrid 工具内隐式 HyDE：假想文档作额外一路 doc-space 稠密召回并入 RRF（Agent 无感知）。"""
    store = RecordingStore([DOC_REIMBURSE])
    registry = ToolRegistry(store, hyde=StubHyde(doc="发票报销提交时限规定"))
    state = AgentState(query="报销发票")
    wave = registry.execute_wave([ToolCallSpec(ACTION_HYBRID, "报销发票")], k=3, recall_k=9, state=state)
    assert not wave[0].note and wave[0].hits, "hybrid 应正常返回命中"
    assert "发票报销提交时限规定" in store.search_calls, "HyDE 假想文档应作为一路被检索"


def test_registry_hybrid_implicit_hyde_rule_fallback_skips():
    """HyDE 规则回退（返回原查询）：不追加额外检索，hybrid 结果不受影响。"""
    store = RecordingStore([DOC_REIMBURSE])

    class NoopHyde:
        def expand(self, query):
            return query

    registry = ToolRegistry(store, hyde=NoopHyde())
    state = AgentState(query="报销发票")
    wave = registry.execute_wave([ToolCallSpec(ACTION_HYBRID, "报销发票")], k=3, recall_k=9, state=state)
    assert not wave[0].note and wave[0].hits, "规则回退时 hybrid 本身仍正常返回"
    # hybrid 默认退化单路 search（仅查询本身那一次），不追加 HyDE 假想文档一路
    assert store.search_calls == ["报销发票"], "规则回退时不追加额外 HyDE 检索"


def test_registry_default_specs_multi_hop_tight_cap():
    """默认注册表：multi_hop 成本高，独立收紧为 1 次。"""
    specs = {s.name: s for s in default_registry_specs(call_cap=3)}
    assert specs[ACTION_MULTI_HOP].call_cap == 1
    assert specs[ACTION_HYBRID].call_cap == 3
    assert set(specs) == {ACTION_SEARCH, ACTION_HYBRID, ACTION_VOLUME, ACTION_MULTI_HOP}


def test_volume_catalog_aggregates_and_dedups():
    """卷目录：聚合全部定向白名单卷且去重保序。"""
    catalog = volume_catalog()
    expected = {v for vols in _TARGET_VOLUME_FILTERS.values() for v in vols}
    assert set(catalog) == expected
    assert len(catalog) == len(set(catalog))


def test_cross_turn_seed_gate():
    """跨轮 seed 闸门：低分丢弃、无共现丢弃、限量 5 条。"""
    q = "报销发票"
    good = {"text": "发票须随报销单一并提交。", "score": 0.9}
    low = {"text": "发票报销相关内容", "score": 0.4}
    unrelated = {"text": "今日食堂供应红烧肉。", "score": 0.95}
    kept = cross_turn_seed(q, [good, low, unrelated])
    assert good in kept and low not in kept and unrelated not in kept
    many = [{"text": f"发票报销材料清单第{i}项", "score": 0.9} for i in range(8)]
    assert len(cross_turn_seed(q, many)) == 5


# ---- roles：LLM 决策 + 规则回退 ----

def test_router_llm_parse_and_mode_normalization(monkeypatch):
    """路由：合法 JSON 解析；非法 generation_mode 归一为 citation。"""
    llm = FakeChatModel(script=[
        role_msg({"retrieval_need": False, "generation_mode": "direct", "reason": "寒暄"}),
        role_msg({"retrieval_need": True, "generation_mode": "离谱值", "reason": "x"}),
    ])
    monkeypatch.setattr("app.rag.agentic.roles.get_chat_model", lambda scenario: llm)
    router = RouterAgent()
    out = router.run("你好", AgentState(query="你好"))
    assert out.retrieval_need is False and out.generation_mode == "direct" and not out.note
    out2 = router.run("问题", AgentState(query="问题"))
    assert out2.generation_mode == "citation", "越界模式归一为 citation"


def test_router_invalid_output_falls_back(monkeypatch):
    """路由：LLM 输出不可解析 → 规则回退（保守检索 + citation）。"""
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model", lambda scenario: FakeChatModel(script=[AIMessage(content="不是JSON")])
    )
    out = RouterAgent().run("报销发票", AgentState(query="报销发票"))
    assert out.retrieval_need is True and out.generation_mode == "citation"
    assert out.note, "回退应有 note（熔断计数依据）"


def test_planner_llm_parses_facts_and_calls(monkeypatch):
    """规划：解析事实清单 + 首发多路调用。"""
    llm = FakeChatModel(script=[PLAN_MSG])
    monkeypatch.setattr("app.rag.agentic.roles.get_chat_model", lambda scenario: llm)
    out = PlannerAgent().run("发票时限", ["卷A"], AgentState(query="发票时限"))
    assert out.facts == ["发票报销时限"]
    assert [c.action for c in out.calls] == [ACTION_HYBRID]
    assert out.calls[0].query == "发票 报销 时限" and not out.note


def test_planner_filters_illegal_actions_and_empty_facts(monkeypatch):
    """规划：越界动作过滤（空计划回退默认 hybrid）；空事实清单 → 规则回退。"""
    llm = FakeChatModel(script=[
        role_msg({"facts": ["事实"], "calls": [{"action": "自造", "query": "x"}], "reason": "r"}),
        role_msg({"facts": ["  "], "calls": [], "reason": "r"}),
    ])
    monkeypatch.setattr("app.rag.agentic.roles.get_chat_model", lambda scenario: llm)
    out = PlannerAgent().run("q", [], AgentState(query="q"))
    assert [c.action for c in out.calls] == [ACTION_HYBRID], "动作越界过滤后回退默认首发"
    assert not out.note
    out2 = PlannerAgent().run("q", [], AgentState(query="q"))
    assert out2.facts == ["q"] and out2.note, "空事实清单回退规则"


def test_grader_llm_keeps_in_range_only(monkeypatch):
    """评审：越界/非法下标过滤，仅保留范围内相关证据。"""
    llm = FakeChatModel(script=[role_msg({"relevant": [0, 5, "x", -1], "missing_facts": ["时限"]})])
    monkeypatch.setattr("app.rag.agentic.roles.get_chat_model", lambda scenario: llm)
    hits = [{"text": DOC_REIMBURSE}, {"text": "无关内容"}]
    out = GraderAgent().run("报销发票", hits, AgentState(query="报销发票"))
    assert out.keep == [0]
    assert out.missing_facts == ["时限"]


def test_grader_rule_fallback_lexical_overlap(no_role_llm):
    """评审规则回退：2 字词共现 ≥2 或分数 ≥0.5 判相关（宽松防误杀）。"""
    hits = [{"text": DOC_REIMBURSE, "score": 0.1}, {"text": "完全无关的内容", "score": 0.9}]
    out = GraderAgent().run(QUERY_REIMBURSE, hits, AgentState(query=QUERY_REIMBURSE))
    assert out.keep == [0, 1], "共现达标的留、高分（≥0.5）无关的也留"
    assert out.note


def test_corrector_llm_parses_calls(monkeypatch):
    """纠错：解析下一波工具调用 JSON。"""
    payload = {
        "calls": [{"action": "volume_search", "query": "报销时限", "volume": "卷A", "reason": "缺口"}],
        "reason": "定向补查",
    }
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model", lambda scenario: FakeChatModel(script=[role_msg(payload)])
    )
    out = CorrectorAgent().run("报销发票", ["报销时限"], [], ["卷A"], AgentState(query="q"))
    assert [c.action for c in out.calls] == [ACTION_VOLUME]
    assert out.calls[0].volume == "卷A" and not out.note


def test_corrector_injects_prior_evidence_summary(monkeypatch):
    """纠错：prior_hits 摘要注入提示词——下一跳查询可基于已确认证据（如张三=研发部→查研发部人数）。"""
    calls: list[list[tuple[str, str]]] = []  # (system, user) 快照

    class Recording(FakeChatModel):
        def invoke(self, messages, **kw):
            calls.append([(m.content if isinstance(m.content, str) else str(m.content)) for m in messages])
            return super().invoke(messages, **kw)

    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        lambda scenario: Recording(script=[role_msg({"calls": [{"action": "hybrid", "query": "研发部 在职人数"}], "reason": "x"})]),
    )
    prior = [{"text": "张三属于研发部。", "metadata": {"volume": "档案卷"}}]
    out = CorrectorAgent().run(
        "张三的部门有多少人", ["部门的人数"], [], ["档案卷"], AgentState(query="q"), prior_hits=prior
    )
    assert [c.action for c in out.calls] == [ACTION_HYBRID]
    assert out.calls[0].query == "研发部 在职人数", "下一跳查询基于已确认证据推进"
    assert calls and "先前轮已确认的证据" in calls[0][1], "用户消息应注入已确认证据摘要"
    assert "张三属于研发部" in calls[0][1]


def test_corrector_rule_fallback_volume_then_hybrid(no_role_llm):
    """纠错规则回退：每个缺失事实一路——首选未用过的定向卷，无卷可用时 hybrid。"""
    out = CorrectorAgent().run(
        QUERY_REIMBURSE, ["报销时限"], [], ["卷A", "卷B"], AgentState(query=QUERY_REIMBURSE)
    )
    assert [c.action for c in out.calls] == [ACTION_VOLUME], "单缺失事实 → 一路定向卷"
    assert out.calls[0].volume == "卷A"
    # 两个缺失事实：第一路定向卷（未用过的），第二路 hybrid
    out2 = CorrectorAgent().run(
        QUERY_REIMBURSE, ["报销时限", "审批人"], [], ["卷A", "卷B"], AgentState(query=QUERY_REIMBURSE)
    )
    assert [c.action for c in out2.calls] == [ACTION_VOLUME, ACTION_HYBRID]
    # 已用过的卷不再重复定向
    executed = [ToolCallSpec(ACTION_VOLUME, "x", "卷A")]
    out3 = CorrectorAgent().run(QUERY_REIMBURSE, ["报销时限"], executed, ["卷A", "卷B"], AgentState(query="q"))
    assert out3.calls[0].volume == "卷B"
    # 无目录 → 全部 hybrid
    out4 = CorrectorAgent().run(QUERY_REIMBURSE, ["缺口一"], [], [], AgentState(query="q"))
    assert all(c.action == ACTION_HYBRID for c in out4.calls)


def test_verifier_llm_and_rule_fallback(monkeypatch, no_role_llm):
    """校验：LLM 解析 answerable；不可解析 → 词法覆盖规则回退。"""
    llm = FakeChatModel(script=[role_msg({"answerable": True, "reason": "支持"}), AIMessage(content="坏输出")])
    monkeypatch.setattr("app.rag.agentic.roles.get_chat_model", lambda scenario: llm)
    out = VerifierAgent().run(QUERY_REIMBURSE, ["发票报销时限"], [{"text": DOC_REIMBURSE}], AgentState(query="q"))
    assert out.answerable is True and not out.note
    out2 = VerifierAgent().run(QUERY_REIMBURSE, ["发票报销"], [{"text": DOC_REIMBURSE}], AgentState(query="q"))
    assert out2.note, "解析失败回退规则"
    # 规则：事实 2 字词 ≥50% 被证据覆盖 → 可答（「发票报销」3 个 2 字词覆盖 2 个）
    assert out2.answerable is True


def test_verifier_rule_fallback_skips_confirmed_facts(no_role_llm):
    """校验规则回退：已确认事实直接跳过，不再因词法误判报缺失（防多轮遗忘）。

    词法对抽象事实天然误判（「张三所属的部门名称」与「张三在研发部」2 字词共现率
    极低 <50%）——若没有 confirmed_facts 记忆，规则回退会把已确认事实误报为缺失。
    """
    facts = ["张三所属的部门名称", "张三所在部门的人数"]
    evidence = [{"text": "张三在研发部。"}]
    # 无已确认记忆：抽象事实「张三所属的部门名称」被词法误判缺失
    out = VerifierAgent().run("张三的部门有多少人", facts, evidence, AgentState(query="q"))
    assert out.note and "张三所属的部门名称" in out.missing_facts, "无记忆时抽象事实被词法误判"
    # 有已确认记忆：该事实跳过，不再报缺失
    out2 = VerifierAgent().run(
        "张三的部门有多少人", facts, evidence, AgentState(query="q"),
        confirmed_facts=["张三所属的部门名称"],
    )
    assert "张三所属的部门名称" not in out2.missing_facts, "已确认事实不应再报缺失"
    assert "张三所在部门的人数" in out2.missing_facts, "未确认事实仍正常判定"


def test_verifier_llm_filters_confirmed_from_missing(monkeypatch):
    """校验 LLM 路径：即使 LLM 把已确认事实误报进 missing_facts，也被过滤掉。"""
    payload = role_msg({
        "answerable": False,
        "missing_facts": ["张三所属的部门名称", "张三所在部门的人数"],
        "reason": "模型误报",
    })
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model", lambda scenario: FakeChatModel(script=[payload])
    )
    out = VerifierAgent().run(
        "张三的部门有多少人", ["张三所属的部门名称", "张三所在部门的人数"], [],
        AgentState(query="q"), confirmed_facts=["张三所属的部门名称"],
    )
    assert "张三所属的部门名称" not in out.missing_facts, "LLM 误报的已确认事实被过滤"
    assert out.missing_facts == ["张三所在部门的人数"]


# ---- orchestrator：状态机 + 预算治理 ----

def test_orchestrator_rule_fallback_end_to_end(no_role_llm):
    """规则回退全链路：route→plan→retrieve→grade→verify 可答收尾，轨迹完整。"""
    orch = make_orchestrator(make_store([DOC_REIMBURSE]))
    result = orch.run(QUERY_REIMBURSE, k=3)
    assert result.hits and result.reranked
    assert result.answerable and result.verdict["recommendation"] == REC_ANSWER
    assert result.generation_mode == "citation" and result.facts == [QUERY_REIMBURSE]
    trace = result.trace
    assert trace["tool_calls"].get(ACTION_HYBRID, 0) == 1
    assert trace["total_tool_exec"] == 1 and trace["corrections"] == 0
    assert trace["role_llm_calls"] == {}, "规则回退不消耗 LLM"
    roles = [s["role"] for s in trace["steps"]]
    assert roles[0] == "route" and roles[1] == "plan" and roles[-1] == "verify"
    assert any(s["role"] == "retriever" and s["hits"] > 0 for s in trace["steps"])
    assert all(s["note"] for s in trace["steps"] if s["role"] in ("route", "plan", "grade", "verify"))


def test_orchestrator_llm_scripted_roles(no_role_llm, monkeypatch):
    """LLM 角色：脚本化 route→plan→grade→verify(可答)，token/调用次数入账。"""
    llm = FakeChatModel(script=[ROUTE_MSG, PLAN_MSG, GRADE_MSG, VERIFY_OK])
    monkeypatch.setattr("app.rag.agentic.roles.get_chat_model", lambda scenario: llm)
    store = make_store([DOC_REIMBURSE])
    orch = make_orchestrator(store, token_budget=0)
    # ROUTE/PLAN/... 的 token 用量来自 usage_metadata：给脚本消息加 usage
    result = orch.run("报销发票 流程", k=3)
    assert result.answerable and result.generation_mode == "citation"
    assert result.facts == ["发票报销时限"]
    trace = result.trace
    assert trace["role_llm_calls"] == {"route": 1, "plan": 1, "grade": 1, "verify": 1}
    assert result.hits, "LLM 评审保留首条 → 证据非空"


def test_orchestrator_no_retrieval_route(no_role_llm, monkeypatch):
    """路由判定无需检索（寒暄）→ 空结果直接收尾，不进检索回环。"""
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(**{"rag_agent_route": FakeChatModel(script=[
            role_msg({"retrieval_need": False, "generation_mode": "direct", "reason": "寒暄"}),
        ])}),
    )
    orch = make_orchestrator(make_store([DOC_REIMBURSE]))
    result = orch.run("你好", k=3)
    assert result.hits == [] and result.retrieval_need is False
    assert result.generation_mode == "direct"
    assert result.trace["total_tool_exec"] == 0
    assert [s["role"] for s in result.trace["steps"]] == ["route"]


def test_orchestrator_correction_loop_recovers(no_role_llm, monkeypatch):
    """CRAG 纠错闭环：首轮校验不足 → 纠错波补检索 → 二轮校验可答（corrections=1）。"""
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(**{"rag_agent_verify": FakeChatModel(script=[
            role_msg({"answerable": False, "missing_facts": ["报销时限"], "reason": "缺时限"}),
            VERIFY_OK,
        ])}),
    )
    orch = make_orchestrator(make_store([DOC_REIMBURSE]))
    result = orch.run(QUERY_REIMBURSE, k=3)
    assert result.answerable and result.corrections == 1
    trace = result.trace
    assert trace["tool_calls"].get(ACTION_VOLUME, 0) == 1, "规则纠错应定向卷补检索"
    assert trace["tool_calls"].get(ACTION_HYBRID, 0) == 1
    assert trace["total_tool_exec"] == 2
    verify_events = [s for s in trace["steps"] if s["role"] == "verify"]
    assert len(verify_events) == 2 and verify_events[1]["note"] == "", "二轮走 LLM 校验"
    correct_events = [s for s in trace["steps"] if s["role"] == "correct"]
    assert correct_events and correct_events[0]["note"], "纠错走规则回退"


def test_orchestrator_correction_rounds_exhausted_clarifies(no_role_llm, monkeypatch):
    """纠错轮数耗尽：始终校验不足 → 如实上报 clarify，不再回环。"""
    not_answerable = role_msg({"answerable": False, "missing_facts": ["完全缺失的事实"], "reason": "缺"})
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(**{"rag_agent_verify": FakeChatModel(script=[not_answerable, not_answerable])}),
    )
    orch = make_orchestrator(make_store([DOC_REIMBURSE]), correction_rounds=1)
    result = orch.run(QUERY_REIMBURSE, k=3)
    assert not result.answerable and result.corrections == 1
    assert result.verdict["recommendation"] == REC_CLARIFY
    assert "完全缺失的事实" in result.missing_facts


def test_orchestrator_confirmed_facts_carry_across_rounds(no_role_llm, monkeypatch):
    """跨轮记忆：首轮校验后已确认事实写回，第二轮 Verifier 不再把其误报为缺失。

    模拟「张三的部门有多少人」：首轮 LLM 校验确认「部门归属」（未列入 missing），
    只缺「人数」→ 纠错补检索 → 第二轮校验时应带着 confirmed_facts 再次调用，
    即便规则回退（词法）也不会把抽象事实「张三所属的部门名称」误报缺失。
    """
    user_bodies: list[str] = []  # 记录每轮 Verifier 用户消息

    class Recording(FakeChatModel):
        def invoke(self, messages, **kw):
            user_bodies.append(next(m.content for m in messages if getattr(m, "type", "") == "human"))
            return super().invoke(messages, **kw)

    plan_two_facts = role_msg({
        "facts": ["张三所属的部门名称", "张三所在部门的人数"],
        "calls": [{"action": "hybrid", "query": "张三 部门 人数", "reason": "主路"}],
        "reason": "两事实",
    })
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(**{
            "rag_agent_route": FakeChatModel(script=[ROUTE_MSG]),
            "rag_agent_plan": FakeChatModel(script=[plan_two_facts]),
            "rag_agent_verify": Recording(script=[
                role_msg({"answerable": False, "missing_facts": ["张三所在部门的人数"], "reason": "缺人数"}),
                role_msg({"answerable": True, "missing_facts": [], "reason": "有证据支持"}),
            ]),
        }),
    )
    orch = make_orchestrator(make_store([DOC_REIMBURSE]), correction_rounds=2)
    result = orch.run(QUERY_REIMBURSE, k=3)
    assert result.answerable and result.corrections == 1
    assert len(user_bodies) == 2, "两轮校验各一次 LLM 调用"
    assert "先前轮已确认的事实" in user_bodies[1], "第二轮应注入已确认事实记忆"
    assert "张三所属的部门名称" in user_bodies[1], "首轮确认的部门归属写回并注入第二轮"


def test_orchestrator_max_steps_budget(no_role_llm):
    """步数预算：max_steps=1 执行一波后即使校验不足也不再纠错。"""
    orch = make_orchestrator(make_store(["完全无关的语料内容。"]), max_steps=1)
    result = orch.run(QUERY_REIMBURSE, k=3)
    assert result.trace["total_tool_exec"] == 1
    assert result.corrections == 0 and not result.answerable


def test_orchestrator_timeout_gates_llm_and_tools(no_role_llm, monkeypatch):
    """墙钟超时：超时后角色规则回退且不再发起工具波次。"""
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(**{"rag_agent_route": FakeChatModel(script=[ROUTE_MSG])}),
    )
    orch = make_orchestrator(make_store([DOC_REIMBURSE]), timeout_s=0.0)
    result = orch.run(QUERY_REIMBURSE, k=3)
    assert result.trace["total_tool_exec"] == 0, "超时不发起新波次"
    assert result.trace["role_llm_calls"] == {}, "超时角色一律规则回退（不调 LLM）"
    assert not result.answerable and result.hits == []


def test_orchestrator_token_budget_degrades_to_rules(no_role_llm, monkeypatch):
    """token 预算：路由 LLM 耗尽预算 → 后续角色全部规则回退。"""
    big_route = role_msg(
        {"retrieval_need": True, "generation_mode": "citation", "reason": "r"}, usage=(80, 20)
    )
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(**{"rag_agent_route": FakeChatModel(script=[big_route])}),
    )
    orch = make_orchestrator(make_store([DOC_REIMBURSE]), token_budget=100)
    result = orch.run(QUERY_REIMBURSE, k=3)
    trace = result.trace
    assert trace["role_llm_calls"] == {"route": 1}
    assert trace["tokens"] == {"prompt": 80, "completion": 20}
    notes = {s["role"]: s["note"] for s in trace["steps"] if s["note"]}
    assert "规划规则回退" in notes["plan"] and "校验规则回退" in notes["verify"]


def test_orchestrator_role_circuit_breaker(no_role_llm, monkeypatch):
    """熔断：verify 连续 2 次 LLM 决策失败 → 第 3 次起锁定规则回退（不再调 LLM）。

    语料与查询无关：规则校验恒判不可答 → 持续纠错直到纠错轮数耗尽，
    从而观察 verify 被多次调用时 LLM 消费在熔断点停住。
    """
    llm = FakeChatModel(script=[AIMessage(content="坏输出"), AIMessage(content="坏输出"), VERIFY_OK])
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model", scripted_llm(**{"rag_agent_verify": llm})
    )
    orch = make_orchestrator(make_store(["完全无关的语料内容。"]), correction_rounds=3, token_budget=0)
    result = orch.run(QUERY_REIMBURSE, k=3)
    assert len(llm.script) == 1, "第 3 次 verify 已熔断，脚本第 3 条消息未被消费"
    verify_events = [s for s in result.trace["steps"] if s["role"] == "verify"]
    assert len(verify_events) >= 3, "多次校验均发生"
    assert all(e["note"] for e in verify_events), "校验全程规则回退"
    assert result.corrections == 3 and not result.answerable


# ---- AgenticRagScheme：端到端与流式协议 ----

def test_scheme_retrieve_full_end_to_end(no_role_llm):
    """端到端：入库 → 编排检索 → 命中/重排/闸门/轨迹齐全。"""
    scheme = make_scheme(make_store([DOC_REIMBURSE]))
    result = scheme.retrieve_full(QUERY_REIMBURSE, top_k=3)
    assert result.hits and result.reranked is True
    assert result.answerability["answerable"] is True
    assert result.answerability["recommendation"] == REC_ANSWER
    trace = result.trace
    assert trace["steps"][0]["role"] == "route"
    assert trace["tool_calls"].get(ACTION_HYBRID, 0) >= 1
    assert trace["total_events"] == len(trace["steps"])


async def test_scheme_astream_event_sequence(no_role_llm):
    """流式事件序列：classify→plan→agent_step→grade→verify→retrieve→answerability。"""
    scheme = make_scheme(make_store([DOC_REIMBURSE]))
    events = [ev async for ev in scheme.astream(QUERY_REIMBURSE, 3)]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "classify" and kinds[1] == "classify"
    assert kinds.count("classify") == 2, "running/done 两条"
    for expected in ("plan", "agent_step", "grade", "verify", "retrieve", "answerability"):
        assert expected in kinds, f"缺少 {expected} 事件"
    assert kinds.index("classify") < kinds.index("plan") < kinds.index("agent_step")
    assert kinds.index("grade") < kinds.index("verify") < kinds.index("retrieve")
    assert kinds[-1] == "answerability"
    plan_done = next(e for e in events if e["type"] == "plan" and e["status"] == "done")
    assert plan_done["facts"] == [QUERY_REIMBURSE] and plan_done["calls"][0]["action"] == ACTION_HYBRID
    steps = [e for e in events if e["type"] == "agent_step"]
    retrieve = next(e for e in events if e["type"] == "retrieve")
    assert [s["step"]["index"] for s in steps] == [
        s["seq"] for s in retrieve["trace"]["steps"] if s["role"] == "retriever"
    ], "agent_step 步号与轨迹中检索步的全局序号一致"
    assert retrieve["hits"] and retrieve["trace"]["total_events"] >= len(steps)
    verdict = events[-1]["verdict"]
    assert verdict["answerable"] is True and verdict["recommendation"] == REC_ANSWER


async def test_scheme_astream_no_retrieval_only_classify(monkeypatch):
    """寒暄流式：路由判无需检索 → 仅 classify 两条，无后续事件。"""
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(**{"rag_agent_route": FakeChatModel(script=[
            role_msg({"retrieval_need": False, "generation_mode": "direct", "reason": "寒暄"}),
        ])}),
    )
    scheme = make_scheme(make_store([DOC_REIMBURSE]))
    events = [ev async for ev in scheme.astream("你好呀", 3)]
    assert [e["type"] for e in events] == ["classify", "classify"]
    assert events[1]["retrieval_need"] is False and events[1]["generation_mode"] == "direct"


async def test_scheme_deictic_rewrite_event():
    """指代消解：消解命中时流式先发 rewrite 事件，检索用消解后查询。"""

    class StubDeictic:
        def resolve(self, query: str, context: str | None) -> str:
            if context and "张三" in context and "他" in query:
                return query.replace("他", "张三")
            return query

    scheme = make_scheme(make_store([DOC_REIMBURSE]), deictic=StubDeictic())
    events = [ev async for ev in scheme.astream("他的报销发票要求", 3, context="用户: 张三的报销制度")]
    rewrite = events[0]
    assert rewrite["type"] == "rewrite" and rewrite["reason"] == "指代消解"
    assert rewrite["rewrites"] == ["张三的报销发票要求"]
    retrieve = next(e for e in events if e["type"] == "retrieve")
    assert retrieve["query"] == "张三的报销发票要求"
    # 同步路径：结果 query 为消解后文本
    result = scheme.retrieve_full("他的报销发票要求", context="用户: 张三的报销制度")
    assert result.query == "张三的报销发票要求"


def test_scheme_seed_reuse_merged(no_role_llm):
    """跨轮 seed：经闸门过滤后作为额外一路参与融合，不丢当前轮召回。"""
    scheme = make_scheme(make_store([DOC_REIMBURSE]))
    seed = [{"text": "发票须在费用发生后三个月内提交报销，逾期不予受理。", "score": 0.9}]
    result = scheme.retrieve_full(QUERY_REIMBURSE, top_k=3, seed_hits=seed)
    joined = "\n".join(h.get("text", "") for h in result.hits)
    assert "三个月内提交" in joined, "seed 证据应并入最终命中"
    assert "报销单" in joined, "当前轮召回仍应保留"


def test_scheme_llm_full_pipeline(no_role_llm, monkeypatch):
    """五角色 LLM 全链路：脚本化决策贯通（verify 可答，corrector 不触发）。"""
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(
            **{
                "rag_agent_route": FakeChatModel(script=[ROUTE_MSG]),
                "rag_agent_plan": FakeChatModel(script=[PLAN_MSG]),
                "rag_agent_grade": FakeChatModel(script=[GRADE_MSG]),
                "rag_agent_verify": FakeChatModel(script=[VERIFY_OK]),
            }
        ),
    )
    scheme = make_scheme(make_store([DOC_REIMBURSE]))
    result = scheme.retrieve_full("报销发票时限", top_k=3)
    assert result.answerability["answerable"] is True
    assert result.trace["role_llm_calls"] == {"route": 1, "plan": 1, "grade": 1, "verify": 1}
    assert result.trace["corrections"] == 0
    assert result.trace["tool_calls"] == {ACTION_HYBRID: 1}
    plan_event_role = result.trace["steps"][1]
    assert plan_event_role["role"] == "plan" and not plan_event_role["note"]


async def test_scheme_astream_llm_events(no_role_llm, monkeypatch):
    """流式 + LLM：plan/grade/verify 事件携带 LLM 决策内容。"""
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(
            **{
                "rag_agent_route": FakeChatModel(script=[ROUTE_MSG]),
                "rag_agent_plan": FakeChatModel(script=[PLAN_MSG]),
                "rag_agent_grade": FakeChatModel(script=[GRADE_MSG]),
                "rag_agent_verify": FakeChatModel(script=[VERIFY_OK]),
            }
        ),
    )
    scheme = make_scheme(make_store([DOC_REIMBURSE]))
    events = [ev async for ev in scheme.astream("报销发票时限", 3)]
    plan = next(e for e in events if e["type"] == "plan" and e.get("status") == "done")
    assert plan["facts"] == ["发票报销时限"] and plan["thought"] == "单事实单路"
    grade = next(e for e in events if e["type"] == "grade")
    assert grade["kept"] == 1 and grade["missing_facts"] == []
    verify = next(e for e in events if e["type"] == "verify")
    assert verify["answerable"] is True
    assert all(e["type"] != "correct" for e in events), "可答不触发纠错"


# ---- 接入：manager 构建与配置 ----

def test_manager_builds_agentic_with_budgets(settings, no_role_llm):
    """rag_schemes 含 agentic：manager 构建方案并透传全部预算配置。"""
    settings.rag_schemes = ["agentic"]
    from app.rag.manager import RagManager

    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    scheme = manager.get("agentic")
    assert isinstance(scheme, AgenticRagScheme)
    assert manager.resolve("agentic") is scheme
    budgets = scheme.orchestrator.budgets
    assert budgets.max_steps == settings.rag_agent_max_steps == 8
    assert budgets.correction_rounds == settings.rag_agent_correction_rounds == 2
    assert budgets.timeout_s == settings.rag_agent_timeout_s == 90.0
    assert budgets.token_budget == settings.rag_agent_token_budget == 0  # 暂放开（0=不限）
    assert budgets.call_cap == settings.rag_agent_tool_call_cap == 3
    assert budgets.parallel == settings.rag_agent_parallel == 4
    assert scheme.orchestrator.registry.catalog == volume_catalog()
    manager.ingest_all([DOC_REIMBURSE])
    assert len(scheme) == 1
    result = scheme.retrieve_full(QUERY_REIMBURSE, top_k=3)
    assert result.hits and result.trace is not None


def test_llm_scenarios_for_agent_roles():
    """五角色各自命名 LLM 场景：qwen3.5-flash 且关闭思考（结构化 JSON 轻量决策）。"""
    profiles = {p["scenario"]: p for p in DEFAULT_PROFILES}
    for scenario in ("rag_agent_route", "rag_agent_plan", "rag_agent_grade",
                     "rag_agent_correct", "rag_agent_verify"):
        assert scenario in profiles, f"缺少场景 {scenario}"
        assert profiles[scenario]["model"] == "qwen3.5-flash"
        assert profiles[scenario]["params"]["enable_thinking"] is False


def test_budget_settings_defaults(settings):
    """Settings 暴露 agentic 预算治理配置（企业级资源口径可调）。"""
    assert settings.rag_agent_max_steps == 8
    assert settings.rag_agent_correction_rounds == 2
    assert settings.rag_agent_timeout_s == 90.0
    assert settings.rag_agent_token_budget == 0  # 暂放开（0=不限），排查多轮遗忘后应恢复预算值
    assert settings.rag_agent_tool_call_cap == 3
    assert settings.rag_agent_parallel == 4


# ---- runner 接入：agentic 循环内工具 vs 循环外前置 ----

async def test_runner_agentic_in_loop_tool_light_route(settings, sessions):
    """runner：agentic 前置走独立 classify 接口（轻量语义路由）——不进入完整链路
    （astream 不被前置调用）、不注入检索块；knowledge_retrieve 工具注入主循环工具集（L2）。"""
    class FakeScheme:
        id = "agentic"
        name = "Agentic RAG"

        def classify(self, query, context=None):  # noqa: ARG002
            return {"retrieval_need": True, "generation_mode": "citation", "reason": "单点事实"}

        async def astream(self, *args, **kwargs):  # noqa: ARG002
            raise AssertionError("agentic 前置不应调用完整链路 astream（路由已由 classify 独立完成）")

    class FakeRagManager:
        def resolve(self, rag_scheme):  # noqa: ARG002
            return FakeScheme()

    class FakeRegistry:
        def __init__(self):
            self.rag_manager = FakeRagManager()
            self.embeddings = None  # 无向量能力：记忆常驻注入/轮末巩固自动跳过

    captured: list[str] = []

    class RecordingLLM(FakeChatModel):
        def _record(self, messages):
            for m in messages:
                if isinstance(m, HumanMessage):
                    captured.append(str(m.content))

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self._record(messages)
            return super()._generate(messages, stop, run_manager, **kwargs)

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            self._record(messages)
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    runner = AgentRunner(
        settings,
        RecordingLLM(script=[AIMessage(content="按知识库来源回答。")]),
        FakeRegistry(),
        sessions,
    )
    async for _ in runner.stream(
        "s1", "报销发票流程", "react", [], "standard", "never",
        rag_scheme="agentic", rag_enabled=True,
    ):
        pass

    assert captured, "应有用户消息"
    latest = captured[-1]
    assert "生成策略" in latest, "agentic 前置应注入 generation_mode 策略提示"
    assert "知识库检索结果" not in latest, "agentic 前置不应注入检索块（检索由工具负责）"
    assert "报销发票流程" in latest, "前置不替换原文 query（指代消解在工具内完成）"
    # 主循环工具集应只含 knowledge_task（单一检索入口：合并后不并列 knowledge_retrieve）
    _, tools = runner._specs["s1"]
    names = {getattr(t, "name", None) for t in tools}
    assert "knowledge_task" in names, "agentic 应把 knowledge_task 注入主循环工具集"
    assert "knowledge_retrieve" not in names, "agentic 单一检索入口：不并列 knowledge_retrieve"


def test_scheme_classify_light_route_only(monkeypatch):
    """scheme.classify：只产出路由决策（不检索、不消解），恰消费一次路由 LLM。"""
    llm = FakeChatModel(script=[role_msg({
        "retrieval_need": True, "generation_mode": "comparison", "reason": "多实体对比",
    })])
    monkeypatch.setattr("app.rag.agentic.roles.get_chat_model", lambda scenario: llm)
    scheme = make_scheme(make_store([DOC_REIMBURSE]))
    decision = scheme.classify(QUERY_REIMBURSE)
    assert decision["retrieval_need"] is True
    assert decision["generation_mode"] == "comparison"
    assert decision["reason"] == "多实体对比"
    assert len(llm.script) == 0, "classify 应只消费一次路由 LLM（不触发 plan/verify 等角色）"


async def test_orchestrator_pre_route_skips_router(no_role_llm, monkeypatch):
    """pre_route 注入：route 节点跳过 RouterAgent（路由场景无脚本即证明未走 LLM），
    classify 事件直接采用前置决策，检索仍正常执行。"""
    monkeypatch.setattr(
        "app.rag.agentic.roles.get_chat_model",
        scripted_llm(**{
            "rag_agent_plan": FakeChatModel(script=[PLAN_MSG]),
            "rag_agent_grade": FakeChatModel(script=[GRADE_MSG]),
            "rag_agent_verify": FakeChatModel(script=[VERIFY_OK]),
        }),
    )
    orch = make_orchestrator(make_store([DOC_REIMBURSE]))
    pre = {"retrieval_need": True, "generation_mode": "comparison", "reason": "前置语义路由"}
    events = [ev async for ev in orch.astream(QUERY_REIMBURSE, k=3, pre_route=pre)]
    classify_done = next(e for e in events if e["type"] == "classify" and e["status"] == "done")
    # route 场景未注入脚本：若走 RouterAgent 会规则回退为 citation；得到 comparison 即证明复用了 pre_route
    assert classify_done["generation_mode"] == "comparison"
    assert classify_done["reason"] == "前置语义路由"
    retrieve = next(e for e in events if e["type"] == "retrieve")
    assert retrieve["hits"], "复用前置路由后检索仍正常执行"


async def test_rag_tool_reuses_pre_route(settings):
    """knowledge_retrieve：从 context_holder 读取前置路由并作为 pre_route 透传给完整链路。"""
    seen: dict = {}

    class FakeScheme:
        id = "agentic"
        name = "Agentic RAG"

        async def astream(self, query, top_k=None, context=None, seed_hits=None, pre_route=None):  # noqa: ARG002
            seen["pre_route"] = pre_route
            yield {"type": "classify", "status": "done", "generation_mode": "comparison"}
            yield {"type": "retrieve", "hits": [{"text": "发票报销时限为费用发生后三个月内。", "score": 0.9}]}
            yield {"type": "answerability", "verdict": {"answerable": True}}

    route = {"retrieval_need": True, "generation_mode": "comparison", "reason": "前置语义路由"}
    holder = {"recent": None, "route": route}
    tool = make_knowledge_retrieve_tool(FakeScheme(), settings, None, "s1", {}, holder)
    result = await tool.ainvoke({"query": "报销发票"})
    assert seen["pre_route"] == route, "工具应把前置路由作为 pre_route 传给完整链路（跳过二次路由）"
    assert "三个月内" in result, "检索命中应正常返回"


def test_rag_block_payload_budgeted_clip():
    """rag_block_payload：长命中按预算句末截断（总块 ≤ _MAX_BLOCK_CHARS），短命中不截断，
    来源编号保留供引用——检索结果不再全量进上下文。"""
    long = "差旅报销须在15日内提交，逾期作废；海外需三重审批并提前5日报备。" * 120
    short = "日常报销仅需发票。"
    ctx = {
        "name": "Agentic RAG",
        "hits": [
            {"text": long, "metadata": {"source": "汇编.md"}},
            {"text": short, "metadata": {"source": "汇编.md"}},
        ],
    }
    out = rag_block_payload(ctx, False, "comparison")
    assert len(out) <= _MAX_BLOCK_CHARS, "块应受预算约束，避免触发落盘"
    assert "15日内提交" in out, "头部要点保留"
    assert "日常报销仅需发票。" in out, "短命中不截断"
    assert "[1]" in out and "[2]" in out, "来源编号保留供引用"
    assert "…" in out, "长命中应被句末截断"


async def test_rag_tool_returns_budgeted_block(settings):
    """knowledge_retrieve：长命中返回要点化块（正文 ≤ _MAX_BLOCK_CHARS）+
    结构化状态行，总量仍低于上下文落盘阈值，落盘几乎不触发。"""
    class FakeScheme:
        id = "agentic"
        name = "Agentic RAG"

        async def astream(self, query, top_k=None, context=None, seed_hits=None, pre_route=None):  # noqa: ARG002
            yield {"type": "classify", "status": "done", "generation_mode": "comparison"}
            yield {"type": "retrieve", "hits": [
                {"text": "差旅报销须在15日内提交，逾期作废；海外需三重审批并提前5日报备。" * 60},
            ]}
            yield {"type": "answerability", "verdict": {"answerable": True}}

    tool = make_knowledge_retrieve_tool(FakeScheme(), settings, None, "s1", {}, {})
    result = await tool.ainvoke({"query": "差旅和日常报销流程差异"})
    block, status = result.rsplit("\n", 1)
    assert len(block) <= _MAX_BLOCK_CHARS, "要点化正文应低于预算上限"
    assert len(result) < 3000, "正文+状态行总量仍低于上下文落盘阈值，落盘几乎不触发"
    assert "知识库检索结果" in block
    assert "15日内提交" in block, "头部要点保留供模型组织回答"
    assert status.startswith("【检索状态】") and "充分性=充分" in status, "工具返回应附结构化状态行"


def test_orchestrator_pipeline_full_link(no_role_llm):
    """检索链路完整明细（pipeline）：触发条件/查询变换/每路策略/筛选/排序 齐备。

    覆盖需求1：工具调用过程返回完整检索步骤——触发条件（trigger）、查询向量生成方法
    （query_pipeline）、每路检索策略（strategy，含命中分数分布）、文档筛选规则（filters）、
    最终结果排序依据（ranking）；trace 与 retrieve 事件共用同一 pipeline。
    """
    orch = make_orchestrator(make_store([DOC_REIMBURSE]))
    result = orch.run(QUERY_REIMBURSE, k=3)
    p = result.pipeline
    # 触发条件（trigger）
    assert p["trigger"]["retrieval_need"] is True
    assert p["trigger"]["mode"] == "citation"
    assert p["trigger"]["reason"], "路由决策理由（trigger.reason）应记录"
    # 查询向量生成方法（query_pipeline）
    assert p["query_pipeline"]["embedding"] in ("dense", "dense+sparse")
    assert p["query_pipeline"]["hyde"] is False, "规则回退不生成 HyDE 假想文档"
    # 每路检索策略（strategy）
    assert p["strategy"], "每路检索明细非空"
    s = p["strategy"][0]
    assert s["tool"] == ACTION_HYBRID and s["query"] == QUERY_REIMBURSE
    assert s["hits"] > 0 and s["scores"], "命中数与分数分布应记录"
    assert s["query_pipeline"], "查询变换明细应记录（检索类型/向量体系）"
    # 文档筛选规则（filters）
    assert any(f["name"] == "grade" for f in p["filters"]), "评审筛选统计应记录"
    # 最终结果排序依据（ranking）
    assert p["ranking"]["fusion"]["method"] == "RRF(K=60)"
    assert p["ranking"]["rerank"]["model"], "重排模型/前后保留数应记录"
    # trace 与 retrieve 事件共用同一 pipeline
    assert result.trace["pipeline"] == p


def test_rag_block_payload_keeps_relevant_sentence_behind_cut():
    """rag_block_payload：query 相关句（落在硬截断点之后）被保留，不从头硬切。

    覆盖需求2：直接截断会把「关键条款/差异点落在截断点之后而丢失」；传 query 后按
    相关度选取句子、原文顺序输出，关键句即使在截断点之后也能进入最终上下文。
    """
    filler = "本规定系根据公司管理需要制定，适用于全体在职员工，具体条款由人事部门负责解释。" * 80
    key = "发票报销必须附上发票原件，逾期未报销的发票一律作废。"
    ctx = {
        "name": "Agentic RAG",
        "hits": [{"text": filler + key, "metadata": {"source": "汇编.md"}}],
    }
    out = rag_block_payload(ctx, False, "citation", "报销发票")
    assert "逾期未报销的发票一律作废" in out, "关键句落在截断点后也应保留"
    assert "…" in out, "超长命中应标注截断"


# ---- P0 层间结构化契约：confidence / cost ----

def test_orchestrator_contract_confidence_cost(no_role_llm):
    """层间结构化契约：run 结果携带 confidence（确定性口径）+ cost（token/调用/时延）。

    规则回退路径：facts=[query] 全覆盖可答 → 基准 1.0，回退降权 0.1 → 0.9；
    cost.calls ≥ 1（首发检索），token 账本与 latency_ms 齐备——外层可编程消费。
    """
    result = make_orchestrator(make_store([DOC_REIMBURSE])).run(QUERY_REIMBURSE, k=3)
    assert result.answerable is True
    assert result.confidence == 0.9
    assert result.cost["tokens"] == {"prompt": 0, "completion": 0}
    assert result.cost["calls"] >= 1
    assert result.cost["latency_ms"] > 0
    assert result.verdict["answerable"] is True


async def test_scheme_astream_contract_fields(no_role_llm):
    """流式事件契约：retrieve / answerability 事件携带 confidence 与 cost（前端可展示）。"""
    events = [ev async for ev in make_scheme(make_store([DOC_REIMBURSE])).astream(QUERY_REIMBURSE, 3)]
    retrieve = next(e for e in events if e["type"] == "retrieve")
    answerability = next(e for e in events if e["type"] == "answerability")
    assert retrieve["confidence"] == 0.9 and retrieve["cost"]["calls"] >= 1
    assert answerability["confidence"] == 0.9
    assert "tokens" in answerability["cost"] and "latency_ms" in answerability["cost"]


def test_scheme_retrieve_full_passes_contract(no_role_llm):
    """契约透传：retrieve_full → RetrieveResult 携带 confidence/cost（其他方案默认 None）。"""
    rr = make_scheme(make_store([DOC_REIMBURSE])).retrieve_full(QUERY_REIMBURSE)
    assert rr.confidence == 0.9
    assert rr.cost["calls"] >= 1
    assert rr.answerability["answerable"] is True


async def test_rag_tool_contract_passthrough(settings):
    """工具层契约旁路：结构化元数据写入 holder["rag_state"]（程序化消费，零文本解析），
    文本返回附状态行（模型直接理解充分性/缺口/置信度）。"""
    class ContractScheme:
        id = "agentic"
        name = "Agentic RAG"

        async def astream(self, query, top_k=None, context=None, seed_hits=None, pre_route=None):  # noqa: ARG002
            yield {"type": "classify", "status": "done", "generation_mode": "citation"}
            yield {"type": "retrieve", "hits": [{"text": "王刚是研发部部门主管。", "score": 0.9}]}
            yield {
                "type": "answerability",
                "verdict": {"answerable": False, "missing_facts": ["王刚的在岗工龄"]},
                "confidence": 0.6,
                "cost": {"tokens": {"prompt": 10, "completion": 5}, "calls": 2, "latency_ms": 350},
            }

    holder: dict = {"recent": None}
    tool = make_knowledge_retrieve_tool(ContractScheme(), settings, None, "s1", {}, holder)
    result = await tool.ainvoke({"query": "王刚年假"})
    # 旁路结构化契约（程序化消费）
    rag_state = holder["rag_state"]
    assert rag_state["verdict"]["answerable"] is False
    assert rag_state["verdict"]["missing_facts"] == ["王刚的在岗工龄"]
    assert rag_state["confidence"] == 0.6
    assert rag_state["cost"]["calls"] == 2
    assert rag_state["hits"] == 1
    # 文本状态行（模型直接理解）
    assert "充分性=不足" in result
    assert "缺失事实=王刚的在岗工龄" in result
    assert "置信度=0.60" in result


# ---- P1 检索任务编排层：拆解器 / 任务图状态机 / knowledge_task 工具 ----

@pytest.fixture
def no_task_llm(monkeypatch):
    """任务层 LLM 全部规则回退（模拟无 Key 离线环境）。"""
    monkeypatch.setattr("app.rag.task.decomposer.get_chat_model", lambda scenario: None)


def task_llm_msg(payload: dict) -> AIMessage:
    """构造任务层 LLM 决策 JSON 消息（拆解器脚本）。"""
    return AIMessage(content=json.dumps(payload, ensure_ascii=False))


def make_contract(answerable=True, hits=None, missing=None, confidence=1.0, note=""):
    """构造单节点内层契约（NodeResult 载荷）。"""
    return {
        "hits": hits or [],
        "verdict": {"answerable": answerable},
        "missing_facts": ([] if answerable else missing) or ([] if answerable else ["缺失事实"]),
        "confidence": confidence,
        "cost": {"tokens": {"prompt": 1, "completion": 1}, "calls": 1, "latency_ms": 1.0},
        "note": note,
    }


class FakeRunNode:
    """脚本化内层闭环消费者：记录执行顺序与黑板 seed，按节点 id 返回契约。"""

    def __init__(self, contracts: dict[str, dict]):
        self.contracts = contracts
        self.order: list[str] = []
        self.seed_seen: dict[str, list[dict]] = {}

    async def __call__(self, node: TaskNode, seed: list[dict] | None):
        self.order.append(node.id)
        self.seed_seen[node.id] = list(seed or [])
        return self.contracts.get(
            node.id,
            {"hits": [], "verdict": {"answerable": False}, "missing_facts": [node.query],
             "confidence": 0.0, "cost": {"tokens": {"prompt": 0, "completion": 0}, "calls": 0, "latency_ms": 0.0}, "note": ""},
        )


def test_decomposer_llm_multi_node_with_deps(monkeypatch):
    """拆解器 LLM 路径：复合问题 → 多节点 DAG（含链式依赖声明，重编号 n1..nN）。"""
    payload = {
        "nodes": [
            {"id": "n1", "query": "张三 所在部门", "deps": [], "reason": "先解析部门"},
            {"id": "n2", "query": "研发部 在职人数", "deps": ["n1"], "reason": "再查人数"},
        ],
        "reason": "链式：先部门后人数",
    }
    monkeypatch.setattr(
        "app.rag.task.decomposer.get_chat_model",
        lambda scenario: FakeChatModel(script=[task_llm_msg(payload)]),
    )
    nodes, thought, note = TaskDecomposer(max_nodes=4).decompose("张三所在部门在职人数")
    assert len(nodes) == 2
    assert nodes[0]["id"] == "n1" and nodes[0]["deps"] == []
    assert nodes[1]["id"] == "n2" and nodes[1]["deps"] == ["n1"]
    assert thought == "链式：先部门后人数"
    assert note == ""


def test_decomposer_rule_fallback_single_node(no_task_llm):
    """复合问题 + 无 LLM → 规则回退：单节点 = 原查询（任务不中断，note 记录回退原因）。"""
    nodes, thought, note = TaskDecomposer(max_nodes=4).decompose("张三所在部门在职人数")
    assert nodes == [{"id": "n1", "query": "张三所在部门在职人数", "deps": [], "reason": "规则回退：单节点"}]
    assert note == "拆解规则回退"


def test_decomposer_rule_simple_skips_llm(monkeypatch):
    """单一入口规则粗筛：简单问题（短 + 无复合标记）不调拆解 LLM，单节点直通内层。"""
    calls: list[str] = []

    def noop_llm(scenario):
        calls.append(scenario)
        return None  # 模拟无 Key：记录调用、不产出，调用方回退规则

    monkeypatch.setattr("app.rag.task.decomposer.get_chat_model", noop_llm)
    nodes, thought, note = TaskDecomposer(max_nodes=4).decompose("报销发票")
    assert calls == [], "规则粗筛短路，拆解 LLM 零调用"
    assert nodes == [{"id": "n1", "query": "报销发票", "deps": [], "reason": "规则判定简单问题：单节点直通"}]
    assert note == "规则判定简单问题"
    # 疑似复合（含并列标记）仍走 LLM 拆解
    nodes2, _, note2 = TaskDecomposer(max_nodes=4).decompose("报销发票和时限")
    assert calls == [SCENARIO_DECOMPOSE], "含复合标记 → 调拆解 LLM"
    assert nodes2[0]["query"] == "报销发票和时限" and note2 == "拆解规则回退", "LLM 无产出 → 规则回退"


def test_decomposer_cap_and_dedup(monkeypatch):
    """拆解器护栏：超 max_nodes 封顶 + 重复查询去重 + 空查询剔除。"""
    payload = {
        "nodes": [
            {"id": "x1", "query": "A", "deps": []},
            {"id": "x2", "query": "A", "deps": []},
            {"id": "x3", "query": "  ", "deps": []},
            {"id": "x4", "query": "B", "deps": []},
            {"id": "x5", "query": "C", "deps": []},
        ],
        "reason": "",
    }
    monkeypatch.setattr(
        "app.rag.task.decomposer.get_chat_model",
        lambda scenario: FakeChatModel(script=[task_llm_msg(payload)]),
    )
    nodes, _, _ = TaskDecomposer(max_nodes=3).decompose("A和B与C的对比")
    assert [n["id"] for n in nodes] == ["n1"]
    assert [n["query"] for n in nodes] == ["A"], "重复与空查询剔除后仅剩 A（B/C 被 max_nodes 截断）"


def test_executor_single_node_complete(no_task_llm):
    """任务图：单节点任务 resolve → completion=complete，证据合并入库。"""
    runner = FakeRunNode({"n1": make_contract(answerable=True, hits=[{"text": "报销需附发票", "score": 0.9}])})
    result = TaskExecutor(runner).run("报销发票")
    assert result.completion == TC_COMPLETE
    assert result.resolved_count == 1 and result.gap_count == 0
    assert len(result.evidence) == 1 and result.evidence[0]["text"] == "报销需附发票"
    assert result.cost["calls"] == 1


def test_executor_dag_order_and_seed(no_task_llm, monkeypatch):
    """任务图 DAG：依赖节点先执行；黑板证据池跨节点 seed 复用（n2 可见 n1 命中）。"""
    payload = {
        "nodes": [
            {"id": "n1", "query": "张三 所在部门", "deps": [], "reason": "先查部门"},
            {"id": "n2", "query": "研发部 在职人数", "deps": ["n1"], "reason": "再查人数"},
        ],
        "reason": "链式",
    }
    monkeypatch.setattr(
        "app.rag.task.decomposer.get_chat_model",
        lambda scenario: FakeChatModel(script=[task_llm_msg(payload)]),
    )
    contracts = {
        "n1": make_contract(answerable=True, hits=[{"text": "张三在研发部。", "score": 0.9}]),
        "n2": make_contract(answerable=True, hits=[{"text": "研发部在职人数 120 人。", "score": 0.9}]),
    }
    runner = FakeRunNode(contracts)
    result = TaskExecutor(runner).run("张三所在部门的在职人数")
    assert runner.order == ["n1", "n2"], "依赖序执行：n1 先于 n2"
    assert runner.seed_seen["n2"] == [{"text": "张三在研发部。", "score": 0.9}], "黑板证据池跨节点复用"
    assert result.completion == TC_COMPLETE and result.resolved_count == 2
    assert [h["text"] for h in result.evidence] == ["张三在研发部。", "研发部在职人数 120 人。"]


def test_executor_gap_partial(no_task_llm, monkeypatch):
    """节点缺口：可答节点先行，缺口如实记录 → completion=partial（可答部分不丢）。"""
    payload = {
        "nodes": [
            {"id": "n1", "query": "报销发票", "deps": [], "reason": ""},
            {"id": "n2", "query": "报销时限", "deps": [], "reason": ""},
        ],
        "reason": "",
    }
    monkeypatch.setattr(
        "app.rag.task.decomposer.get_chat_model",
        lambda scenario: FakeChatModel(script=[task_llm_msg(payload)]),
    )
    contracts = {
        "n1": make_contract(answerable=True, hits=[{"text": "报销需附发票", "score": 0.9}]),
        "n2": make_contract(answerable=False, missing=["报销时限"], confidence=0.2),
    }
    runner = FakeRunNode(contracts)
    result = TaskExecutor(runner).run("报销发票与时限")
    assert result.completion == TC_PARTIAL
    assert result.resolved_count == 1 and result.gap_count == 1
    assert result.gaps[0]["node_id"] == "n2" and result.gaps[0]["missing_facts"] == ["报销时限"]
    assert len(result.evidence) == 1


def test_executor_inner_calls_budget(no_task_llm, monkeypatch):
    """任务账本护栏：max_inner_calls 触顶即终止（预算耗尽 ≠ 失败，note 说明原因）。"""
    payload = {
        "nodes": [
            {"id": "n1", "query": "A", "deps": [], "reason": ""},
            {"id": "n2", "query": "B", "deps": [], "reason": ""},
            {"id": "n3", "query": "C", "deps": [], "reason": ""},
        ],
        "reason": "",
    }
    monkeypatch.setattr(
        "app.rag.task.decomposer.get_chat_model",
        lambda scenario: FakeChatModel(script=[task_llm_msg(payload)]),
    )
    runner = FakeRunNode({"n1": make_contract(answerable=True)})
    result = TaskExecutor(runner, budgets=TaskBudgets(max_nodes=4, max_inner_calls=1)).run("A和B与C的对比")
    assert runner.order == ["n1"], "只允许触发 1 次内层闭环"
    assert result.trace["note"] == "任务内层触发已达上限"
    assert result.completion == TC_PARTIAL, "可答节点先行，剩余如实上报"


async def test_executor_astream_event_sequence(no_task_llm):
    """任务图事件序列：task_plan → task_node* → task_done（前端可展示任务轨迹）。"""
    runner = FakeRunNode({"n1": make_contract(answerable=True, hits=[{"text": "报销需附发票", "score": 0.9}])})
    events = [ev async for ev in TaskExecutor(runner).astream("报销发票")]
    assert [e["type"] for e in events] == ["task_plan", "task_node", "task_done"]
    plan = events[0]
    assert plan["type"] == "task_plan" and plan["nodes"][0]["id"] == "n1"
    node_ev = events[1]
    assert node_ev["node_id"] == "n1" and node_ev["state"] == NS_RESOLVED
    assert node_ev["confidence"] == 1.0 and node_ev["cost"]["calls"] == 1
    done = events[2]
    assert done["result"]["completion"] == TC_COMPLETE


async def test_knowledge_task_tool_passthrough(settings, no_task_llm):
    """knowledge_task 工具：任务编排 → 文本附任务状态行 + holder["task_state"] 旁路透传。"""
    class TaskScheme:
        id = "agentic"
        name = "Agentic RAG"

        async def astream(self, query, top_k=None, context=None, seed_hits=None, pre_route=None):  # noqa: ARG002
            yield {"type": "classify", "status": "done", "generation_mode": "citation"}
            yield {"type": "retrieve", "hits": [{"text": "报销需附发票，随报销单一并提交财务。", "score": 0.9}]}
            yield {
                "type": "answerability",
                "verdict": {"answerable": True, "missing_facts": []},
                "confidence": 0.9,
                "cost": {"tokens": {"prompt": 1, "completion": 1}, "calls": 1, "latency_ms": 100},
            }

    holder: dict = {}
    ledger = SessionLedger(max_inner_calls=5)
    emitted: list[dict] = []
    tool = make_knowledge_task_tool(TaskScheme(), settings, emitted.append, "s1", {}, holder, ledger)
    result = await tool.ainvoke({"query": "报销发票和时限"})
    task_state = holder["task_state"]
    assert task_state["completion"] == TC_COMPLETE
    assert task_state["resolved"] == 1 and task_state["gaps"] == 0
    assert "【检索状态】" in result and "任务完成度=complete" in result
    assert "报销需附发票" in result
    assert f"会话内层余量={ledger.remaining_inner()}/5" in result, "P3 会话账本余量进入状态行"
    assert ledger.inner_calls == 1, "任务内层消耗并入会话账本"
    # P4 统一事件协议：任务图事件 + 内层事件均携带 task_id/node_id（前端可按节点聚合轨迹）
    task_ids = {e["task_id"] for e in emitted if e["type"] == "task_plan"}
    assert len(task_ids) == 1
    tid = task_ids.pop()
    inner = [e for e in emitted if e["type"] in ("classify", "retrieve", "answerability")]
    assert inner, "节点执行内层事件被转发"
    assert all(e.get("task_id") == tid and e.get("node_id") == "n1" for e in inner)


# ---- P2 缺口策略中心：分类 / 决策表 / 改写重查回环 ----

def gap_msg(payload: dict) -> AIMessage:
    """构造缺口分类 LLM 决策 JSON 消息。"""
    return AIMessage(content=json.dumps(payload, ensure_ascii=False))


def gap_contract(answerable=False, missing=("缺失事实",), confidence=0.2, hits=None):
    """构造单节点缺口/可答契约（重查回环断言用）。"""
    return {
        "hits": hits or [],
        "verdict": {"answerable": answerable},
        "missing_facts": list(missing),
        "confidence": confidence,
        "cost": {"tokens": {"prompt": 1, "completion": 1}, "calls": 1, "latency_ms": 1.0},
        "note": "",
    }


class QueryScriptedRunNode:
    """按节点查询串返回契约（记录尝试序列），供改写重查回环断言。"""

    def __init__(self, by_query: dict[str, dict], default: dict | None = None):
        self.by_query = by_query
        self.default = default or gap_contract()
        self.calls: list[str] = []

    async def __call__(self, node: TaskNode, seed: list[dict] | None):
        self.calls.append(node.query)
        return self.by_query.get(node.query, self.default)


class StubClassifier:
    """固定分类结果的缺口分类器（决策表收敛直测用）。"""

    def __init__(self, decision: GapDecision):
        self.decision = decision

    def classify(self, query, missing_facts, evidence=None, ledger=None):
        return self.decision


def test_gap_classifier_llm_query_type(monkeypatch):
    """分类器 LLM 路径：query 类缺口 → rewrite 动作 + 改写查询（rag_task_gap 场景）。"""
    monkeypatch.setattr(
        "app.rag.task.decomposer.get_chat_model",
        lambda scenario: FakeChatModel(script=[gap_msg({"gap_type": "query", "rewrite_query": "发票 报销 时限", "reason": "换词重查"})]),
    )
    d = GapClassifier().classify("报销发票", ["发票报销时限"], ledger={"prompt": 0, "completion": 0})
    assert d.gap_type == "query" and d.action == "rewrite"
    assert d.rewrite_query == "发票 报销 时限"


def test_gap_classifier_rule_fallback(no_task_llm):
    """分类器规则回退：缺失项与查询强重叠 → 表达问题（拼词改写）；否则保守数据缺失。"""
    overlap = GapClassifier().classify("张三 部门", ["张三 所在部门"])
    assert overlap.gap_type == "query" and overlap.action == "rewrite"
    assert "张三 所在部门" in overlap.rewrite_query
    data = GapClassifier().classify("报销发票", ["报销时限"])
    assert data.gap_type == "data" and data.action == "report"
    assert data.note == "缺口分类规则回退"


async def test_gap_center_decision_table(no_task_llm):
    """决策表收敛：改写超次数上限 → 降级上报；跨域 → 降级上报 + 建议；低价值 → 接受。"""
    center = GapStrategyCenter()
    # 规则回退改写类：retries 未触顶 → 保留 rewrite
    d1 = await center.decide("报销发票", ["发票"], [], retries=0, max_retries=1)
    assert d1.action == "rewrite" and d1.rewrite_query
    # 改写类 retries 触顶 → 降级 report（note 记录次数上限）
    d2 = await center.decide("报销发票", ["发票"], [], retries=1, max_retries=1)
    assert d2.action == "report" and "次数达上限" in d2.note
    # 跨域 → 降级 report + 建议转交（本系统工具面无对应目标）
    cross = GapStrategyCenter(classifier=StubClassifier(GapDecision("cross_domain", "delegate", reason="计算类")))
    d3 = await cross.decide("报销发票", ["发票"], [], 0, 1)
    assert d3.action == "report" and "转交其它工具" in d3.note
    # 低价值 → accept（部分回答 + 标注，不阻塞）
    low = GapStrategyCenter(classifier=StubClassifier(GapDecision("low_value", "accept", reason="不影响结论")))
    d4 = await low.decide("报销发票", ["发票"], [], 0, 1)
    assert d4.action == "accept" and d4.gap_type == "low_value"


def test_executor_gap_rewrite_retry_resolves(no_task_llm):
    """缺口策略回环：首查缺口（表达问题）→ 改写重查 → 节点 resolve（attempt 证据并入黑板）。"""
    runner = QueryScriptedRunNode({
        "张三 部门": gap_contract(missing=["张三 所在部门"]),
        "张三 部门 张三 所在部门": gap_contract(answerable=True, hits=[{"text": "张三在研发部。", "score": 0.9}]),
    })
    result = TaskExecutor(runner).run("张三 部门")
    assert runner.calls == ["张三 部门", "张三 部门 张三 所在部门"], "改写重查触发第二次内层闭环"
    assert result.completion == TC_COMPLETE and result.resolved_count == 1
    assert [h["text"] for h in result.evidence] == ["张三在研发部。"]


def test_executor_gap_data_reported(no_task_llm):
    """数据缺失型缺口：如实上报不重查（仅 1 次内层触发），缺口带类型/动作/备注。"""
    runner = QueryScriptedRunNode({"报销发票": gap_contract(missing=["报销时限"])})
    result = TaskExecutor(runner).run("报销发票")
    assert runner.calls == ["报销发票"]
    assert result.completion == TC_CLARIFIED, "单节点缺口且无重查 → 如实上报追问"
    assert result.gaps[0]["gap_type"] == "data" and result.gaps[0]["action"] == "report"
    assert result.gaps[0]["missing_facts"] == ["报销时限"]


def test_executor_gap_rewrite_exhausted(no_task_llm):
    """改写重查达上限：如实上报（note 记录次数），不无限重试。"""
    runner = QueryScriptedRunNode({}, default=gap_contract(missing=["张三 所在部门"]))
    result = TaskExecutor(runner, budgets=TaskBudgets(max_nodes=4, max_retries=1)).run("张三 部门")
    assert len(runner.calls) == 2, "首查 + 1 次改写重查后停止"
    assert result.gaps[0]["gap_type"] == "query" and result.gaps[0]["action"] == "report"
    assert "次数达上限" in result.gaps[0]["note"]


async def test_executor_astream_task_retry_event(no_task_llm):
    """改写重查事件：task_retry 进入事件流（前端可见缺口分类决策轨迹）。"""
    runner = QueryScriptedRunNode({
        "张三 部门": gap_contract(missing=["张三 所在部门"]),
        "张三 部门 张三 所在部门": gap_contract(answerable=True, hits=[{"text": "张三在研发部。", "score": 0.9}]),
    })
    events = [ev async for ev in TaskExecutor(runner).astream("张三 部门")]
    types = [e["type"] for e in events]
    assert types == ["task_plan", "task_retry", "task_node", "task_done"]
    retry = events[1]
    assert retry["node_id"] == "n1" and retry["rewrite_query"] == "张三 部门 张三 所在部门"
    assert retry["gap_type"] == "query" and retry["retries"] == 1
    # P4 事件协议：task_node 携带节点真实 missing_facts / gap note
    node_ev = events[2]
    assert node_ev["state"] == "resolved" and node_ev["missing_facts"] == []
    assert node_ev["task_id"] == events[0]["task_id"] and node_ev["node_id"] == "n1"


async def test_executor_astream_gap_node_event(no_task_llm):
    """缺口节点事件：task_node 如实携带 missing_facts 与缺口 note（前端可展示缺口）。"""
    runner = QueryScriptedRunNode({"报销发票": gap_contract(missing=["报销时限"])})
    events = [ev async for ev in TaskExecutor(runner).astream("报销发票")]
    node_ev = events[1]
    assert node_ev["type"] == "task_node" and node_ev["state"] == "gap"
    assert node_ev["missing_facts"] == ["报销时限"]
    assert node_ev["note"] == "缺口分类规则回退"  # 规则回退 data 缺口的 note 如实透传


# ---- P3 会话账本：跨任务累计，与任务账本叠加 ----

def test_session_ledger_unit():
    """会话账本记账与触顶判定：merge 累计、remaining/over_inner/over_token/exhausted。"""
    ledger = SessionLedger(max_inner_calls=3, token_budget=10)
    assert ledger.remaining_inner() == 3 and not ledger.exhausted()
    ledger.merge({"prompt": 4, "completion": 2}, inner_calls=1)
    assert ledger.inner_calls == 1 and ledger.tokens == {"prompt": 4, "completion": 2}
    assert not ledger.over_inner() and not ledger.over_token()
    ledger.merge({"prompt": 5, "completion": 0}, inner_calls=2)
    assert ledger.inner_calls == 3 and ledger.over_inner() and ledger.exhausted()
    assert ledger.remaining_inner() == 0
    # token 账本：0=不限
    assert not SessionLedger(token_budget=0).over_token()


def test_executor_session_ledger_cross_task_cap(no_task_llm, monkeypatch):
    """会话账本叠加：任务 A 消耗会话余量 → 任务 B 触发前触顶即止（跨任务防级联超支）。"""
    payload = {
        "nodes": [
            {"id": "n1", "query": "A", "deps": [], "reason": ""},
            {"id": "n2", "query": "B", "deps": [], "reason": ""},
            {"id": "n3", "query": "C", "deps": [], "reason": ""},
        ],
        "reason": "",
    }
    monkeypatch.setattr(
        "app.rag.task.decomposer.get_chat_model",
        lambda scenario: FakeChatModel(script=[task_llm_msg(payload), task_llm_msg(payload)]),
    )
    ledger = SessionLedger(max_inner_calls=2)
    runner = FakeRunNode({"n1": make_contract(), "n2": make_contract()})
    # 任务 A：会话余量 2 → 只执行 2 个节点，第 3 节点因会话预算耗尽转缺口
    ra = TaskExecutor(runner, session_ledger=ledger).run("A和B与C的对比")
    assert runner.order == ["n1", "n2"], "会话余量只够触发 2 次内层"
    assert ra.completion == TC_PARTIAL and ledger.inner_calls == 2
    assert ra.trace["session_inner_calls"] == 2
    # 任务 B：会话已耗尽 → 0 次内层触发，如实上报
    rb = TaskExecutor(runner, session_ledger=ledger).run("D和E与F的对比")
    assert runner.order == ["n1", "n2"], "会话耗尽后不再触发内层"
    assert rb.completion == TC_CLARIFIED and "会话预算耗尽" in rb.trace["note"]


def test_executor_session_ledger_token_cap(no_task_llm, monkeypatch):
    """会话账本 token 触顶：累计 token 达会话上限 → 后续任务如实上报。"""
    payload = {
        "nodes": [{"id": "n1", "query": "A", "deps": [], "reason": ""}],
        "reason": "",
    }
    monkeypatch.setattr(
        "app.rag.task.decomposer.get_chat_model",
        lambda scenario: FakeChatModel(script=[task_llm_msg(payload), task_llm_msg(payload)]),
    )
    ledger = SessionLedger(token_budget=2)
    runner = FakeRunNode({"n1": make_contract()})
    r1 = TaskExecutor(runner, session_ledger=ledger).run("A和B的对比")
    assert r1.completion == TC_COMPLETE and ledger.over_token()
    r2 = TaskExecutor(runner, session_ledger=ledger).run("D和E的对比")
    assert r2.completion == TC_CLARIFIED and "会话预算耗尽" in r2.trace["note"]
