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
from langchain_core.messages import AIMessage

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
