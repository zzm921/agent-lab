"""Modular RAG 方案测试：结构化路由决策 + 执行计划编排（分解/多路召回/压缩），全程离线。

使用 FakeEmbeddings（无 api_key → 词法重排回退）、FakeChatModel（脚本化路由/分解输出）
与测试桩 StubClassifier（替代已移除的规则路由，保证路由确定性）、RuleQueryDecomposer
等确定性规则实现，不联网、不依赖 Key。
"""
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.runner import AgentRunner
from app.core.errors import ConfigError
from app.llm.fake_model import FakeChatModel, FakeEmbeddings
from app.memory.stores.memory_store import MemoryStore
from app.rag.manager import RagManager
from app.rag.retrieval.answerability import (
    ESCALATE,
    ANSWER,
    CLARIFY,
    AnswerabilityVerdict,
    LLMAnswerabilityVerifier,
    RuleAnswerabilityVerifier,
)
from app.rag.retrieval.context_compress import ExtractiveContextCompressor
from app.rag.retrieval.iterative_retrieval import (
    LLMMultiHopRetriever,
    HopPlan,
    PlanExecuteRetriever,
    PlanStep,
    RuleMultiHopRetriever,
    VerifyResult,
)
from app.rag.retrieval.planner import LLMMultiHopPlanner, RuleMultiHopPlanner
from app.rag.retrieval.reranker import LexicalReranker
from app.rag.retrieval.verifier import LLMMultiHopVerifier, RuleMultiHopVerifier
from app.rag.routing.classifier import (
    CITATION,
    COMPARISON,
    DECOMPOSE,
    DIRECT,
    HYBRID,
    MULTIHOP,
    MULTI_RECALL,
    REWRITE,
    SIMPLE,
    VECTOR,
    LLMQueryClassifier,
    RouteDecision,
)
from app.rag.routing.deictic_resolver import LLMDeicticResolver, RuleDeicticResolver
from app.rag.routing.query_decompose import RuleQueryDecomposer
from app.rag.routing.query_rewrite import RuleQueryRewriter
from app.rag.schemes.modular import ExecutionPlan, ModularRagScheme


class StubClassifier:
    """测试桩：替代已移除的规则路由，按测试查询确定性返回路由决策（保持离线确定性）。"""

    def classify(self, query: str) -> RouteDecision:
        if "你好" in query or "谢谢" in query:
            return RouteDecision(
                retrieval_need=False, retrieval_mode=VECTOR, complexity=SIMPLE,
                generation_mode=DIRECT, confidence=1.0, reason="寒暄，无需检索",
            )
        if "区别" in query or "对比" in query:
            return RouteDecision(
                retrieval_need=True, retrieval_mode=MULTI_RECALL, complexity=DECOMPOSE,
                generation_mode=COMPARISON, confidence=0.9, reason="多实体对比",
            )
        if "这个" in query or "它" in query:
            return RouteDecision(
                retrieval_need=True, retrieval_mode=HYBRID, complexity=REWRITE,
                generation_mode=CITATION, confidence=0.8, reason="含指代，需改写",
            )
        if "流程" in query or "领导" in query:
            return RouteDecision(
                retrieval_need=True, retrieval_mode=MULTI_RECALL, complexity=MULTIHOP,
                generation_mode=CITATION, confidence=0.8, reason="多跳/流程",
            )
        return RouteDecision(
            retrieval_need=True, retrieval_mode=HYBRID, complexity=SIMPLE,
            generation_mode=CITATION, confidence=0.9, reason="单点事实",
        )


class StubDeicticResolver:
    """测试桩：确定性指代消解——查询含指代词且上下文提到「王刚」时替换为具体实体。"""

    def resolve(self, query: str, context: str | None) -> str:
        if context and "王刚" in context and re.search(r"他|她|它", query):
            return re.sub(r"他|她|它", "王刚", query)
        return query


def make_modular(settings, store=None, **kw) -> ModularRagScheme:
    """构造 modular 方案：未指定 store 时用内存回退，各模块可注入（默认确定性实现/测试桩）。"""
    if store is None:
        store = MemoryStore(FakeEmbeddings(), collection="knowledge_modular")
    kw.setdefault("classifier", StubClassifier())
    kw.setdefault("deictic", StubDeicticResolver())
    kw.setdefault("rewriter", RuleQueryRewriter())
    kw.setdefault("reranker", LexicalReranker())
    kw.setdefault("decomposer", RuleQueryDecomposer())
    kw.setdefault("compressor", ExtractiveContextCompressor())
    kw.setdefault("multi_hop", PlanExecuteRetriever(RuleMultiHopPlanner(), RuleMultiHopVerifier()))
    kw.setdefault("answerability", RuleAnswerabilityVerifier())
    return ModularRagScheme(FakeEmbeddings(), store, top_k=3, **kw)


# ---- 语义路由（纯 LLM：不再有规则路由） ----

def test_llm_classifier_relation_identity_simple(monkeypatch):
    """LLM 路由：只问关系实体本身（「张三的领导是谁」）→ 单跳 simple，不进入多跳（修复误检回归）。"""
    llm = FakeChatModel(
        script=[
            AIMessage(
                content='{"retrieval_need": true, "retrieval_mode": "vector", '
                '"complexity": "simple", "generation_mode": "citation", '
                '"confidence": 0.95, "reason": "只问关系实体本身，单跳直接检索"}'
            )
        ]
    )
    monkeypatch.setattr("app.rag.routing.classifier.get_chat_model", lambda scenario: llm)
    decision = LLMQueryClassifier().classify("张三的领导是谁")
    assert decision.retrieval_need is True
    assert decision.complexity == SIMPLE
    assert decision.complexity != MULTIHOP
    assert decision.retrieval_mode == VECTOR


def test_llm_classifier_relation_attribute_multihop(monkeypatch):
    """LLM 路由：关系词后查属性（「张三的领导有几天年假」）→ 多跳 multihop。"""
    llm = FakeChatModel(
        script=[
            AIMessage(
                content='{"retrieval_need": true, "retrieval_mode": "multi_recall", '
                '"complexity": "multihop", "generation_mode": "citation", '
                '"confidence": 0.85, "reason": "实体链多跳，先定位领导再查年假"}'
            )
        ]
    )
    monkeypatch.setattr("app.rag.routing.classifier.get_chat_model", lambda scenario: llm)
    decision = LLMQueryClassifier().classify("张三的领导有几天年假")
    assert decision.retrieval_mode == MULTI_RECALL
    assert decision.complexity == MULTIHOP


def test_llm_classifier_requires_llm(monkeypatch):
    """纯 LLM 路由：未配置聊天模型（get_chat_model 返回 None）时直接报错，无规则兜底。"""
    monkeypatch.setattr("app.rag.routing.classifier.get_chat_model", lambda scenario: None)
    with pytest.raises(ConfigError):
        LLMQueryClassifier().classify("张三的领导是谁")


def test_llm_classifier_invalid_enum_raises(monkeypatch):
    """纯 LLM 路由：LLM 输出非法枚举/不可解析时直接报错（不再回退规则路由）。"""
    llm = FakeChatModel(script=[AIMessage(content='{"retrieval_mode": "自造路径"}')])
    monkeypatch.setattr("app.rag.routing.classifier.get_chat_model", lambda scenario: llm)
    with pytest.raises(ConfigError):
        LLMQueryClassifier().classify("发票什么时候交")


def test_llm_classifier_parses_decision(monkeypatch):
    """LLM 路由：按场景懒取模型解析结构化 JSON 路由决策（五维度）。"""
    llm = FakeChatModel(
        script=[
            AIMessage(
                content='{"retrieval_need": true, "retrieval_mode": "multi_recall", '
                '"complexity": "decompose", "generation_mode": "comparison", '
                '"confidence": 0.9, "reason": "多实体对比"}'
            )
        ]
    )
    monkeypatch.setattr("app.rag.routing.classifier.get_chat_model", lambda scenario: llm)
    decision = LLMQueryClassifier().classify("分析考勤和年假的区别")
    assert decision.retrieval_mode == MULTI_RECALL
    assert decision.complexity == DECOMPOSE
    assert decision.generation_mode == COMPARISON
    assert decision.confidence == 0.9
    assert decision.reason


# ---- 指代消解（Deictic Resolution） ----

def test_llm_deictic_resolver_replaces_pronoun(monkeypatch):
    """LLM 指代消解：结合会话上下文把「他」替换为具体实体（王刚）。"""
    llm = FakeChatModel(script=[AIMessage(content="王刚的年假有多少天")])
    monkeypatch.setattr("app.rag.routing.deictic_resolver.get_chat_model", lambda scenario: llm)
    resolver = LLMDeicticResolver()
    resolved = resolver.resolve("他的年假有多少天", "用户: 张三的领导是谁\n助手: 王刚")
    assert resolved == "王刚的年假有多少天"


def test_llm_deictic_resolver_no_context_passthrough(monkeypatch):
    """无上下文时不做 LLM 调用，原样返回（避免无谓调用）。"""
    resolver = LLMDeicticResolver()
    assert resolver.resolve("他的年假有多少天", None) == "他的年假有多少天"


def test_rule_deictic_resolver_noop():
    """规则兜底：指代消解属语义判定，不做规则猜测，原样返回。"""
    resolver = RuleDeicticResolver()
    assert resolver.resolve("他的年假有多少天", "用户: 张三的领导是谁\n助手: 王刚") == "他的年假有多少天"


def test_deictic_prompt_prioritizes_assistant_answer_entity(monkeypatch):
    """指代消解 prompt：明确指示优先取「上一轮助手回答中给出/确定的实体」（答案对象）作为指代对象，
    避免把「他的年假有多少天」中的「他」误指为上轮问题主语「张三」。"""
    class RecordingLLM:
        def __init__(self):
            self.system = None

        def invoke(self, messages, *args, **kwargs):  # noqa: ARG002
            self.system = messages[0].content
            return AIMessage(content="王刚的年假有多少天")

    recorder = RecordingLLM()
    monkeypatch.setattr("app.rag.routing.deictic_resolver.get_chat_model", lambda scenario: recorder)
    resolver = LLMDeicticResolver()
    resolved = resolver.resolve(
        "他的年假有多少天",
        "用户: 张三的领导是谁\n助手: 根据知识库检索结果，张三的领导是王刚。",
    )
    assert resolved == "王刚的年假有多少天"
    assert "给出/确定" in recorder.system and "答案对象" in recorder.system


async def test_recent_context_includes_ai_message():
    """_recent_context：历史含助手 AI 回复时应拼接为「用户/助手」回合上下文
    （回归：runner.py 曾漏导入 AIMessage，历史含 AI 消息时抛 NameError 导致取不到上下文）。"""
    class FakeSnap:
        values = {
            "messages": [
                SystemMessage(content="系统提示"),
                HumanMessage(content="张三的领导是谁"),
                AIMessage(content="根据知识库检索结果，张三的领导是王刚。"),
            ]
        }

    class FakeGraph:
        async def aget_state(self, config):  # noqa: ARG002
            return FakeSnap()

    ctx = await AgentRunner._recent_context(FakeGraph(), {})
    assert ctx is not None
    assert "用户: 张三的领导是谁" in ctx
    assert "助手: 根据知识库检索结果，张三的领导是王刚。" in ctx


async def test_recent_context_none_when_no_messages():
    """_recent_context：无历史消息（新会话首轮）时返回 None，供 modular 指代消解按无上下文处理。"""
    class EmptySnap:
        values = {"messages": []}

    class FakeGraph:
        async def aget_state(self, config):  # noqa: ARG002
            return EmptySnap()

    assert await AgentRunner._recent_context(FakeGraph(), {}) is None


async def test_runner_injects_resolved_query_to_llm(settings, sessions):
    """runner：指代消解后注入给主 LLM 的用户消息应为消解后 query，而非含指代词的原文
    （回归：曾注入原文「他…」，主 LLM 二次解析把「他」误指回上轮问题主语「张三」，答非所问）。"""
    class FakeScheme:
        name = "模块化 RAG"

        async def astream(self, query, top_k=None, context=None):  # noqa: ARG002
            yield {"type": "rewrite", "query": query, "scheme": "modular",
                   "rewrites": ["王刚有多少天年假"], "reason": "指代消解"}
            yield {"type": "classify", "query": "王刚有多少天年假", "scheme": "modular",
                   "retrieval_need": True, "retrieval_mode": "vector", "complexity": "simple",
                   "generation_mode": "citation", "confidence": 0.95, "reason": "单点事实"}
            yield {"type": "retrieve", "query": "王刚有多少天年假", "scheme": "modular",
                   "hits": [{"text": "员工年假按工龄计算：满一年5天起，王刚作为部门主管年假按标准执行。", "score": 0.5}]}

    class FakeRagManager:
        def resolve(self, rag_scheme):  # noqa: ARG002
            return FakeScheme()

    class FakeRegistry:
        def __init__(self):
            self.rag_manager = FakeRagManager()

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
        RecordingLLM(script=[AIMessage(content="王刚作为部门主管年假按标准执行，满一年5天起。")]),
        FakeRegistry(),
        sessions,
    )
    async for _ in runner.stream(
        "s1", "他有多少天年假", "react", [], "standard", "never",
        rag_scheme="modular", rag_enabled=True,
    ):
        pass

    injected = [c for c in captured if "知识库检索结果" in c]
    assert injected, "应有注入检索结果的用户消息"
    latest = injected[-1]
    assert "王刚有多少天年假" in latest
    assert "他有多少天年假" not in latest, "注入给主 LLM 的消息不应再含指代词"


# ---- 执行计划：路由决策 → 模块组合 ----

def test_build_plan_no_retrieval(settings):
    """不检索：执行计划 need_retrieval=False，无任何检索/后处理模块。"""
    scheme = make_modular(settings)
    plan = scheme._build_plan(scheme.classifier.classify("你好"))
    assert isinstance(plan, ExecutionPlan)
    assert plan.need_retrieval is False
    assert plan.pre_retrieval == [] and plan.retrieval == [] and plan.post_retrieval == []
    assert plan.generation_strategy == DIRECT


def test_build_plan_decompose_full_chain(settings):
    """对比/多实体：计划 = 分解 → 多路召回 → 重排 → 压缩，生成策略=对比。"""
    scheme = make_modular(settings)
    plan = scheme._build_plan(scheme.classifier.classify("出差和报销有什么区别"))
    assert plan.need_retrieval is True
    assert [m.name for m in plan.pre_retrieval] == ["decompose"]
    assert [m.name for m in plan.retrieval] == ["multi_recall"]
    assert [m.name for m in plan.post_retrieval] == ["rerank", "compress"]
    assert plan.generation_strategy == COMPARISON


def test_build_plan_deictic_rewrite_hybrid(settings):
    """指代/歧义：计划 = 改写 → 混合检索 → 压缩。"""
    scheme = make_modular(settings)
    plan = scheme._build_plan(scheme.classifier.classify("这个怎么申请"))
    assert [m.name for m in plan.pre_retrieval] == ["rewrite"]
    assert [m.name for m in plan.retrieval] == ["hybrid_search"]
    assert [m.name for m in plan.post_retrieval] == ["compress"]


def test_build_plan_multihop(settings):
    """多跳/流程：计划 = 迭代检索（multi_hop）→ 重排 → 压缩，无预处理。"""
    scheme = make_modular(settings)
    plan = scheme._build_plan(scheme.classifier.classify("报销发票的流程是什么"))
    assert plan.need_retrieval is True
    assert plan.pre_retrieval == []
    assert [m.name for m in plan.retrieval] == ["multi_hop"]
    assert [m.name for m in plan.post_retrieval] == ["rerank", "compress"]
    assert plan.generation_strategy == "citation"


# ---- 按执行计划路由 ----

def test_route_greeting_no_retrieval(settings):
    """寒暄：不检索，返回空命中（直接生成省延迟）。"""
    scheme = make_modular(settings)
    scheme.ingest(["公司规定员工每日按时打卡考勤。"])
    result = scheme.retrieve_full("你好", top_k=2)
    assert result.hits == []
    assert result.rewrites == []
    assert result.reranked is False


def test_route_simple_hybrid_single_retrieval(settings):
    """简单事实：混合检索，不重写、不重排。"""
    scheme = make_modular(settings)
    scheme.ingest(["公司要求发票随报销单据一次性上传，逾期不受理。"])
    result = scheme.retrieve_full("发票什么时候交", top_k=2)
    assert result.reranked is False
    assert result.rewrites == []
    assert result.decomposed == []
    assert result.hits, "应召回相关片段"


def test_route_complex_decompose_full_pipeline(settings):
    """对比/多实体：分解为子查询 → 多路召回 → 重排 → 压缩。"""
    scheme = make_modular(settings)
    scheme.ingest(
        [
            "公司要求出差结束后15天内提交报销材料。逾期未提交不予受理。",
            "员工申请年假需提前2个工作日提交OA审批，3天以上年假需人力资源部终审。",
        ]
    )
    result = scheme.retrieve_full("出差和报销有什么区别", top_k=2)
    assert result.decomposed, "复杂对比应产生分解子查询"
    assert result.reranked is True
    assert result.compressed is not None, "多路召回后应执行上下文压缩"
    assert result.hits, "应召回相关片段"


def test_route_multihop_iterative(settings):
    """多跳/流程：迭代检索（multi_hop）→ 重排 → 压缩，逐跳记录非空。"""
    scheme = make_modular(settings)
    scheme.ingest(
        [
            "公司差旅管理制度规定，员工出差前需提交出差申请，经部门审批通过后方可出差。",
            "出差结束后员工需在15个自然日内上传发票与报销单据，逾期不予受理。",
        ]
    )
    result = scheme.retrieve_full("报销发票的流程是什么", top_k=2)
    assert result.hops, "多跳应产生逐跳记录"
    assert result.reranked is True
    assert result.compressed is not None, "多跳合并后应执行上下文压缩"
    assert result.hits, "应召回相关片段"


def test_route_multihop_keeps_chain_evidence(settings):
    """多跳链式证据：最终保留命中数按实际跳数放大（>top_k），避免链式上下文被截断。"""
    scheme = make_modular(settings)
    scheme.ingest(
        [
            "出差申请流程：员工出差前需提交出差申请，经部门审批通过后方可出差，报销需在结束后办理。",
            "发票报销流程：出差结束后需在15个自然日内上传发票与报销单据，逾期不予受理。",
            "报销审批流程：报销单据需附发票、行程凭证与出差审批单，材料齐全方可报销。",
            "报销打款流程：报销审批通过后财务在7个工作日内打款到员工工资卡。",
        ]
    )
    result = scheme.retrieve_full("报销发票的流程是什么", top_k=2)
    keep = 2 * len(result.hops)  # top_k × 实际跳数
    assert keep > 2, "多跳场景应存在放大后的保留数"
    assert len(result.hits) == keep, f"多跳应保留 top_k×跳数={keep} 条链式证据，实际 {len(result.hits)}"


def test_rule_multi_hop_retriever_follows_trail(settings):
    """规则兜底（贪心迭代）：top-1 命中暴露新领域关键词时续跳，无新材料即停。"""
    scheme = make_modular(settings, multi_hop=RuleMultiHopRetriever())
    scheme.ingest(
        [
            "公司差旅管理制度规定，员工出差前需提交出差申请，经部门审批通过后方可出差。",
            "出差结束后员工需在15个自然日内上传发票与报销单据，逾期不予受理。",
        ]
    )
    result = scheme.multi_hop.retrieve(
        "报销发票的流程是什么", scheme.store, top_k=2, max_hops=3, recall_k=9
    )
    assert len(result.hops) == 2, "顺藤摸瓜应续跳 1 次（共 2 跳）"
    assert result.hops[0].next_query, "首跳应产出下一跳查询"
    assert result.hops[1].next_query is None
    assert result.hits, "合并命中非空"


def test_llm_multi_hop_retriever(settings, monkeypatch):
    """LLM 兜底：按脚本（续跳→停止）迭代检索，合并去重。"""
    llm = FakeChatModel(
        script=[
            AIMessage(content='{"continue": true, "next_query": "发票上传时限"}'),
            AIMessage(content='{"continue": false}'),
        ]
    )
    monkeypatch.setattr("app.rag.retrieval.iterative_retrieval.get_chat_model", lambda scenario: llm)
    scheme = make_modular(settings, multi_hop=LLMMultiHopRetriever())
    scheme.ingest(
        [
            "公司差旅管理制度规定，员工出差前需提交出差申请，经部门审批通过后方可出差。",
            "出差结束后员工需在15个自然日内上传发票与报销单据，逾期不予受理。",
        ]
    )
    result = scheme.multi_hop.retrieve(
        "报销发票的流程是什么", scheme.store, top_k=2, max_hops=3, recall_k=9
    )
    assert len(result.hops) == 2
    assert result.hops[1].query == "发票上传时限"
    assert result.hops[1].next_query is None
    assert result.hits


async def test_llm_multi_hop_retriever_streams_per_hop(settings, monkeypatch):
    """LLM 兜底：astream_retrieve 逐跳流式产出（每跳一个 HopRecord，而非一次性全部返回）。"""
    llm = FakeChatModel(
        script=[
            AIMessage(content='{"continue": true, "next_query": "发票上传时限"}'),
            AIMessage(content='{"continue": false}'),
        ]
    )
    monkeypatch.setattr("app.rag.retrieval.iterative_retrieval.get_chat_model", lambda scenario: llm)
    scheme = make_modular(settings, multi_hop=LLMMultiHopRetriever())
    scheme.ingest(
        [
            "公司差旅管理制度规定，员工出差前需提交出差申请，经部门审批通过后方可出差。",
            "出差结束后员工需在15个自然日内上传发票与报销单据，逾期不予受理。",
        ]
    )
    hops = [
        h
        async for h in scheme.multi_hop.astream_retrieve(
            "报销发票的流程是什么", scheme.store, top_k=2, max_hops=3, recall_k=9
        )
    ]
    assert len(hops) == 2
    assert hops[0].query == "报销发票的流程是什么"
    assert hops[0].next_query == "发票上传时限"
    assert hops[1].query == "发票上传时限"
    assert hops[1].next_query is None
    assert all(h.hits for h in hops), "每一跳都应携带该跳命中"


async def test_llm_multihop_accumulates_evidence(settings):
    """累积证据：下一跳决策应携带此前所有跳的命中（而非只给当前跳），
    避免重复查询已被证据解决的环节（如「张三的领导是谁」已在首跳命中出现）。"""
    captured: list[list[str]] = []

    class RecordingRetriever(LLMMultiHopRetriever):
        def _decide_next(self, query, all_hits, top_k):  # noqa: ARG002
            captured.append([h.get("text", "") for hop in all_hits for h in hop])
            return "王刚的年假有多少天"

    scheme = make_modular(settings, multi_hop=RecordingRetriever())
    scheme.ingest(
        [
            "张三的直属领导是王刚，王刚担任部门经理。",
            "员工年假按工龄计算：满一年5天，满十年10天。",
        ]
    )
    hops = [
        h
        async for h in scheme.multi_hop.astream_retrieve(
            "张三的领导有多少天年假", scheme.store, top_k=2, max_hops=3, recall_k=9
        )
    ]
    assert len(captured) == 2, "max_hops=3 应有 2 次续跳决策"
    assert hops[1].next_query == "王刚的年假有多少天"
    # 第 2 次决策（第 2 跳后）携带的证据应包含第 1 跳已召回的文本，而非只含当前跳
    first_hop_texts = set(captured[0])
    assert captured[1], "第 2 次决策应收到累积证据"
    assert any(t in captured[1] for t in first_hop_texts), "后续跳决策应参考此前已召回的命中"


def test_rrf_fusion_ranks_common_docs_higher():
    """RRF 融合：出现在多路 / 更靠前的文档融合分更高（跨路分数不可比时仍稳定）。"""
    from app.rag.retrieval.fusion import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(
        [
            [{"text": "A", "score": 0.9}, {"text": "B", "score": 0.8}],
            [{"text": "B", "score": 0.05}, {"text": "C", "score": 0.02}],
        ]
    )
    texts = [h["text"] for h in fused]
    # B 同时出现在两路 → 融合分最高；A/C 各一路且 A 更靠前
    assert texts == ["B", "A", "C"]
    assert fused[0]["score"] > fused[1]["score"] > fused[2]["score"]


# ---- 规划器（MultiHop Planner：规划-执行-验证的规划阶段） ----

def test_rule_planner_entity_chain():
    """规则规划：实体链拆两跳（先定位中间实体再查属性），依赖显式化。"""
    plan = RuleMultiHopPlanner().plan("张三的领导有几天年假")
    assert len(plan.steps) == 2
    assert plan.steps[0].target == "张三的领导是谁"
    assert plan.steps[0].query == "张三的领导是谁"
    assert plan.steps[1].query == "张三的领导有几天年假"
    assert plan.steps[1].depends_on == ["张三的领导是谁"], "第2跳应显式依赖第1跳"


def test_rule_planner_process_single_step():
    """规则规划：流程/原因型退化为单步计划（原查询 + 目标=流程）。"""
    plan = RuleMultiHopPlanner().plan("报销发票的流程是什么")
    assert len(plan.steps) == 1
    assert plan.steps[0].query == "报销发票的流程是什么"
    assert plan.steps[0].target == "流程"


def test_llm_planner_parses_plan(monkeypatch):
    """LLM 规划：按场景懒取模型解析结构化计划 JSON（steps + entity + depends_on + reason）。"""
    llm = FakeChatModel(
        script=[
            AIMessage(
                content='{"steps": ['
                '{"target": "领导是谁", "query": "张三的领导是谁", "entity": null, "depends_on": []}, '
                '{"target": "年假天数", "query": "王刚的年假有多少天", "entity": "王刚", '
                '"depends_on": ["领导是谁"]}], "reason": "实体链多跳"}'
            )
        ]
    )
    monkeypatch.setattr("app.rag.retrieval.planner.get_chat_model", lambda scenario: llm)
    plan = LLMMultiHopPlanner().plan("张三的领导有几天年假")
    assert len(plan.steps) == 2
    assert plan.steps[0].query == "张三的领导是谁"
    assert plan.steps[1].entity == "王刚"
    assert plan.steps[1].depends_on == ["领导是谁"]
    assert plan.reason


def test_llm_planner_invalid_falls_back(monkeypatch):
    """LLM 规划非法/不可解析时回退规则规划（不抛错）。"""
    llm = FakeChatModel(script=[AIMessage(content="乱七八糟，不是 JSON")])
    monkeypatch.setattr("app.rag.retrieval.planner.get_chat_model", lambda scenario: llm)
    plan = LLMMultiHopPlanner().plan("张三的领导有几天年假")
    assert len(plan.steps) >= 1, "应回退到规则规划"


# ---- 验证器（MultiHop Verifier：质量闸门） ----

def test_rule_verifier_marks_covered():
    """规则验证：可预判实体已出现在证据中 → 该目标判定为已覆盖（复用，不重复查）。"""
    plan = HopPlan(steps=[PlanStep(target="年假天数", query="王刚的年假有多少天", entity="王刚")])
    result = RuleMultiHopVerifier().verify(
        "张三的领导有几天年假",
        plan,
        [{"text": "张三的直属领导是王刚，王刚享有10天年假。"}],
    )
    assert result.covered == ["年假天数"]
    assert result.missing == []


def test_rule_verifier_patches_gap():
    """规则验证：缺口产出补缺子查询（顺藤摸瓜关键词扩展），供执行器局部修正。"""
    plan = HopPlan(steps=[PlanStep(target="流程", query="报销发票的流程是什么")])
    result = RuleMultiHopVerifier().verify(
        "报销发票的流程是什么",
        plan,
        [{"text": "公司差旅管理制度规定，员工出差前需提交出差申请，经部门审批通过后方可出差。"}],
    )
    assert result.missing == ["流程"]
    assert result.patched, "缺口应产出补缺子查询"
    assert "出差" in result.patched[0]["query"]


def test_llm_verifier_parses(monkeypatch):
    """LLM 验证：按场景懒取模型解析结构化对表 JSON（covered / missing / patched）。"""
    llm = FakeChatModel(
        script=[AIMessage(content='{"covered": ["领导是谁"], "missing": [], "patched": []}')]
    )
    monkeypatch.setattr("app.rag.retrieval.verifier.get_chat_model", lambda scenario: llm)
    plan = HopPlan(steps=[PlanStep(target="领导是谁", query="张三的领导是谁")])
    result = LLMMultiHopVerifier().verify(
        "张三的领导有几天年假", plan, [{"text": "张三的直属领导是王刚。"}]
    )
    assert result.covered == ["领导是谁"]
    assert result.missing == []


def test_llm_verifier_invalid_falls_back(monkeypatch):
    """LLM 验证非法输出时回退规则验证（不抛错）。"""
    llm = FakeChatModel(script=[AIMessage(content="no json")])
    monkeypatch.setattr("app.rag.retrieval.verifier.get_chat_model", lambda scenario: llm)
    plan = HopPlan(steps=[PlanStep(target="流程", query="报销发票的流程是什么")])
    result = LLMMultiHopVerifier().verify(
        "报销发票的流程是什么", plan, [{"text": "某制度文档片段。"}]
    )
    assert isinstance(result, VerifyResult)


# ---- 规划-执行-验证检索器（PlanExecuteRetriever） ----

def test_plan_execute_reuses_covered_step(settings):
    """覆盖复用：计划中某步的关键实体已被首跳证据覆盖 → 该步复用跳过（不重复检索）。"""
    class DummyPlanner:
        def plan(self, query):  # noqa: ARG002
            return HopPlan(
                steps=[
                    PlanStep(target="领导是谁", query="张三的领导是谁"),
                    PlanStep(target="年假天数", query="王刚的年假有多少天", entity="王刚", depends_on=["领导是谁"]),
                ]
            )

    class DummyVerifier:
        def verify(self, query, plan, evidence_hits):  # noqa: ARG002
            return VerifyResult(covered=["年假天数"], missing=[], patched=[])

    scheme = make_modular(settings, multi_hop=PlanExecuteRetriever(DummyPlanner(), DummyVerifier()))
    scheme.ingest(["张三的直属领导是王刚，王刚享有10天年假。"])
    result = scheme.multi_hop.retrieve(
        "张三的领导有几天年假", scheme.store, top_k=2, max_hops=3, recall_k=9
    )
    assert len(result.hops) == 2
    assert result.hops[0].skipped is False
    assert result.hops[1].skipped is True, "第2跳已被首跳证据覆盖，应复用跳过"
    assert result.hops[1].hits == []
    assert result.plan is not None and len(result.plan.steps) == 2
    assert result.verification.covered == ["年假天数"]


def test_step_covered_content_keywords():
    """内容级覆盖检测（共享函数）：无预判实体的步骤，其新领域词已全部出现在证据中 → 判定已覆盖。"""
    from app.rag.retrieval.iterative_retrieval import step_covered

    step = PlanStep(target="审批环节", query="报销审批的流程是什么")
    assert step_covered(step, "报销发票的流程是什么", "报销审批流程：单据需附发票，经部门审批通过后打款。")
    assert not step_covered(step, "报销发票的流程是什么", "公司要求员工每日按时打卡考勤。")


def test_plan_execute_content_coverage_skips_then_patches(settings):
    """内容级覆盖（规则验证器）：第2跳被首跳证据覆盖 → 复用跳过；终局对表仍对第1跳缺口补修。"""
    class TwoStepPlanner:
        def plan(self, query):  # noqa: ARG002
            return HopPlan(
                steps=[
                    PlanStep(target="基础流程", query="报销发票的流程是什么"),
                    PlanStep(target="审批环节", query="报销审批的流程是什么", depends_on=["基础流程"]),
                ]
            )

    scheme = make_modular(settings, multi_hop=PlanExecuteRetriever(TwoStepPlanner(), RuleMultiHopVerifier()))
    scheme.ingest(["报销审批流程：报销单据需附发票，经部门审批通过后方可打款报销。"])
    result = scheme.multi_hop.retrieve(
        "报销发票的流程是什么", scheme.store, top_k=2, max_hops=3, recall_k=9
    )
    assert len(result.hops) == 3, "第1跳执行 + 第2跳复用跳过 + 终局对第1跳缺口补修1跳"
    assert result.hops[0].skipped is False
    assert result.hops[1].skipped is True, "第2跳已被首跳证据覆盖，应复用跳过"
    assert result.hops[2].skipped is False, "补修子查询应实际检索"
    assert result.hops[2].target == "基础流程"


def test_plan_execute_llm_verifier_skips_covered_step(settings, monkeypatch):
    """LLM 验证器：逐跳前调用验证器（关闭思考模式），判定该步已被证据覆盖 → 复用跳过。"""
    llm = FakeChatModel(
        script=[
            AIMessage(content='{"covered": ["审批环节"], "missing": [], "patched": []}'),  # 第2跳逐跳覆盖对表
            AIMessage(content='{"covered": ["基础流程", "审批环节"], "missing": [], "patched": []}'),  # 终局对表
        ]
    )
    monkeypatch.setattr("app.rag.retrieval.verifier.get_chat_model", lambda scenario: llm)

    class TwoStepPlanner:
        def plan(self, query):  # noqa: ARG002
            return HopPlan(
                steps=[
                    PlanStep(target="基础流程", query="报销发票的流程是什么"),
                    PlanStep(target="审批环节", query="报销审批的流程是什么", depends_on=["基础流程"]),
                ]
            )

    scheme = make_modular(
        settings,
        multi_hop=PlanExecuteRetriever(TwoStepPlanner(), LLMMultiHopVerifier()),
    )
    scheme.ingest(["报销审批流程：报销单据需附发票，经部门审批通过后方可打款报销。"])
    result = scheme.multi_hop.retrieve(
        "报销发票的流程是什么", scheme.store, top_k=2, max_hops=3, recall_k=9
    )
    assert len(result.hops) == 2
    assert result.hops[0].skipped is False
    assert result.hops[1].skipped is True, "LLM 验证器判定第2跳已被覆盖，应复用跳过"


def test_rag_verify_scenario_thinking_off():
    """验证场景（rag_verify）显式关闭思考模式：轻量决策调用不拖慢链路。"""
    from app.llm.service import DEFAULT_PROFILES

    profile = next(p for p in DEFAULT_PROFILES if p["scenario"] == "rag_verify")
    assert profile["params"]["enable_thinking"] is False


def test_plan_execute_rule_verify_patches(settings):
    """验证闸门：规则路径下流程型单步计划缺口 → 局部修正补查一跳，并如实上报缺口。"""
    scheme = make_modular(settings)  # 默认规则规划-执行-验证
    scheme.ingest(
        [
            "公司差旅管理制度规定，员工出差前需提交出差申请，经部门审批通过后方可出差。",
            "出差结束后员工需在15个自然日内上传发票与报销单据，逾期不予受理。",
        ]
    )
    result = scheme.multi_hop.retrieve(
        "报销发票的流程是什么", scheme.store, top_k=2, max_hops=3, recall_k=9
    )
    assert result.plan is not None and result.plan.steps, "应产出多跳计划"
    assert result.verification.missing, "质量闸门应如实上报未覆盖目标（可观测）"
    assert len(result.hops) == 2, "计划1跳 + 补缺1跳"
    assert result.hits, "合并命中非空"


def test_plan_execute_respects_budget(settings):
    """预算约束：超过 max_hops 的步骤不执行（unexecuted），防死循环。"""
    class ManyStepPlanner:
        def plan(self, query):  # noqa: ARG002
            return HopPlan(
                steps=[PlanStep(target=f"步{i}", query=f"子查询{i}") for i in range(1, 6)]
            )

    class NoopVerifier:
        def verify(self, query, plan, evidence_hits):  # noqa: ARG002
            return VerifyResult(covered=[], missing=[], patched=[])

    scheme = make_modular(settings, multi_hop=PlanExecuteRetriever(ManyStepPlanner(), NoopVerifier()))
    scheme.ingest(["公司差旅管理制度规定，员工出差前需提交出差申请。"])
    result = scheme.multi_hop.retrieve(
        "报销发票的流程是什么", scheme.store, top_k=2, max_hops=3, recall_k=9
    )
    assert len(result.hops) == 3, "只应执行 max_hops=3 跳"
    assert all(not h.skipped for h in result.hops)


# ---- 流式事件顺序 ----

async def test_astream_deictic_resolution_rewrite_first(settings):
    """指代消解在前：astream 先产 rewrite(指代消解) 事件，classify 使用消解后 query。"""
    scheme = make_modular(settings)
    scheme.ingest(["研发部主管王刚负责年假初审，员工年假按工龄每年5天起。"])
    events = [
        ev
        async for ev in scheme.astream(
            "他的年假有多少天", 2, context="用户: 张三的领导是谁\n助手: 王刚"
        )
    ]
    assert events[0]["type"] == "rewrite"
    assert events[0]["reason"] == "指代消解"
    assert events[0]["rewrites"] == ["王刚的年假有多少天"]
    classify = next(e for e in events if e["type"] == "classify")
    assert classify["query"] == "王刚的年假有多少天"


async def test_astream_classify_before_retrieve(settings):
    """modular 应先产出 classify（路由）事件，再按执行计划产出检索事件。"""
    scheme = make_modular(settings)
    scheme.ingest(["公司要求出差结束后15天内提交报销材料。"])
    events = [ev async for ev in scheme.astream("出差和报销有什么区别", 2)]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "classify"
    assert "retrieve" in kinds
    assert kinds.index("classify") < kinds.index("retrieve")


async def test_astream_simple_no_hits(settings):
    """寒暄：classify（retrieval_need=False）后不产出检索事件（不检索不注入上下文）。"""
    scheme = make_modular(settings)
    scheme.ingest(["公司规定员工每日按时打卡考勤。"])
    events = [ev async for ev in scheme.astream("你好", 2)]
    # classify 先发 running 占位、再发 done 填充：取最后一条（done）验证完整决策
    classify = [e for e in events if e["type"] == "classify"][-1]
    assert classify["type"] == "classify"
    assert classify["status"] == "done"
    assert classify["retrieval_need"] is False
    assert classify["complexity"] == SIMPLE
    assert all(e["type"] != "retrieve" for e in events)


async def test_astream_decompose_and_compress_events(settings):
    """对比/多实体：产出 decompose（分解子查询）与 compress（压缩统计）事件。"""
    long_chunk = (
        "公司差旅报销管理制度：所有出差费用仅限公务刚需支出，个人消费、娱乐消费、超额消费一律不予报销。"
        "所有住宿报销需提供对应城市、对应出差时段的正规增值税普通发票，发票信息与出差审批单信息不一致不予受理。"
        "所有出差报销单据、发票、行程凭证，必须在出差结束后的15个自然日内完整上传OA报销系统，逾期未提交、"
        "材料不全、信息有误的报销申请一律不予受理，自动作废。"
        "差旅交通补贴规则：日常短途出差优先选择高铁二等座、动车二等座公共交通，单程交通里程超过1500公里的"
        "长途出差可申请民航经济舱机票报销，商务舱、头等舱不予报销。市内交通优先公共交通，凭票据实报销，"
        "无票据交通支出不予核算。所有出差费用仅限公务刚需支出，个人消费、娱乐消费、超额消费一律不予报销。"
        "所有住宿报销需提供对应城市、对应出差时段的正规增值税普通发票，发票信息与出差审批单信息不一致不予受理。"
        "所有出差报销单据、发票、行程凭证，必须在出差结束后的15个自然日内完整上传OA报销系统，逾期未提交、"
        "材料不全、信息有误的报销申请一律不予受理，自动作废。"
    )
    scheme = make_modular(settings)
    scheme.ingest([long_chunk])
    events = [ev async for ev in scheme.astream("出差和报销有什么区别", 2)]
    kinds = [e["type"] for e in events]
    assert "decompose" in kinds
    decompose = next(e for e in events if e["type"] == "decompose")
    assert len(decompose["sub_queries"]) >= 2
    assert "compress" in kinds
    compress = next(e for e in events if e["type"] == "compress")
    assert compress["metrics"]["truncated"] >= 1, "超长块应被截断"


async def test_astream_multihop_events(settings):
    """多跳/流程：classify → multi_hop_plan（规划）→ 逐跳 multi_hop → multi_hop_verify（验证）→ retrieve，事件顺序正确。"""
    scheme = make_modular(settings)
    scheme.ingest(
        [
            "公司差旅管理制度规定，员工出差前需提交出差申请，经部门审批通过后方可出差。",
            "出差结束后员工需在15个自然日内上传发票与报销单据，逾期不予受理。",
        ]
    )
    events = [ev async for ev in scheme.astream("报销发票的流程是什么", 2)]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "classify"
    assert "multi_hop_plan" in kinds, "规划-执行-验证应先产出多跳规划事件"
    assert "multi_hop_verify" in kinds, "规划-执行-验证应产出质量闸门验证事件"
    assert (
        kinds.index("classify")
        < kinds.index("multi_hop_plan")
        < kinds.index("multi_hop")
        < kinds.index("multi_hop_verify")
        < kinds.index("retrieve")
    ), "事件顺序应为 classify → 规划 → 逐跳 → 验证 → 检索"
    # classify / multi_hop_plan 均先发 running 占位、再发 done 填充：验证 running→done 的流式渐进呈现
    classify_evs = [e for e in events if e["type"] == "classify"]
    assert classify_evs[0]["status"] == "running" and classify_evs[-1]["status"] == "done"
    plan_evs = [e for e in events if e["type"] == "multi_hop_plan"]
    assert plan_evs[0]["status"] == "running" and plan_evs[-1]["status"] == "done"
    multi_hops = [e for e in events if e["type"] == "multi_hop"]
    assert len(multi_hops) >= 2, "多跳应逐跳流式产出多个 multi_hop 事件，而非一次性合并返回"
    assert [e["index"] for e in multi_hops] == list(range(1, len(multi_hops) + 1)), "逐跳事件 index 从 1 递增"
    assert all(e["hop"]["hits"] for e in multi_hops), "每一跳事件都应携带该跳命中"
    assert all(e["hop"]["target"] for e in multi_hops), "每一跳事件都应携带目标维度"
    plan_ev = plan_evs[-1]
    assert plan_ev["plan"]["steps"], "规划事件应携带子查询步骤"
    verify_ev = next(e for e in events if e["type"] == "multi_hop_verify")
    assert "covered" in verify_ev["verification"] and "missing" in verify_ev["verification"]
    retrieve = next(e for e in events if e["type"] == "retrieve")
    assert retrieve["hits"], "合并命中应注入检索结果"


async def test_astream_multihop_keeps_chain_evidence(settings):
    """多跳流式：最终注入的检索命中数按实际检索跳数放大（>top_k），避免链式上下文被截断。"""
    scheme = make_modular(settings)
    scheme.ingest(
        [
            "出差申请流程：员工出差前需提交出差申请，经部门审批通过后方可出差，报销需在结束后办理。",
            "发票报销流程：出差结束后需在15个自然日内上传发票与报销单据，逾期不予受理。",
            "报销审批流程：报销单据需附发票、行程凭证与出差审批单，材料齐全方可报销。",
            "报销打款流程：报销审批通过后财务在7个工作日内打款到员工工资卡。",
        ]
    )
    events = [ev async for ev in scheme.astream("报销发票的流程是什么", 2)]
    multi_hops = [e for e in events if e["type"] == "multi_hop"]
    retrieved = [e for e in multi_hops if e["hop"]["hits"]]
    keep = 2 * len(retrieved)  # top_k=2 × 实际检索跳数（覆盖复用跳不计入）
    assert keep > 2, "多跳场景应存在放大后的保留数"
    retrieve = next(e for e in events if e["type"] == "retrieve")
    assert len(retrieve["hits"]) == keep, (
        f"多跳应保留 top_k×跳数={keep} 条链式证据注入上下文，实际 {len(retrieve['hits'])}"
    )


# ---- 方案注册与构建（前端选择的实际入口） ----

def test_manager_builds_and_resolves_modular(settings, monkeypatch):
    """rag_schemes 含 modular 时，manager 通过注册表构建该方案并可解析/入库/检索。"""
    settings.rag_schemes = ["naive", "advanced", "modular"]
    # 路由为纯 LLM：注入脚本化路由决策，避免测试依赖真实模型
    llm = FakeChatModel(
        script=[
            AIMessage(
                content='{"retrieval_need": true, "retrieval_mode": "multi_recall", '
                '"complexity": "decompose", "generation_mode": "comparison", '
                '"confidence": 0.9, "reason": "多实体对比"}'
            )
        ]
    )
    monkeypatch.setattr("app.rag.routing.classifier.get_chat_model", lambda scenario: llm)
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    assert "modular" in manager.schemes
    scheme = manager.get("modular")
    assert isinstance(scheme, ModularRagScheme)
    assert scheme.collection == f"{settings.qdrant_collection_prefix}_modular"
    assert manager.resolve("modular") is scheme
    manager.ingest_all(["公司要求出差结束后15天内提交报销材料。"])
    assert len(scheme) == 1
    assert scheme.retrieve_full("出差和报销有什么区别", top_k=2).hits


# ---- 检索后答案充分性验证（Answerability Gate：跨复杂度路径的统一兜底） ----

def test_rule_answerability_empty_hits_escalates():
    """规则验证：未检索到任何命中 → 明确不足（升级多路召回），不静默进入生成。"""
    verdict = RuleAnswerabilityVerifier().verify("王刚的年假有多少天", [])
    assert verdict.answerable is False
    assert verdict.recommendation == ESCALATE
    assert verdict.escalate_to == "multi_recall"


def test_rule_answerability_missing_keyword_escalates():
    """规则验证：查询中的领域关键词未出现在命中文本 → 缺失对应事实（升级）。"""
    verdict = RuleAnswerabilityVerifier().verify(
        "王刚的年假有多少天", [{"text": "王刚是研发部部门主管，负责日常考勤审核。"}]
    )
    assert verdict.answerable is False
    assert verdict.recommendation == ESCALATE
    assert any("年假" in f for f in verdict.missing_facts)


def test_rule_answerability_covered_answerable():
    """规则验证：查询领域词已在命中文本中 → 保守判定可答（不误伤）。"""
    verdict = RuleAnswerabilityVerifier().verify(
        "王刚的年假有多少天", [{"text": "王刚在岗6年，工龄满6年及以上员工年假统一固定为10天。"}]
    )
    assert verdict.answerable is True
    assert verdict.recommendation == ANSWER


def test_llm_answerability_parses(monkeypatch):
    """LLM 验证：按场景懒取模型解析结构化充分性判定 JSON（answerable / missing_facts / recommendation）。"""
    llm = FakeChatModel(
        script=[
            AIMessage(
                content='{"answerable": false, "missing_facts": ["王刚的在岗工龄"], '
                '"recommendation": "escalate", "escalate_to": "multi_recall"}'
            )
        ]
    )
    monkeypatch.setattr("app.rag.retrieval.answerability.get_chat_model", lambda scenario: llm)
    verdict = LLMAnswerabilityVerifier().verify(
        "王刚的年假有多少天", [{"text": "王刚是研发部部门主管，负责年假初审。"}]
    )
    assert verdict.answerable is False
    assert verdict.recommendation == ESCALATE
    assert verdict.escalate_to == "multi_recall"
    assert "工龄" in verdict.missing_facts[0]


def test_llm_answerability_invalid_falls_back(monkeypatch):
    """LLM 验证非法输出时回退规则验证（不抛错、不中断链路）。"""
    llm = FakeChatModel(script=[AIMessage(content="no json")])
    monkeypatch.setattr("app.rag.retrieval.answerability.get_chat_model", lambda scenario: llm)
    verdict = LLMAnswerabilityVerifier().verify("王刚的年假有多少天", [])
    assert verdict.answerable is False, "空命中经规则回退应判不足"


def test_llm_answerability_keeps_far_table_rows(monkeypatch):
    """LLM 验证：证据按条保留 800 字符，表格型明细远在 200 字符外的关键行不丢失
    （回归：表格明细的「李雪→产品部」位于 200 字符之外，原 200 截断会把该事实切掉误判缺失。
    注：部门规模对比类查询已由确定性兜底短路（不依赖 LLM），故此处用人员属性查询验证截断行为）。"""
    table = (
        "第四章 关键人员权益明细\n"
        "| 姓名 | 部门/岗位 | 入职 | 工龄(截至2026) | 年假 | 夜间交通报销 | 远程办公 | 大额报销终审 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| 张三 | 研发部/算法工程师 | 2022-06-10 | 3年 | 7天 | 享有 | 可申请 | 无 |\n"
        "| 王刚 | 研发部/部门主管 | 2018-03-01 | 7年 | 10天 | 享有 | 可申请 | 无（仅初审） |\n"
        "| 李雪 | 产品部/产品经理 | 2023-03-05 | 2年 | 6天 | 不享有 | 可申请 | 无 |\n"
        "| 陈丽 | 人力资源部/经理 | 2016-01-15 | 9年 | 10天 | 不享有 | 不可申请 | 无（终审年假） |\n"
        "| 刘芳 | 财务部/主管 | 2017-08-22 | 8年 | 10天 | 不享有 | 不可申请 | 2000元以上二次复核 |\n"
        "| 赵凯 | 市场部/市场专员 | 2024-02-20 | 1年 | 5天 | 不享有 | 可申请 | 无 |"
    )
    assert table.index("李雪 | 产品部") > 200, "测试前提：李雪行须位于 200 字符之外（原截断会切掉）"

    captured = {}

    class RecordingLLM:
        def invoke(self, messages, *args, **kwargs):  # noqa: ARG002
            captured["human"] = messages[1].content
            return AIMessage(
                content='{"answerable": false, "missing_facts": ["李雪的夜间交通报销"], '
                '"recommendation": "escalate", "escalate_to": "multi_recall"}'
            )

    monkeypatch.setattr("app.rag.retrieval.answerability.get_chat_model", lambda scenario: RecordingLLM())
    verdict = LLMAnswerabilityVerifier().verify("李雪的夜间交通报销是什么", [{"text": table}])
    assert "李雪" in captured["human"] and "产品部" in captured["human"], (
        "800 字符截断后「李雪→产品部」应仍出现在验证证据中，不应被误判为缺失"
    )
    assert verdict.recommendation == ESCALATE


def test_llm_answerability_prompt_escalates_for_missing_structural_data(monkeypatch):
    """LLM 验证提示词：明确「缺失库内结构化数据（部门编制规模/花名册/表格明细）→ escalate 而非 clarify」
    （回归：验证器曾把「张三/李雪所在部门的人数」缺失判成需追问澄清，而数据本就存在于库内第五章）。"""
    captured = {}

    class RecordingLLM:
        def invoke(self, messages, *args, **kwargs):  # noqa: ARG002
            captured["system"] = messages[0].content
            return AIMessage(content='{"answerable": true, "missing_facts": [], "recommendation": "answer"}')

    monkeypatch.setattr("app.rag.retrieval.answerability.get_chat_model", lambda scenario: RecordingLLM())
    LLMAnswerabilityVerifier().verify("张三的部门比李雪的部门哪个人多", [{"text": "研发部 130 人"}])
    assert "部门编制规模" in captured["system"]
    assert "escalate" in captured["system"]


def test_llm_answerability_deterministic_escalates_without_llm(monkeypatch):
    """确定性兜底：部门规模对比类查询、证据无规模数据 → 直接判 escalate（multihop），且不调用 LLM
    （回归：LLM 曾仅凭人员花名册误判「可答/需追问」，绕过升级检索 → 第五章部门规模表永远召不回）。"""
    called = {"n": 0}

    class RecordingLLM:
        def invoke(self, messages, *args, **kwargs):  # noqa: ARG002
            called["n"] += 1
            return AIMessage(content='{"answerable": true, "missing_facts": [], "recommendation": "answer"}')

    monkeypatch.setattr("app.rag.retrieval.answerability.get_chat_model", lambda scenario: RecordingLLM())
    hits = [{"text": "| 张三 | 研发部/算法工程师 | 2022-06-10 | 3年 |\n| 李雪 | 产品部/产品经理 | 2023-03-05 | 2年 |"}]
    verdict = LLMAnswerabilityVerifier().verify("张三的部门比李雪的部门哪个人多", hits)
    assert called["n"] == 0, "证据缺规模数据时应由确定性兜底直接升级，不依赖 LLM 判定"
    assert verdict.answerable is False
    assert verdict.recommendation == ESCALATE
    assert verdict.escalate_to == "multihop"
    assert "部门规模" in "".join(verdict.missing_facts)


class ScriptedVerifier:
    """测试桩：按调用次数返回脚本化验证结论，记录调用次数（验证编排是否升级重检）。"""

    def __init__(self, *verdicts: AnswerabilityVerdict):
        self.verdicts = list(verdicts)
        self.calls = 0

    def verify(self, query, hits):  # noqa: ARG002
        v = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return v


def test_execute_plan_escalates_on_insufficient(settings):
    """同步：单次检索不足 → 有界升级 1 轮（多路召回+重排）→ 二次验证可答，最终命中采用升级轮结果。"""
    verifier = ScriptedVerifier(
        AnswerabilityVerdict(
            answerable=False, missing_facts=["报销时限"], recommendation=ESCALATE, escalate_to="multi_recall"
        ),
        AnswerabilityVerdict(answerable=True, missing_facts=[], recommendation=ANSWER),
    )
    scheme = make_modular(settings, answerability=verifier)
    scheme.ingest(["公司要求发票随报销单据一次性上传，逾期不受理。"])
    result = scheme.retrieve_full("发票什么时候交", top_k=2)
    assert verifier.calls == 2, "应验证两次：首轮不足 + 升级后复验"
    assert result.answerability is not None
    assert result.answerability["answerable"] is True
    assert result.reranked is True, "升级到多路召回后应执行重排"


def test_execute_plan_department_size_escalation_reaches_chapter5(settings):
    """回归：对比「张三/李雪所在部门人数」首轮多路召回未命中部门规模表 → 答案充分性建议升级 →
    升级轮应命中第五章「各部门人员规模与编制」，最终判定可答（修复前会误报需追问澄清）。"""

    class ComparisonClassifier:
        def classify(self, query):
            if "比" in query:
                return RouteDecision(
                    retrieval_need=True, retrieval_mode=MULTI_RECALL, complexity=DECOMPOSE,
                    generation_mode=COMPARISON, confidence=0.92, reason="多实体对比",
                )
            return StubClassifier().classify(query)

    ch4 = (
        "第四章 关键人员权益明细\n"
        "| 姓名 | 部门/岗位 | 入职 | 年假 |\n"
        "| --- | --- | --- | --- |\n"
        "| 张三 | 研发部/算法工程师 | 2022-06-10 | 7天 |\n"
        "| 李雪 | 产品部/产品经理 | 2023-03-05 | 6天 |\n"
    )
    ch5 = (
        "第五章 各部门人员规模与编制\n"
        "| 部门 | 在职人数 | 说明 |\n"
        "| --- | --- | --- |\n"
        "| 研发部 | 130 | 核心研发团队 |\n"
        "| 产品部 | 120 | 产品体系 |\n"
    )
    verifier = ScriptedVerifier(
        AnswerabilityVerdict(
            answerable=False,
            missing_facts=["张三所在部门的人数", "李雪所在部门的人数"],
            recommendation=ESCALATE,
            escalate_to="multi_recall",
        ),
        AnswerabilityVerdict(answerable=True, missing_facts=[], recommendation=ANSWER),
    )
    scheme = make_modular(settings, answerability=verifier, classifier=ComparisonClassifier())
    scheme.ingest([ch4, ch5])
    result = scheme.retrieve_full("张三的部门比李雪的部门哪个人多", top_k=2)
    assert verifier.calls == 2, "首轮不足应升级一轮后复验"
    assert result.answerability["answerable"] is True
    joined = "\n".join(h.get("text", "") for h in result.hits)
    assert "在职人数" in joined, "升级轮应命中第五章部门规模表（修复前此轮缺失会误报需追问澄清）"


def test_execute_plan_multihop_insufficient_maps_to_clarify(settings):
    """同步：已是最全路径（多跳）仍不足 → 不重复升级，最终建议归一为追问澄清。"""
    verifier = ScriptedVerifier(
        AnswerabilityVerdict(
            answerable=False, missing_facts=["在岗工龄"], recommendation=ESCALATE, escalate_to="multi_recall"
        ),
    )
    scheme = make_modular(settings, answerability=verifier)
    scheme.ingest(["员工出差前需提交出差申请，经部门审批通过后方可出差。"])
    result = scheme.retrieve_full("报销发票的流程是什么", top_k=2)
    assert verifier.calls == 1, "多跳路径不支持再升级，不应二次检索"
    assert result.answerability["answerable"] is False
    assert result.answerability["recommendation"] == CLARIFY


async def test_astream_insufficient_escalates_and_final_verdict(settings):
    """流式：首轮检索不足 → 下发 answerability(升级) → 二次检索 → 下发 answerability(最终可答，escalated=True)。"""
    verifier = ScriptedVerifier(
        AnswerabilityVerdict(
            answerable=False, missing_facts=["报销时限"], recommendation=ESCALATE, escalate_to="multi_recall"
        ),
        AnswerabilityVerdict(answerable=True, missing_facts=[], recommendation=ANSWER),
    )
    scheme = make_modular(settings, answerability=verifier)
    scheme.ingest(["公司要求发票随报销单据一次性上传，逾期不受理。"])
    events = [ev async for ev in scheme.astream("发票什么时候交", 2)]
    kinds = [e["type"] for e in events]
    assert kinds.count("retrieve") == 2, "升级后应二次检索"
    answers = [e for e in events if e["type"] == "answerability"]
    assert len(answers) == 2
    assert answers[0]["verdict"]["recommendation"] == ESCALATE and answers[0]["escalated"] is False
    assert answers[1]["verdict"]["answerable"] is True and answers[1]["escalated"] is True
    assert kinds.index("answerability") < kinds.index("retrieve", kinds.index("retrieve") + 1), (
        "首次 answerability(升级) 应位于两次 retrieve 之间"
    )


async def test_astream_clarify_recommendation_does_not_escalate(settings):
    """流式：验证建议澄清（如缺指代/信息确实缺失）→ 不再升级检索，直接如实上报
    （回归：astream 曾无视 clarify 建议盲目升级，二次验证后可能越权作答）。"""
    verifier = ScriptedVerifier(
        AnswerabilityVerdict(
            answerable=False, missing_facts=["'他'具体指代哪一位员工"], recommendation=CLARIFY, escalate_to=None
        ),
    )
    scheme = make_modular(settings, answerability=verifier)
    scheme.ingest(["张三的直属领导是王刚，王刚在岗6年，工龄满6年及以上员工年假统一固定为10天。"])
    events = [ev async for ev in scheme.astream("他的年假有多少天", 2)]
    kinds = [e["type"] for e in events]
    assert kinds.count("retrieve") == 1, "建议澄清不应二次检索升级"
    answers = [e for e in events if e["type"] == "answerability"]
    assert len(answers) == 1, "建议澄清应只下发一次验证结论"
    assert answers[0]["verdict"]["answerable"] is False
    assert answers[0]["verdict"]["recommendation"] == CLARIFY
    assert answers[0]["escalated"] is False
    assert verifier.calls == 1, "不应触发升级后的二次验证"


async def test_runner_insufficient_injects_clarify_directive(settings, sessions):
    """runner：答案充分性验证判定不足 → 注入给主 LLM 的消息携带「追问澄清、不编造」指令。"""
    class FakeScheme:
        name = "模块化 RAG"

        async def astream(self, query, top_k=None, context=None):  # noqa: ARG002
            yield {"type": "classify", "query": query, "scheme": "modular",
                   "retrieval_need": True, "retrieval_mode": "vector", "complexity": "simple",
                   "generation_mode": "citation", "confidence": 0.9, "reason": "单点事实"}
            yield {"type": "retrieve", "query": query, "scheme": "modular",
                   "hits": [{"text": "王刚是研发部部门主管，负责年假初审。", "score": 0.3}]}
            yield {"type": "answerability", "query": query, "scheme": "modular",
                   "verdict": {"answerable": False, "missing_facts": ["王刚的在岗工龄"],
                               "recommendation": "clarify", "escalate_to": None},
                   "escalated": False}

    class FakeRagManager:
        def resolve(self, rag_scheme):  # noqa: ARG002
            return FakeScheme()

    class FakeRegistry:
        def __init__(self):
            self.rag_manager = FakeRagManager()

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
        RecordingLLM(script=[AIMessage(content="抱歉，检索结果缺少王刚的在岗工龄，无法计算具体年假。")]),
        FakeRegistry(),
        sessions,
    )
    async for _ in runner.stream(
        "s2", "王刚的年假有多少天", "react", [], "standard", "never",
        rag_scheme="modular", rag_enabled=True,
    ):
        pass
    injected = [c for c in captured if "知识库检索结果" in c]
    assert injected, "应有注入检索结果的用户消息"
    latest = injected[-1]
    assert "追问" in latest and "不要编造" in latest, "不足时注入的指令应强制追问澄清、不编造"
