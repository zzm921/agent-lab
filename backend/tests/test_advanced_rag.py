"""Advanced RAG 方案测试：语义分块 / Query 重写 / 多路召回 + 重排，全程离线。

使用 FakeEmbeddings（无 api_key → 词法重排回退）与 FakeChatModel（脚本化改写输出），
不联网、不依赖 Key；稀疏集合用内嵌 QdrantClient(":memory:") 验证。
"""
import asyncio
import uuid

from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient

from app.llm.fake_model import FakeChatModel, FakeEmbeddings
from app.memory.stores.memory_store import MemoryStore
from app.memory.stores.qdrant_store import QdrantStore
from app.rag.advanced import AdvancedRagScheme, CHUNK_MAX
from app.rag.manager import RagManager
from app.rag.query_rewrite import LLMQueryRewriter, RuleQueryRewriter
from app.rag.reranker import LexicalReranker


def make_advanced(settings, store=None, **kw) -> AdvancedRagScheme:
    """构造 advanced 方案：未指定 store 时用内存回退，rewriter/reranker 可注入。"""
    if store is None:
        store = MemoryStore(FakeEmbeddings(), collection="knowledge_advanced")
    kw.setdefault("rewriter", RuleQueryRewriter())
    kw.setdefault("reranker", LexicalReranker())
    return AdvancedRagScheme(FakeEmbeddings(), store, top_k=3, **kw)


# ---- 入库拆分优化：语义分块 ----

def test_semantic_chunking(settings):
    """高相似长文本被语义合并为少数块：无块超限、块边界在句末、块数远少于句数。"""
    long_text = "公司规定员工每日按时打卡考勤。" * 30  # 30 句高相似句
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    manager.ingest_all([long_text])
    scheme = manager.get("advanced")
    texts = scheme.store.all_texts()
    assert texts, "应产生语义块"
    assert all(len(t) <= CHUNK_MAX for t in texts)
    assert all(t.endswith("。") for t in texts)  # 块边界保持在句末
    assert len(texts) <= 4  # 语义合并后块数远少于 30 句


def test_semantic_chunking_idempotent(settings):
    """advanced 入库幂等：语料未变时重复入库不增加块。"""
    manager = RagManager(settings, FakeEmbeddings(), top_k=3)
    manager.ingest_all(["公司规定员工每日按时打卡考勤。" * 20])
    count = len(manager.get("advanced"))
    manager.ingest_all(["公司规定员工每日按时打卡考勤。" * 20])
    assert len(manager.get("advanced")) == count


def test_split_long_without_punctuation(settings):
    """无标点超长句走兜底硬切，任一块不超上限。"""
    scheme = make_advanced(settings)
    chunks = scheme._semantic_chunks("考勤" * (CHUNK_MAX * 2 + 50))
    assert len(chunks) >= 2
    assert all(len(c) <= CHUNK_MAX for c in chunks)


# ---- Query 重写 ----

def test_query_rewrite_llm():
    """LLM 改写：解析多行变体，且首位始终为原始查询（保证基础召回）。"""
    llm = FakeChatModel(
        script=[AIMessage(content="出差结束后需在多久内提交报销凭证\n出差报销材料提交时限")]
    )
    rw = LLMQueryRewriter(llm, variants=2)
    variants = rw.rewrite("出差结束多久提交报销材料")
    assert variants[0] == "出差结束多久提交报销材料"
    assert "出差结束后需在多久内提交报销凭证" in variants
    assert "出差报销材料提交时限" in variants


def test_query_rewrite_fallback_rule():
    """无 LLM 的规则回退：去客套语 + 关键词变体，始终含原始查询。"""
    rw = RuleQueryRewriter()
    variants = rw.rewrite("请问出差结束多久提交报销材料")
    assert variants[0] == "请问出差结束多久提交报销材料"
    assert "出差结束多久提交报销材料" in variants  # 去客套语
    assert any("出差" in v and "报销" in v for v in variants)  # 关键词变体


def test_query_rewrite_rule_invoice_timeline():
    """发票 + 时限意图：规则改写补齐「提交时限/截止日期/流程及时限」变体，提高召回。"""
    rw = RuleQueryRewriter()
    variants = rw.rewrite("发票什么时候交")
    assert variants[0] == "发票什么时候交"
    assert "发票提交时限规定" in variants
    assert "发票报销截止日期规定" in variants
    assert "发票上交流程及时限要求" in variants
    # 无时限意图（如「发票丢了怎么补开」）不硬塞时限变体
    assert len(rw.rewrite("发票丢了怎么补开")) < len(variants)


def test_llm_rewrite_failure_keeps_original():
    """LLM 调用异常（如脚本耗尽抛错）时至少保留原始查询。"""
    llm = FakeChatModel(script=[])  # 队列耗尽返回默认回答而非抛错
    rw = LLMQueryRewriter(llm, variants=2)
    variants = rw.rewrite("如何申请年假")
    assert variants[0] == "如何申请年假"


# ---- 检索：多查询 × 多路召回 + 重排 ----

def test_retrieve_full_pipeline(settings):
    """advanced 完整链路：规则重写 + 稀疏混合召回 + 词法重排，retrieve 与 retrieve_full 一致。"""
    store = QdrantStore(
        FakeEmbeddings(),
        collection=f"adv_{uuid.uuid4().hex[:8]}",
        dim=32,
        sparse=True,
        client=QdrantClient(":memory:"),
    )
    scheme = make_advanced(settings, store=store)
    scheme.ingest(["公司要求出差结束后15天内提交报销材料。逾期未提交不予受理。"])
    result = scheme.retrieve_full("出差结束多久提交报销", top_k=2)
    assert result.rewrites[0] == "出差结束多久提交报销"
    assert len(result.rewrites) >= 2  # 规则重写产生多查询
    assert result.reranked is True
    assert result.hits, "应召回相关片段"
    assert all("score" in h for h in result.hits)
    assert scheme.retrieve("出差结束多久提交报销", top_k=2) == result.hits


def test_retrieve_full_pipeline_with_llm_rewrite(settings):
    """LLM 重写端到端：改写变体进入多路召回并重排。"""
    store = QdrantStore(
        FakeEmbeddings(),
        collection=f"adv_{uuid.uuid4().hex[:8]}",
        dim=32,
        sparse=True,
        client=QdrantClient(":memory:"),
    )
    llm = FakeChatModel(script=[AIMessage(content="出差结束报销凭证提交时限")])
    scheme = AdvancedRagScheme(
        FakeEmbeddings(), store, top_k=3, llm=llm, reranker=LexicalReranker()
    )
    scheme.ingest(["出差结束后15个自然日内必须上传报销材料。逾期不予受理。"])
    result = scheme.retrieve_full("出差结束后多久提交报销凭证", top_k=2)
    assert "出差结束报销凭证提交时限" in result.rewrites  # LLM 改写变体参与召回
    assert result.reranked is True
    assert result.hits


async def test_astream_yields_between_rewrite_and_retrieve(settings):
    """rewrite 与 retrieve 事件之间让出事件循环：重写事件才能经 SSE 先行下发，
    而不是与检索事件攒在同一次刷出、被前端「同时」展示。"""
    scheme = make_advanced(settings)
    scheme.ingest(["公司要求出差结束后15天内提交报销材料。"])

    ticks = 0

    async def probe():
        # 只有在事件循环被让出时才有机会被调度，用于探测 astream 是否阻塞循环
        nonlocal ticks
        while True:
            await asyncio.sleep(0)
            ticks += 1

    task = asyncio.create_task(probe())
    await asyncio.sleep(0)

    ticks_at_rewrite = ticks_at_retrieve = -1
    try:
        async for ev in scheme.astream("出差结束多久提交报销", 2):
            if ev["type"] == "rewrite":
                ticks_at_rewrite = ticks
            elif ev["type"] == "retrieve":
                ticks_at_retrieve = ticks
    finally:
        task.cancel()

    assert ticks_at_rewrite >= 0 and ticks_at_retrieve >= 0, "应先后产出 rewrite 与 retrieve 事件"
    assert ticks_at_retrieve > ticks_at_rewrite, (
        "rewrite 与 retrieve 之间未让出事件循环：召回/重排阻塞会导致两事件一起刷出"
    )


def test_lexical_reranker():
    """词法重排：含查询关键词的片段被顶到前面（过滤语义噪声）。"""
    r = LexicalReranker()
    hits = [
        {"text": "公司标准上下班时间为上午9点至下午6点。", "score": 0.9},
        {"text": "一线城市出差住宿报销上限为单日450元。", "score": 0.6},
    ]
    ranked = r.rerank("出差住宿报销", hits)
    assert ranked[0]["text"] == "一线城市出差住宿报销上限为单日450元。"
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_naive_retrieve_full_default(settings):
    """naive 的 retrieve_full 契约：无重写、无重排（回归）。"""
    manager = RagManager(settings, FakeEmbeddings(), top_k=2)
    manager.ingest_all(["LangGraph 基于 StateGraph 构建 Agent。"])
    result = manager.get("naive").retrieve_full("LangGraph", top_k=2)
    assert result.rewrites == []
    assert result.reranked is False
    assert result.hits
