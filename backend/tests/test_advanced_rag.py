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
from app.rag.manager import RagManager
from app.rag.retrieval.reranker import LexicalReranker
from app.rag.routing.query_rewrite import LLMQueryRewriter, RuleQueryRewriter
from app.rag.schemes.advanced import (
    AdvancedRagScheme,
    CHILD_MAX,
    CHILD_MIN,
    CHUNK_MAX,
    PARENT_MAX,
    PARENT_MIN,
)


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

def test_query_rewrite_llm(monkeypatch):
    """LLM 改写：按场景懒取模型解析多行变体，且首位始终为原始查询（保证基础召回）。"""
    llm = FakeChatModel(
        script=[AIMessage(content="出差结束后需在多久内提交报销凭证\n出差报销材料提交时限")]
    )
    monkeypatch.setattr("app.rag.routing.query_rewrite.get_chat_model", lambda scenario: llm)
    rw = LLMQueryRewriter(variants=2)
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


def test_llm_rewrite_failure_keeps_original(monkeypatch):
    """LLM 调用异常（如脚本耗尽抛错）时至少保留原始查询。"""
    llm = FakeChatModel(script=[])  # 队列耗尽返回默认回答而非抛错
    monkeypatch.setattr("app.rag.routing.query_rewrite.get_chat_model", lambda scenario: llm)
    rw = LLMQueryRewriter(variants=2)
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


def test_retrieve_full_pipeline_with_llm_rewrite(settings, monkeypatch):
    """LLM 重写端到端：改写变体进入多路召回并重排。"""
    store = QdrantStore(
        FakeEmbeddings(),
        collection=f"adv_{uuid.uuid4().hex[:8]}",
        dim=32,
        sparse=True,
        client=QdrantClient(":memory:"),
    )
    llm = FakeChatModel(script=[AIMessage(content="出差结束报销凭证提交时限")])
    monkeypatch.setattr("app.rag.routing.query_rewrite.get_chat_model", lambda scenario: llm)
    scheme = AdvancedRagScheme(
        FakeEmbeddings(), store, top_k=3, reranker=LexicalReranker()
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


# ---- 结构父子分块（云帆制度语料：卷→章→节→条→表格） ----

def _clause(n: int, kw: str = "差旅报销") -> str:
    """构造一条约 120 字符的条款正文，模拟「第X条（…） …。」的制度语料风格。"""
    body = (
        f"{kw}管理规定要求所有员工出差必须提前提交出差申请，经部门主管审批通过后方可出差，"
        "出差期间产生的交通、住宿、餐饮费用需凭有效票据实报实销，逾期未报视为自动放弃，"
        "违规报销将按公司奖惩制度严肃处理，相关记录纳入年度绩效考核。"
    )
    return f"第{n}条（{kw}条款） {body}"


def _structured_volume() -> str:
    """构造含 卷/章/节/条/表格 的强层级结构文本（结构分块断言用）。"""
    clauses = "\n\n".join(_clause(i) for i in range(1, 11))  # 10 条 → 约 5 个正文子块
    small_table = (
        "| 费用类型 | 报销上限 | 说明 |\n"
        "| --- | --- | --- |\n"
        "| 高铁二等座 | 实报实销 | 单程1500公里内优先 |\n"
        "| 市内交通 | 50元/天 | 凭票据实报 |\n"
        "| 住宿费 | 450元/天 | 一线城市 |\n"
    )
    large_rows = "\n".join(f"| 参数{i} | 标准值{i} | 说明{i} |" for i in range(1, 31))
    large_table = "| 参数项 | 标准值 | 说明 |\n| --- | --- | --- |\n" + large_rows
    intro = (
        "> 本卷为公司差旅报销管理的基础性制度文件，与考勤、休假、薪酬福利、人事组织等制度共同构成公司行政管理的完整制度体系。\n"
        "> 全体员工因公出差产生的交通、住宿、餐饮等费用报销，一律以本卷全部条款及配套表单为准，其他文件与本卷不一致的以本卷为准。\n"
        "> 本卷每年十二月由人力资源部联合财务部统一复盘修订，次年一月一日生效，旧版条款自动失效。"
    )
    return (
        "# 卷九 差旅报销管理制度\n\n"
        f"{intro}\n\n"
        "## 第一章 总则\n\n"
        f"{clauses}\n\n"
        "### 1.1 差旅费用速查表\n\n"
        f"{small_table}\n\n"
        "### 1.2 差旅参数大表\n\n"
        f"{large_table}\n"
    )


def test_structure_chunking_hierarchy(settings):
    """结构分块：卷/章/节/条/表格 → 正文子块 150-250、表格原子组、父块 800-1200（章末可低）。"""
    scheme = make_advanced(settings)
    scheme.ingest([_structured_volume()])
    pairs = list(zip(scheme.store.all_texts(), scheme.store._store.metadatas))
    assert pairs, "结构语料应产出子块"
    body = [(t, m) for t, m in pairs if not m["table"]]
    tables = [(t, m) for t, m in pairs if m["table"]]
    assert body and tables, "应同时包含正文子块与表格子块"
    # 正文子块长度落在 [150, 250]
    body_lens = [len(t) for t, _ in body]
    assert all(CHILD_MIN <= n <= CHILD_MAX for n in body_lens), (
        f"正文子块应落在 [{CHILD_MIN},{CHILD_MAX}]，实际 {sorted(body_lens)}"
    )
    # 元数据溯源字段齐备：卷首引言前的子块无章/节（chapter/section 为空），章内子块带章标题
    for _, m in pairs:
        assert m["source"] == "云帆科技有限公司行政管理制度汇编.md"
        assert m["volume"] == "卷九 差旅报销管理制度"
        assert m["parent"] and isinstance(m["parent"], str)
    chaptered = [m for _, m in pairs if m["chapter"]]
    assert chaptered and all(m["chapter"] == "第一章 总则" for m in chaptered)
    assert all(m["section"] for t, m in tables), "表格所在节应记录节标题"
    # 父块长度：全部 ≤1200，且章内存在达到 [800,1200] 的完整父块（卷首引言/章末短父块就低闭合）
    unique_parents = list(dict.fromkeys(m["parent"] for _, m in pairs))
    assert all(len(p) <= PARENT_MAX for p in unique_parents), "父块不应超 1200"
    assert any(len(p) >= PARENT_MIN for p in unique_parents), "章内应有达到 [800,1200] 的完整父块"
    # 表格：≤25 行单表 1 子块且 table=True；>25 行按 25 行一组且每组重复表头
    #（表格子块带头部章/节标题，见 _structure_chunks；表头数据行位于标题行之后）
    def _header(t: str) -> str:
        for ln in t.splitlines():
            if ln.startswith("|"):
                return ln
        return ""

    small = [t for t, _ in tables if _header(t).startswith("| 费用类型")]
    large = [t for t, _ in tables if _header(t).startswith("| 参数项")]
    assert len(small) == 1, "≤25 行小表应整体为 1 个原子子块"
    assert len(large) == 2, f">25 行大表（30 行）应按 25 行切为 2 组，实际 {len(large)}"
    assert all(_header(t) == "| 参数项 | 标准值 | 说明 |" for t in large), "大表每组应重复表头"


def test_structure_chunks_falls_back_on_flat(settings):
    """无 `##` 的平坦文本：_structure_chunks 返回 [] → ingest 走语义分块兜底（回归保护）。"""
    scheme = make_advanced(settings)
    assert scheme._structure_chunks("公司规定员工每日按时打卡考勤。") == []
    scheme.ingest(["公司规定员工每日按时打卡考勤。" * 20])
    texts = scheme.store.all_texts()
    assert texts, "平坦文本应回退语义分块产出子块"
    assert all(m == {"source": "builtin"} for m in scheme.store._store.metadatas)


def test_resolve_parents_backfills_and_dedupes(settings):
    """父块回填：含 parent 的命中按父块去重、text 替换为父块全文并保留最高分；无 parent 命中原样。"""
    scheme = make_advanced(settings)
    parent = "第一章 总则\n第一条（目的） 为规范差旅报销……第二条（适用） 适用于全体员工。"
    hits = [
        {"text": "子块A", "score": 0.9, "metadata": {"parent": parent}},
        {"text": "子块B", "score": 0.8, "metadata": {"parent": parent}},
        {"text": "无父块片段", "score": 0.7, "metadata": {"source": "builtin"}},
    ]
    resolved = scheme._resolve_parents(hits)
    assert len(resolved) == 2, "同一父块应去重为 1 条"
    by_text = {h["text"]: h for h in resolved}
    assert by_text.get(parent, {}).get("score") == 0.9, "父块回填应保留最高分"
    assert by_text.get("无父块片段", {}).get("score") == 0.7, "无 parent 命中原样保留"
