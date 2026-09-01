"""高级 RAG 方案：入库语义分块 + Query 重写 + 多查询×多路混合召回 + 重排序。

对应 Advanced RAG 的「检索前后全链路优化」，解决 Naive 的固定切块断裂、纯向量
语义偏差、无关键词命中、上下文被噪声污染四大痛点：
- 入库：句子边界感知 + Embedding 相似度贪心合并（含重叠），保留嵌套规则完整语义；
- 检索前：Query 重写（LLM Multi-Query，无 LLM 规则回退）扩展查询变体；
- 检索中：每变体做稠密+稀疏混合召回（Qdrant RRF 融合），多路宽召回后去重合并；
- 检索后：交叉编码器（qwen3-rerank）精排，把真正相关的片段顶到前面。
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from app.memory.stores.base import StoreBackend
from app.rag.base import RagScheme, RetrieveResult
from app.rag.retrieval.reranker import Reranker, build_reranker
from app.rag.routing.query_hyde import HydeExpander, build_hyde
from app.rag.routing.query_rewrite import QueryRewriter, build_rewriter

# 语义分块参数
CHUNK_MAX = 300      # 单块最大字符数：超限强制闭合，避免超长块稀释向量语义
CHUNK_MIN = 60       # 单块最小字符数：过小的碎片不单独成块（长文本仍合并到上限）
OVERLAP_SENTENCES = 1  # 相邻块重叠句数：块尾续接上一块末句，保留跨块上下文
MERGE_THRESHOLD = 0.75  # 语义合并阈值：下一句与当前块的余弦相似度低于该值则闭合块

# 结构父子分块参数（云帆制度语料：卷→章→节→条→表格）
CHILD_MIN = 150       # 子块下限：过短正文单元与后续合并攒长度
CHILD_MAX = 250       # 子块上限：正文子块不超限（表格原子组除外）
PARENT_MIN = 800      # 父块下限：章末不足则就低闭合
PARENT_MAX = 1200     # 父块上限：达到即闭合，超限强制断开
TABLE_GROUP_ROWS = 25  # 大表格每组分块数据行数（组首重复表头，原子不切行）
SOURCE_NAME = "云帆科技有限公司行政管理制度汇编.md"  # 结构化分块的 source 溯源

# 句边界：中文句号/问号/感叹号/分号 + 换行
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])\s*|\n+")
# 无标点超长句的硬切（退化为固定长度，防单句超长）
_CHUNK_OVERFLOW_SPLIT = re.compile(r"(?<=[\u4e00-\u9fff，,])")


class AdvancedRagScheme(RagScheme):
    """Advanced RAG：语义分块 + Query 重写 + 混合多路召回 + 重排。"""

    id: str = "advanced"
    name: str = "高级 RAG"
    description: str = "语义分块 + 混合检索 + Query重写 + Rerank 精排"
    hybrid: bool = True   # 启用稀疏向量（稠密+稀疏多路召回）
    multi_backend: bool = True  # 跨后端多路召回（Qdrant + Elasticsearch 双路融合）

    def __init__(
        self,
        embeddings,
        store: StoreBackend,
        top_k: int = 3,
        rewrite_variants: int = 3,
        rerank_model: str = "qwen3-rerank",
        rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        hyde: HydeExpander | None = None,
    ):
        super().__init__(embeddings, store, top_k)
        self.rewriter = (
            rewriter if rewriter is not None else build_rewriter(variants=rewrite_variants)
        )
        self.reranker = (
            reranker if reranker is not None else build_reranker(embeddings, model=rerank_model)
        )
        self.hyde = hyde if hyde is not None else build_hyde()

    # ---- 入库拆分优化：结构感知 + 父子分层分块 ----

    def ingest(self, texts: list[str]) -> None:
        expected: list[tuple[str, dict]] = []
        for text in texts:
            structured = self._structure_chunks(text)
            if structured:
                expected.extend(structured)
            else:
                # 无 `##` 结构（测试/历史平坦语料）回退语义分块
                expected.extend(
                    (chunk, {"source": "builtin"})
                    for chunk in self._semantic_chunks(text)
                )
        self._rebuild_if_changed(expected)

    def _doc_chunks(self, text: str, source: str) -> list[tuple[str, dict]]:
        """增量入库分块：与整批相同的分块策略，块来源为文档真实路径。"""
        structured = self._structure_chunks(text, source)
        if structured:
            return structured
        return [(chunk, {"source": source}) for chunk in self._semantic_chunks(text)]

    def _structure_chunks(self, text: str, source: str = SOURCE_NAME) -> list[tuple[str, dict]]:
        """结构感知父子分块：解析卷/章/节标题与表格块，产出子块并聚合父块。

        子块：正文 150-250（短单元贪心合并、超长单元句边界切分）、表格 25 行原子组；
        父块：章内 800-1200，父块全文写入各子块 metadata["parent"] 供检索回填。
        全文无 `##` 章节结构（平坦语料/测试文本）返回 []，调用方回退语义分块。
        source：溯源来源（整批内置语料用默认名，增量入库传文档真实路径）。
        """
        lines = text.split("\n")
        h2 = sum(1 for line in lines if line.startswith("## "))
        h3 = sum(1 for line in lines if line.startswith("### "))
        if h2 == 0:
            # 无 ## 章：有 ### 条目（FAQ/案例/场景/卡片等条目式卷）→ 条目分块，
            # 避免「无结构回退语义分块」把多个问答/案例混切进同块且丢失卷/条目元数据
            return self._entry_chunks(text, source) if h3 else []
        if h3 > h2 * 10:
            # 条目式主导（如 FAQ 卷仅一个 ## 分编、### 问答上百条）→ 条目分块优先：
            # 主路径贪心合并会把相邻问答答案拼进同块，破坏一问一答原子性与 section 归属。
            # 阈值 10 倍：层级卷（章多节少，如 15 章 75 节）仍走结构分块保留表格原子组
            return self._entry_chunks(text, source)
        # 结构解析：按空行分块，识别 卷/章/节 标题上下文与表格块，其余为正文单元
        units: list[dict] = []
        volume = chapter = section = ""
        for block in re.split(r"\n\s*\n", text):
            b = block.strip()
            if not b:
                continue
            blines = [line.strip() for line in b.split("\n")]
            first = blines[0]
            if first.startswith("# "):
                volume = first[2:].strip()
                chapter = section = ""
                continue
            if first.startswith("## "):
                chapter = first[3:].strip()
                section = ""
                continue
            if first.startswith("### "):
                section = first[4:].strip()
                continue
            ctx = {"volume": volume, "chapter": chapter, "section": section}
            if first.startswith("|"):
                # 表格单元：表头 + 分隔行 + 数据行；>25 行按 25 行一组、组首重复表头
                data_rows = blines[2:] if len(blines) > 1 else []
                group_count = max(1, (len(data_rows) + TABLE_GROUP_ROWS - 1) // TABLE_GROUP_ROWS)
                for g in range(group_count):
                    group_rows = [first] + (blines[1:2] if len(blines) > 1 else []) + data_rows[g * TABLE_GROUP_ROWS:(g + 1) * TABLE_GROUP_ROWS]
                    units.append({"kind": "table", "text": "\n".join(group_rows), **ctx})
                continue
            units.append({"kind": "para", "text": "\n".join(blines), **ctx})
        # 子块构造：按章分组（不跨章），正文贪心合并至 [150,250]、表格 25 行原子组
        children: list[dict] = []
        grouped: list[list[dict]] = []
        cur_group: list[dict] = []
        cur_chapter = ""
        for u in units:
            if u["chapter"] != cur_chapter:
                if cur_group:
                    grouped.append(cur_group)
                cur_group = []
                cur_chapter = u["chapter"]
            cur_group.append(u)
        if cur_group:
            grouped.append(cur_group)
        for group in grouped:
            chapter = group[0]["chapter"]
            volume = group[0]["volume"]
            buffer: list[str] = []
            buf_len = 0
            buf_section = group[0]["section"]
            for u in group:
                if u["kind"] == "table":
                    if buffer:
                        children.append(self._child_block("\n\n".join(buffer), volume, chapter, buf_section, False))
                        buffer, buf_len = [], 0
                    # 表格子块带章/节标题：表格本体常无引导段落，光秃表格嵌入与关键词都难命中
                    #（回归：「第五章 各部门人员规模与编制」表因缺标题，查「部门人数」无法召回）。
                    heading = " / ".join(x for x in (u["chapter"], u["section"]) if x) or u["volume"]
                    table_text = f"{heading}\n{u['text']}" if heading else u["text"]
                    children.append(self._child_block(table_text, u["volume"], u["chapter"], u["section"], True))
                    continue
                pieces = self._split_long_body(u["text"]) if len(u["text"]) > CHILD_MAX else [u["text"]]
                for p in pieces:
                    # 跨 section 不合并：相邻条目（### 问题）答案拼块会错配 section 归属
                    if buffer and (buf_len + len(p) > CHILD_MAX or u["section"] != buf_section):
                        children.append(self._child_block("\n\n".join(buffer), volume, chapter, buf_section, False))
                        buffer, buf_len = [], 0
                    buffer.append(p)
                    buf_len += len(p)
                    buf_section = u["section"]
            if buffer:
                children.append(self._child_block("\n\n".join(buffer), volume, chapter, buf_section, False))
        # 父块聚合：章内按文档顺序串联子块至 [800,1200]，父块文本 = 章/节标题 + 子块串联
        parents: dict[int, str] = {}
        p_window: list[tuple[int, str]] = []
        p_len = 0
        p_chapter = ""
        p_volume = ""
        p_section = ""
        for idx, child in enumerate(children):
            if p_window and (child["chapter"] != p_chapter or p_len + len(child["text"]) + 2 > PARENT_MAX):
                anchor = (p_chapter or p_volume) + ("\n" + p_section if p_section else "")
                body = "\n\n".join(t for _, t in p_window)
                # 表格子块文本已带头部章标题时，不再重复拼接锚点标题
                parent_text = (anchor + "\n" + body) if not body.startswith(anchor) else body
                for ci, _ in p_window:
                    parents[ci] = parent_text
                p_window, p_len = [], 0
            p_window.append((idx, child["text"]))
            p_len += len(child["text"]) + 2
            p_chapter = child["chapter"]
            p_volume = child["volume"]
            p_section = child["section"]
        if p_window:
            anchor = (p_chapter or p_volume) + ("\n" + p_section if p_section else "")
            body = "\n\n".join(t for _, t in p_window)
            parent_text = (anchor + "\n" + body) if not body.startswith(anchor) else body
            for ci, _ in p_window:
                parents[ci] = parent_text
        return [
            (
                child["text"],
                {
                    "source": source,
                    "volume": child["volume"],
                    "chapter": child["chapter"],
                    "section": child["section"],
                    "table": child["table"],
                    "parent": parents[i],
                },
            )
            for i, child in enumerate(children)
        ]

    def _entry_chunks(self, text: str, source: str = SOURCE_NAME) -> list[tuple[str, dict]]:
        """条目式卷分块（`# 卷标题` + `### 条目`，无 `##` 章）：一条目一子块。

        FAQ/案例/场景/知识卡片卷的条目是天然原子单元（一问一答/一案例/一场景），
        按条目切分保证问答不混块；条目即父块（parent = 卷标题 + 条目全文），
        metadata 带 volume/section，与结构化卷一致（不再走 builtin 语义分块回退）。
        """
        volume = ""
        items: list[list[str]] = []  # 每条目 = [行文本...]（标题行为首行）
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                continue
            if s.startswith("# ") and not s.startswith("##"):
                volume = s[2:].strip()
                continue
            if s.startswith("### "):
                items.append([s[4:].strip()])
                continue
            if items:
                items[-1].append(s)
            else:
                items.append([s])  # 卷标题后、首个条目前的导语：独立条目，不丢弃
        children: list[dict] = []
        for lines in items:
            entry_text = "\n".join(lines)
            if not entry_text.strip():
                continue
            section = self._entry_section(entry_text)
            parent = f"{volume}\n{entry_text}" if volume else entry_text
            if len(parent) > PARENT_MAX:
                parent = parent[:PARENT_MAX]
            pieces = self._split_long_body(entry_text) if len(entry_text) > CHILD_MAX else [entry_text]
            for p in pieces:
                children.append(
                    {"text": p, "volume": volume, "chapter": "", "section": section, "table": False, "parent": parent}
                )
        return [
            (
                c["text"],
                {
                    "source": source,
                    "volume": c["volume"],
                    "chapter": c["chapter"],
                    "section": c["section"],
                    "table": c["table"],
                    "parent": c["parent"],
                },
            )
            for c in children
        ]

    @staticmethod
    def _entry_section(entry_text: str) -> str:
        """条目 section 标识：首句（到第一个句读符号），截 40 字。"""
        head = re.split(r"[。？！；\n]", entry_text, 1)[0].strip()
        return head[:40]

    def _split_long_body(self, text: str) -> list[str]:
        """超长正文单元：按句边界（。！？；/换行）切成约 CHILD_MAX 的子块；
        无句边界的超长段按汉字/逗号边界兜底硬切，保证任一块不超上限。"""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        pieces, cur = [], ""
        for s in sentences:
            if cur and len(cur) + len(s) > CHILD_MAX:
                pieces.append(cur)
                cur = s
            else:
                cur += s
        if cur:
            pieces.append(cur)
        out: list[str] = []
        for p in pieces:
            if len(p) <= CHILD_MAX:
                out.append(p)
                continue
            sub, buf = [], ""
            for part in [x for x in _CHUNK_OVERFLOW_SPLIT.split(p) if x]:
                if len(buf) + len(part) > CHILD_MAX and buf:
                    sub.append(buf)
                    buf = part
                else:
                    buf += part
            if buf:
                sub.append(buf)
            out.extend(sub or [p])
        return out

    @staticmethod
    def _child_block(text: str, volume: str, chapter: str, section: str, table: bool) -> dict:
        """构造一个子块记录（含溯源上下文，父块聚合后回填 parent 元数据）。"""
        return {
            "text": text,
            "volume": volume,
            "chapter": chapter,
            "section": section,
            "table": table,
        }

    def _semantic_chunks(self, text: str) -> list[str]:
        """句子边界感知的语义分块：贪心按语义相似度合并，块尾重叠续接上一句。"""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return []
        chunks: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            # 单句超长（无标点长文本）：退化为固定长度硬切（保留最朴素兜底）
            if len(sentence) > CHUNK_MAX:
                if current:
                    chunks.append("".join(current))
                    current = []
                chunks.extend(self._split_long(sentence))
                continue
            if current and not self._should_merge("".join(current), sentence):
                last = current[-1]
                chunks.append("".join(current))
                # 重叠续接上一句：仅当上一块多于一句且不超限时携带，避免单句块重复
                current = (
                    [last]
                    if len(current) > 1 and len(last) + len(sentence) <= CHUNK_MAX
                    else []
                )
            current.append(sentence)
        if current:
            chunks.append("".join(current))
        return [c for c in chunks if c]

    def _should_merge(self, acc: str, sentence: str) -> bool:
        """是否把下一句并入当前块：语义相近（余弦 ≥ 阈值）且不超长度上限。"""
        if len(acc) + len(sentence) > CHUNK_MAX:
            return False
        if len(acc) < CHUNK_MIN or len(sentence) < CHUNK_MIN:
            return True  # 太短不判定语义，直接合并攒长度
        return self._cosine(self.embeddings.embed_query(acc), self.embeddings.embed_query(sentence)) >= MERGE_THRESHOLD

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _split_long(self, sentence: str) -> list[str]:
        """无标点超长句的兜底硬切：优先在汉字/逗号边界断开，块间不重叠。"""
        pieces = [p for p in _CHUNK_OVERFLOW_SPLIT.split(sentence) if p]
        chunks, cur = [], ""
        for piece in pieces:
            if len(cur) + len(piece) > CHUNK_MAX and cur:
                chunks.append(cur)
                cur = piece
            else:
                cur += piece
        if cur:
            chunks.append(cur)
        return chunks or [sentence]

    # ---- 检索：Query 重写 + 多查询×多路召回 + 重排 ----

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.retrieve_full(query, top_k).hits

    def _multi_recall_rerank(self, query: str, variants: list[str], k: int) -> list[dict[str, Any]]:
        """多查询×多路宽召回 + 精排，返回 Top-K 命中（rewrite 由调用方先行完成）。"""
        # 多路宽召回：每个查询变体分别走「稠密语义路」与「混合路」两条互补路径——
        # - 稠密路 search：纯向量语义召回，同义不同词也能命中；
        # - 混合路 hybrid_search：稠密+稀疏 RRF（Qdrant）或 kNN + BM25（ES），
        #   multi_backend 下跨 Qdrant+ES 双库融合，补足专有名词/编号/精确表达的精确命中。
        # 各路结果按文本去重合并（保留最高分），形成宽候选集，再交给精排压缩噪声。
        recall_k = max(k * 3, 9)
        candidates: dict[str, dict[str, Any]] = {}
        for variant in variants:
            for hit in self.store.search(variant, recall_k):        # 稠密语义路
                candidates.setdefault(hit.get("text", ""), hit)
            for hit in self.store.hybrid_search(variant, recall_k):  # 混合路（稠密+稀疏/关键词）
                candidates.setdefault(hit.get("text", ""), hit)
        # HyDE：用 LLM 生成的假想答案文档做一次稠密 doc-space 召回（规则回退时为原查询，跳过）
        hyde_doc = self.hyde.expand(query)
        if hyde_doc and hyde_doc != query:
            for hit in self.store.search(hyde_doc, recall_k):
                candidates.setdefault(hit.get("text", ""), hit)
        hits = list(candidates.values())
        # 检索后精排：交叉编码器重排（失败回退词法），取 Top-K
        return self.reranker.rerank(query, hits)[:k]

    def _resolve_parents(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """检索后父块回填：metadata 含 parent 的命中以父块全文替换 text，同一父块去重保留最高分。

        回填时机在重排/压缩之后：重排基于子块（精准定位），注入 LLM 的上下文用父块（完整）；
        无 parent 的命中（语义分块回退路径）原样保留（no-op 安全）。
        """
        best: dict[str, dict[str, Any]] = {}
        for h in hits:
            parent = (h.get("metadata") or {}).get("parent")
            text = parent if parent else h.get("text", "")
            if text in best:
                if (h.get("score") or 0) > (best[text].get("score") or 0):
                    best[text]["score"] = h["score"]
                continue
            out = dict(h)
            out["text"] = text
            best[text] = out
        return list(best.values())

    def retrieve_full(self, query: str, top_k: int | None = None, context: str | None = None) -> RetrieveResult:
        """同步完整检索结果（供非流式场景/测试）；页面事件走 astream 流式下发。"""
        k = top_k or self.top_k
        variants = self.rewriter.rewrite(query)
        hits = self._multi_recall_rerank(query, variants, k)
        hits = self._resolve_parents(hits)  # 子块命中回填父块全文，供 LLM 完整上下文
        return RetrieveResult(query=query, hits=hits, rewrites=variants, reranked=True)

    async def astream(self, query: str, top_k: int | None = None, context: str | None = None):
        """异步流式检索：重写一结束立即 yield 重写事件，再召回/重排后 yield 检索事件。

        召回/重排为同步阻塞调用（向量库/重排模型的同步 HTTP），放线程池执行，
        避免阻塞事件循环导致上面的重写事件与检索事件被攒到同一次 SSE 刷出、
        在前端「同时」展示——放线程池后事件循环保持畅通，重写事件先行下发。
        """
        k = top_k or self.top_k
        variants = self.rewriter.rewrite(query)
        if variants:
            yield {
                "type": "rewrite",
                "query": query,
                "scheme": self.id,
                "rewrites": variants,
            }
        hits = await asyncio.to_thread(self._multi_recall_rerank, query, variants, k)
        hits = self._resolve_parents(hits)  # 子块命中回填父块全文，供 LLM 完整上下文
        yield {
            "type": "retrieve",
            "query": query,
            "scheme": self.id,
            "hits": hits,
            "reranked": True,
        }
