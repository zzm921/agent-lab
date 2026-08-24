"""高级 RAG 方案：入库语义分块 + Query 重写 + 多查询×多路混合召回 + 重排序。

对应 Advanced RAG 的「检索前后全链路优化」，解决 Naive 的固定切块断裂、纯向量
语义偏差、无关键词命中、上下文被噪声污染四大痛点：
- 入库：句子边界感知 + Embedding 相似度贪心合并（含重叠），保留嵌套规则完整语义；
- 检索前：Query 重写（LLM Multi-Query，无 LLM 规则回退）扩展查询变体；
- 检索中：每变体做稠密+稀疏混合召回（Qdrant RRF 融合），多路宽召回后去重合并；
- 检索后：交叉编码器（qwen3-rerank）精排，把真正相关的片段顶到前面。
"""
from __future__ import annotations

import re
from typing import Any

from app.memory.stores.base import StoreBackend
from app.rag.base import RagScheme, RetrieveResult
from app.rag.query_rewrite import QueryRewriter, build_rewriter
from app.rag.reranker import Reranker, build_reranker

# 语义分块参数
CHUNK_MAX = 300      # 单块最大字符数：超限强制闭合，避免超长块稀释向量语义
CHUNK_MIN = 60       # 单块最小字符数：过小的碎片不单独成块（长文本仍合并到上限）
OVERLAP_SENTENCES = 1  # 相邻块重叠句数：块尾续接上一块末句，保留跨块上下文
MERGE_THRESHOLD = 0.75  # 语义合并阈值：下一句与当前块的余弦相似度低于该值则闭合块

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
    needs_llm: bool = True  # 需要注入 LLM 做 Query 重写
    multi_backend: bool = True  # 跨后端多路召回（Qdrant + Elasticsearch 双路融合）

    def __init__(
        self,
        embeddings,
        store: StoreBackend,
        top_k: int = 3,
        llm=None,
        rewrite_variants: int = 3,
        rerank_model: str = "qwen3-rerank",
        rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
    ):
        super().__init__(embeddings, store, top_k)
        self.llm = llm
        self.rewriter = (
            rewriter if rewriter is not None else build_rewriter(llm, variants=rewrite_variants)
        )
        self.reranker = (
            reranker if reranker is not None else build_reranker(embeddings, model=rerank_model)
        )

    # ---- 入库拆分优化：语义分块 ----

    def ingest(self, texts: list[str]) -> None:
        expected = [chunk for text in texts for chunk in self._semantic_chunks(text)]
        self._rebuild_if_changed(expected)

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

    def retrieve_full(self, query: str, top_k: int | None = None) -> RetrieveResult:
        k = top_k or self.top_k
        variants = self.rewriter.rewrite(query)
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
        hits = list(candidates.values())
        # 检索后精排：交叉编码器重排（失败回退词法），取 Top-K
        hits = self.reranker.rerank(query, hits)[:k]
        return RetrieveResult(query=query, hits=hits, rewrites=variants, reranked=True)
