"""RAG 方案抽象：每个方案 = 一个独立 Qdrant 集合 + 一种检索策略。

方案层只依赖 StoreBackend 接口，与具体存储后端（Qdrant/ES）解耦；
graph / modular / agentic 后续各实现一个 RagScheme 子类即可扩展。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.memory.stores.base import StoreBackend


@dataclass
class RetrieveResult:
    """一次检索的完整结果：命中 + 检索过程信息（供前端展示）。"""

    query: str
    hits: list[dict[str, Any]]
    rewrites: list[str] = field(default_factory=list)  # Query 重写变体（naive 为空）
    reranked: bool = False  # 是否经过重排
    decomposed: list[str] = field(default_factory=list)  # Query 分解子问题（decompose 路径）
    compressed: dict[str, int] | None = None  # 上下文压缩统计（{"original","kept","truncated"}）
    hops: list[dict[str, Any]] = field(default_factory=list)  # 多跳检索逐跳记录（[{"query","hits","target","skipped"}]，非多跳为空）
    plan: dict[str, Any] | None = None  # 多跳检索计划（规划-执行-验证：steps+reason，非多跳为空）
    verification: dict[str, Any] | None = None  # 多跳质量闸门结果（{"covered","missing","patched"}，非多跳为空）
    answerability: dict[str, Any] | None = None  # 答案充分性验证（{"answerable","missing_facts","recommendation","escalate_to"}）


class RagScheme(ABC):
    """RAG 方案基类：定义 id/name/集合名，以及入库与检索行为。"""

    id: str = ""
    name: str = ""
    collection: str = ""
    description: str = ""
    # 该方案集合是否启用稀疏向量（混合检索）；manager 据此构建稀疏后端
    hybrid: bool = False
    # 该方案是否跨后端多路召回（同时查询 Qdrant + Elasticsearch，融合去重）
    multi_backend: bool = False

    def __init__(self, embeddings, store: StoreBackend, top_k: int = 3):
        self.embeddings = embeddings
        self.store = store
        self.top_k = top_k
        self.collection = store.collection  # 集合名随绑定后端（如 knowledge_naive）

    @abstractmethod
    def ingest(self, texts: list[str]) -> None:
        """把语料写入本方案独立的集合（幂等：集合非空则跳过）。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """按本方案策略检索，返回 [{text, score, metadata}]。"""

    def retrieve_full(self, query: str, top_k: int | None = None, context: str | None = None) -> RetrieveResult:
        """同步完整检索结果（供非流式场景/测试）；页面事件走 astream 流式下发。

        context：最近会话上下文（指代消解用，naive 等无需消解的方案忽略）。
        """
        return RetrieveResult(query=query, hits=self.retrieve(query, top_k))

    async def astream(self, query: str, top_k: int | None = None, context: str | None = None):
        """异步流式检索：按阶段即时产出事件，供 runner 经 async for 直达前端。

        默认实现（naive）无重写阶段，仅产出检索命中事件；子类可先 yield 重写事件再检索。
        context：最近会话上下文（指代消解用，naive 等无需消解的方案忽略）。
        """
        hits = self.retrieve(query, top_k)
        yield {
            "type": "retrieve",
            "query": query,
            "scheme": self.id,
            "hits": hits,
            "reranked": False,
        }

    def _rebuild_if_changed(self, expected: list[str]) -> None:
        """按预期分块列表重建集合：指纹未变则幂等跳过，语料变更则清空重建。"""
        if len(self.store) > 0 and sorted(expected) == sorted(self.store.all_texts()):
            return  # 语料未变化：幂等跳过，避免重复向量化
        if len(self.store) > 0:
            self.store.clear()  # 语料已更新：清空后按新语料重建
        for chunk in expected:
            self.store.add(chunk, {"source": "builtin"})

    def __len__(self) -> int:
        return len(self.store)
