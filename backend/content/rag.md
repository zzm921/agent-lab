---
id: rag
name: 检索增强生成
shortDesc: 向量检索 + 增强生成，让回答有据可查、不靠幻觉。
icon: database
difficulty: int
completeLevel: 85
tags: [RAG, Qdrant, Embedding, Hybrid-Search, Rerank]
techFilters: [Qdrant, FastAPI]
accent: '#38bdf8'
enabledTools: [rag]
---
## 为什么需要它

RAG（Retrieval-Augmented Generation）把知识库文档向量化存储，回答问题时先检索最相关片段，再基于检索结果生成答案，有效减少幻觉、回答有据可查。生产实践通常在朴素"向量检索 + 生成"之上叠加三层优化：分层切分、混合检索、Reranker 重排序。

## 怎么解决

难点在于检索质量优化——切分策略、embedding 模型、相似度阈值、重排序都影响最终效果。平台实现了分层切分（小 chunk 精确定位 + parent chunk 完整上下文）、混合检索（向量语义 + BM25 关键词）、Reranker 精排三层优化。

## 核心实现

```python
# RAG 检索服务：混合检索 + 重排序
class RAGRetriever:
    def __init__(self, qdrant_client, embedder):
        self.client = qdrant_client
        self.embedder = embedder

    async def search(self, query, top_k=5):
        # 1. 向量检索
        query_vec = await self.embedder.encode(query)
        vector_results = self.client.search(
            collection_name="docs",
            query_vector=query_vec,
            limit=top_k * 2,
        )
        # 2. 关键词检索（BM25）
        keyword_results = self.bm25.search(query, top_k * 2)
        # 3. 融合重排序
        fused = self.reciprocal_rank_fusion(
            vector_results, keyword_results
        )
        return self.reranker.rerank(query, fused[:top_k])
```

## 收益与边界

- 混合检索：向量语义 + 关键词精确，召回率更高
- 分层切分：小 chunk 精确定位 + parent chunk 完整上下文
- Reranker 重排序，Top-K 准确率提升显著
