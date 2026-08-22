---
id: graph-rag
name: 知识图谱 RAG
shortDesc: 实体-关系建模 + 多跳推理，回答"跨文档关联"的全局性问题。
icon: globe
difficulty: adv
tags: [GraphRAG, Knowledge-Graph, Multi-Hop, RAG]
techFilters: [Qdrant]
accent: '#06b6d4'
experience: false
---
## 为什么需要它

向量检索擅长"一对一的局部片段匹配"，却看不懂跨文档的宏观结构与实体间关系。Graph RAG（微软 2024）把文档提炼成实体-关系知识图谱并预构建社区摘要，支持多跳推理（"A 产品的供应商的 CEO 是谁"）与全局性问题（"这些文档的主要主题是什么"）。

## 怎么解决

难点在知识图谱构建成本——从文档抽取实体关系需要大量 LLM 调用；检索阶段需区分 Local Search（实体遍历）与 Global Search（Map-Reduce 汇总社区摘要）。业界以 LightRAG / KAG 等轻量图方案降低预处理成本。

## 核心实现

```python
# Graph RAG 双检索：Local（实体遍历） vs Global（社区摘要汇总）
def graph_rag_query(query):
    # Local：图谱内实体查找 + 关系遍历
    local = kg.find_entities(query_entities(query))
    if is_global_question(query):          # 全局性问题
        global_ = map_reduce(
            [summarize(c) for c in kg.community_summaries],
            aggregate=combine,
        )
        return global_
    return local
```

## 收益与边界

- 实体-关系建模，弥补向量相似度的"关系盲区"
- 多跳推理：A 的供应商的 CEO 是谁
- Local / Global 双检索，兼顾局部事实与全局主题
