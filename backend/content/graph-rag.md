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

## 推荐 Prompt

Graph RAG 有两处需要 LLM 参与：**离线建图**（文档 → 实体/关系抽取）与**在线路由**（全局性问题 vs 局部问题判定）。

### 1. 实体 / 关系抽取（离线建图）

**作用**：把文档 chunk 抽成「实体-关系」三元组并归一化实体名，作为图谱构建的输入。Graph RAG 预处理成本高，正是因为这步要对全部文档逐个 LLM 调用。

**示例 Prompt**：

```
你是知识图谱构建器。从给定文本中抽取实体及其关系，输出 JSON 列表。
要求：
1. 实体（entities）：人名/部门/制度/流程/属性等关键对象，每项含 name（归一化后的规范名）
   与 type（类型，如 person/dept/policy/process）；
2. 关系（relations）：两个实体之间的语义关系，每项含 source、target、relation
   （关系标签，如 属于/负责/规定/前置 等）；
3. 只抽取文本中明确出现的信息，不要臆造；同一实体在不同片段出现时用同一规范名。
输出必须严格是以下 JSON（不要输出任何其他文字）：
{"entities": [{"name": "...", "type": "..."}],
 "relations": [{"source": "...", "relation": "...", "target": "..."}]}
```

### 2. 全局 vs 局部问题判定（在线路由）

**作用**：决定走 Local Search（实体遍历，回答具体事实）还是 Global Search（Map-Reduce 汇总社区摘要，回答全局主题）。

**示例 Prompt**：

```
你是知识图谱检索的路由器。判断给定问题属于哪一类，输出 JSON。
- local：问题指向具体实体/事实，适合图谱内实体查找与关系遍历
  （如「A 产品的供应商的 CEO 是谁」）；
- global：问题面向整个语料的全局主题/趋势/汇总，需要遍历社区摘要汇总回答
  （如「这些文档的主要主题是什么」）。
输出必须严格是以下 JSON（不要输出任何其他文字）：
{"mode": "local|global", "reason": "一句话说明判定依据"}
```

## 收益与边界

- 实体-关系建模，弥补向量相似度的"关系盲区"
- 多跳推理：A 的供应商的 CEO 是谁
- Local / Global 双检索，兼顾局部事实与全局主题
