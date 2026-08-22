---
id: memory
name: 跨轮长期记忆
shortDesc: 跨会话记住关键事实并按语义召回，Agent 不再每次从零开始。
icon: history
difficulty: int
completeLevel: 90
tags: [Memory, Vector-Store, Persistence, Qdrant]
techFilters: [Qdrant]
accent: '#a855f7'
enabledTools: [memory]
---
## 为什么需要它

模型本身无状态，"记忆"来自外部存储。长期记忆把关键事实写入向量库（平台用 Qdrant），后续对话按语义召回。行业按认知科学把 Agent 记忆分为四类：工作记忆（当前上下文）、情景记忆（具体过往交互）、语义记忆（通用事实）、程序记忆（操作技能）。生产方案是短期 in-context + 长期向量组合。

## 怎么解决

难点在写入与召回的取舍——什么值得记、何时忘、召回阈值如何定。平台实现：写入时语义去重避免重复堆积，召回按相似度阈值过滤噪声，与 RAG 共享向量库（memory_tool 已实现）。

## 核心实现

```python
# 长期记忆写入与语义召回
async def recall_memory(query: str, top_k=3):
    q = await embedder.encode(query)
    hits = qdrant.search("memory", q, limit=top_k, score_threshold=0.75)
    return [h.payload["fact"] for h in hits]

async def remember(fact: str, importance=1.0):
    vec = await embedder.encode(fact)
    # 语义去重：与已有记忆太相似则更新而非追加
    dup = qdrant.search("memory", vec, limit=1, score_threshold=0.92)
    if dup:
        qdrant.update(dup[0].id, vec, {"fact": fact, "ts": now()})
    else:
        qdrant.insert(vec, {"fact": fact, "ts": now(), "importance": importance})
```

## 收益与边界

- 四类记忆模型：工作 / 情景 / 语义 / 程序，认知科学映射
- 跨会话语义召回，记住偏好与关键事实
- 短期 in-context + 长期向量组合，平衡成本与覆盖
