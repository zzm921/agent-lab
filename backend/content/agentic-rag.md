---
id: agentic-rag
name: 智能体式 RAG
shortDesc: 让 Agent 自主决定"何时检索、检索几次、结果靠不靠谱"，流程不再是固定管道。
icon: compass
difficulty: adv
tags: [Agentic-RAG, Self-RAG, CRAG, RAG]
techFilters: [LangGraph, Qdrant]
accent: '#0ea5e9'
experience: false
---
## 为什么需要它

固定流程 RAG 假设"所有问题都检索一次、检索结果都可用"。Agentic RAG 让系统成为主动决策者：Self-RAG 用反思 token 决定"要不要检索、结果是否相关、答案有没有文档支撑"；CRAG 在检索质量差时自动降级网络搜索兜底；复杂问题支持多轮迭代检索。

## 怎么解决

难点是自主检索的收敛控制——何时该停、检索多少次、如何评估检索质量。业界做法：反思 token 微调（Self-RAG）、质量评估 + 三级路由（CRAG），以及用 LangGraph 实现 Agent 驱动的迭代检索循环。

## 核心实现

```python
# Agentic RAG：Agent 自主决定检索（Self-RAG + CRAG 思路）
async def agentic_rag(query, retriever, llm, web_search):
    need = await llm.reflect("Retrieve", query)   # 要不要检索？
    if need == "no":
        return await llm.answer(query)

    hits = await retriever(query)
    relevant = [h for h in hits
                if await llm.reflect("Relevance", query, h) == "yes"]

    if not relevant:                              # CRAG：知识库兜不住
        hits = await web_search(query)

    answer = await llm.answer(query, hits)
    return answer if await llm.reflect("Support", query, answer) == "yes" \
           else await revise(query, answer)
```

## 收益与边界

- 自主决策：常识问题直接答，复杂问题多轮检索
- Self-RAG 反思：无据可依不硬答，防幻觉
- CRAG 纠错：知识库覆盖不到自动走网络兜底
