---
id: agentic-rag
name: 智能体式 RAG
shortDesc: 让 Agent 自主决定"何时检索、检索几次、结果靠不靠谱"，流程不再是固定管道。
icon: compass
difficulty: adv
tags: [Agentic-RAG, Self-RAG, CRAG, RAG]
techFilters: [LangGraph, Qdrant]
accent: '#0ea5e9'
enabledTools: [rag]
rag_scheme: agentic
prompts:
  - 请对比差旅报销和日常报销的完整流程差异，并说明依据
  - 研发部主管王刚的上级是谁？他分管哪些部门？
  - 张三入职以来的年假权益是怎么算的？
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

## 推荐 Prompt

Agentic RAG 的自主性来自模型**反思决策**。每条分支（要不要检索 / 结果相不相关 / 答案有没有支撑 / 检索质量够不够）各配一个轻量判定 Prompt——全部是「结构化 JSON 输出」的低延迟调用。

### 1. 检索必要性（Self-RAG · Retrieve）

**作用**：决定这条问题要不要检索——寒暄/常识直接答，事实/领域问题才查库（省 token、防无关检索污染）。

**示例 Prompt**：

```
你是 RAG 系统的检索决策器。判断给定问题是否需要检索外部知识库，输出 JSON。
- 常识/寒暄/主观问题，模型自身知识足够 → no；
- 事实性/领域性/需要最新信息/需要引用来源 → yes。
输出必须严格是以下 JSON（不要输出任何其他文字）：
{"need_retrieval": true/false, "reason": "一句话"}
```

### 2. 相关性判定（Self-RAG · Relevance）

**作用**：对每个检索片段判断与当前问题的相关性，过滤无关噪声（替代一刀切 Top-K）。

**示例 Prompt**：

```
你是 RAG 系统的相关性判断器。给定用户问题与一个检索片段，
判断该片段是否与回答该问题相关，输出 JSON。
- 片段包含回答问题所需的信息或直接相关 → yes；
- 只沾边但无实质信息 → no。
输出必须严格是以下 JSON（不要输出任何其他文字）：
{"relevant": true/false, "reason": "一句话"}
```

### 3. 支撑度判定（Self-RAG · Support）

**作用**：生成后校验「答案是否真的被检索片段支撑」——无据可依不硬答（防幻觉），不支撑则触发修订/重答。

**示例 Prompt**：

```
你是 RAG 系统的答案支撑度判断器。给定用户问题、检索片段与模型生成答案，
判断答案是否被检索片段充分支撑，输出 JSON。
- 答案的所有关键论断都能在片段中找到对应依据 → yes；
- 存在片段不支撑、甚至相矛盾的论断 → no（须在 reason 中说明哪一句无依据）。
输出必须严格是以下 JSON（不要输出任何其他文字）：
{"supported": true/false, "reason": "一句话说明依据或缺口"}
```

### 4. 检索质量评估（CRAG · Corrective）

**作用**：检索后评估质量分级——高质量直接用 / 中质量修正后重检 / 低质量走网络搜索兜底。

**示例 Prompt**：

```
你是 RAG 系统的检索质量评估器。给定用户问题与检索片段集，
评估本次检索结果质量，输出 JSON。
- accurate：片段与问题高度相关且包含充分答案 → correct；
- ambiguous：部分相关但含无关/重复噪声，或答案不完整 → incorrect（触发修正后重检）；
- missing：几乎不相关，知识库兜不住 → missing（触发网络搜索兜底）。
输出必须严格是以下 JSON（不要输出任何其他文字）：
{"grade": "correct|incorrect|missing", "reason": "一句话"}
```

## 收益与边界

- 自主决策：常识问题直接答，复杂问题多轮检索
- Self-RAG 反思：无据可依不硬答，防幻觉
- CRAG 纠错：知识库覆盖不到自动走网络兜底
