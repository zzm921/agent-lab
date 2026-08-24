---
id: prompt-strategy
name: 提示词策略
shortDesc: Standard / Few-Shot / CoT 三种提示策略一键切换对比，直观感受"直接答 / 示例引导 / 逐步思考"对输出的影响。
icon: sparkles
difficulty: beg
completeLevel: 100
tags: [Prompt-Engineering, Few-Shot, CoT]
techFilters: [FastAPI]
accent: '#f59e0b'
strategy: few_shot
prompts:
  - 如何提高公司内部团队的协作效率？
  - 如何提高公司内部团队的协作效率？参考以下案例格式：案例：某技术团队通过每日15分钟站会，解决信息同步问题，效率提升20%。请用同样的“问题-对策-预期效果”三式，分析我们团队的协作问题并给出建议。
  - 如何提高公司内部团队的协作效率？请分两步回答：第一步：按顺序思考——①当前协作中的最大堵点是什么？②这些堵点最可能的原因是什么？③针对每个原因，可行的低成对策有哪些？第二步：基于以上推理，汇总出最终建议清单。
---
## 概述

提示词工程（Prompt Engineering）是 Agent 能力的基础——**不改模型，只改输入**。同一个模型、同一个任务，仅仅因为提示词的写法不同，输出质量就可能天差地别。提示词策略回答的正是这个问题：**同一句话，用什么姿势问，效果最好**。

本节聚焦三种可一键切换的策略：**Standard（直接回答）**、**Few-Shot（示例引导）**、**CoT（逐步思考）**。平台把策略参数化，在实验室里点一下按钮即可切换对比，直观感受提示策略对输出的影响。

## 为什么需要它

模型能力是固定的，"怎么问"决定它能发挥多少。没有策略意识时，常见三类问题：

- **答得笼统**：直接提问（Standard）简单直接，但复杂任务容易一句话带过，漏步骤、漏关键信息；
- **不知道"该长什么样"**：模型没被告知期望的输出格式，答非所问——Few-Shot 用示例把期望的结构教给它；
- **跳步出错**：需要推理的任务，模型"想当然"地给结论，推理过程缺失则结果必错——CoT 逼它先列推理步骤再作答。

更本质的诉求是**可切换**：提示词若写死在代码里，换一种策略就要改代码。把它做成参数化的策略表，才能对比、才能试验、才能沉淀经验。

## 三种策略的核心思想

### Standard（零样本直接问）

不提供任何示例或推理引导，直接把问题丢给模型。简单直接、成本最低，适合答案明确、无需推理的任务。这是最朴素也最常用的形态，也是其它策略的基线。

### Few-Shot（示例引导）

在提示词里给出 1~N 个「输入 → 期望输出」示例，让模型模仿示例的模式作答。按示例数量分为：

- **Zero-shot**：0 个示例，纯靠指令；
- **One-shot**：1 个示例，让模型"照猫画虎"；
- **Few-shot**：多个示例，覆盖更多形态，模型更稳。

Few-shot 的本质是**用示例替代长篇指令**——与其花几百字描述"要什么格式"，不如直接给它一个正确的样子。

### CoT（思维链）

在提示词里引导模型**先逐步推理、再给出结论**（如用"思考："标记推理过程）。把隐式的"想"变成显式的"写"，让模型在长链条上不易跳步、不易算错，也让推理过程可被人类检查。

三者在"自由度 / 稳定性 / 成本"上的取舍：

| 策略 | 一句话概括 | 适用场景 | 成本 |
|------|-----------|---------|------|
| Standard | 直接问 | 简单明确的问题 | 最低 |
| Few-Shot | 给示例照着答 | 输出格式 / 风格要统一的重复性任务 | 低 |
| CoT | 先想清楚再答 | 数学 / 逻辑 / 多步推理 | 中（多输出推理步骤） |

## 本项目的做法

平台把策略实现为**首轮 System Prompt 的映射表**——策略只改 system prompt，不碰推理循环、不碰工具调用，新增策略只需加一个 key：

```python
# app/agents/runner.py —— 策略 → 首轮 System Prompt 映射
STRATEGY_PROMPTS = {
    "standard": "你是专业的 AI 助手，请直接、准确地回答用户的问题。",
    "few_shot": (
        "你是专业的 AI 助手。请参照示例格式组织回答：\n"
        "示例：用户问'计算 2+2'→'结果为 4，计算过程：2+2=4。'\n"
        "请按此清晰结构作答。"
    ),
    "cot": "你是专业的 AI 助手。回答前请逐步思考（chain-of-thought），先列出推理步骤再给出结论。",
}

# 首次发言时把策略提示词作为 system prompt 注入（与工具重试规范拼接）
async def _make_inputs(self, graph, config, message, strategy, rag_context=None):
    snap = await graph.aget_state(config)
    msgs = []
    if snap is not None and snap.values:
        msgs = list(snap.values.get("messages", []))
    if not msgs:  # 仅首轮生效，后续轮次沿用上下文，不重复注入
        base = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["standard"])
        msgs.append(SystemMessage(content=f"{base}\n\n{TOOL_RETRY_HINT}"))
    msgs.append(HumanMessage(content=self._augment_query(message, rag_context)))
    return {"messages": msgs}
```

关键设计：

- **参数化切换**：`prompt_strategy` 随请求传入，`STRATEGY_PROMPTS.get(strategy, ...)` 取模板，换策略零侵入；
- **首轮注入**：只在会话首轮注入策略提示词，后续轮次沿用既有上下文，避免重复指令、避免污染工具执行历史；
- **与护栏解耦**：策略提示词与工具重试规范（`TOOL_RETRY_HINT`）拼接但互不影响，策略只负责"怎么答"；
- **前端一键切换**：实验室侧边栏用 Standard / Few-Shot / CoT 三个按钮切换，选择结果随请求下发后端。

## 收益与边界

**收益**

- 三种策略一键切换、直观对比，无需改代码即可做 A/B 试验；
- 策略做成数据表，新增策略只加一个 key，零侵入；
- 与推理模式、工具调用解耦，策略层可独立演进。

**边界 / 局限**

- 策略只改 system prompt，效果受模型自身能力上限约束——提示词救不了能力边界之外的任务；
- Few-Shot 的示例质量决定上限，示例写错、带偏会放大到所有同类任务；
- CoT 增加 token 消耗与首字延迟，简单任务用它是浪费。

## 演进与关联

提示词策略属于"输入侧"优化，与相邻能力互补：

```
输入侧（怎么问）                 输出侧（怎么想 / 怎么答）
提示词策略（Standard / Few-Shot / CoT）
   │  CoT 是推理增强的基础
   ├─→ 结构化输出（JSON Schema 约束格式）      —— 让输出可被程序消费
   ├─→ Self-Consistency / Tree-of-Thoughts    —— 多次采样 / 多路径搜索，更稳
   └─→ 上下文缓存（提示词是静态前缀，可命中缓存）  —— 反复复用不重复计费
```

- **与推理增强**：CoT 是最基础的推理策略，Self-Consistency、Tree-of-Thoughts 都以"会逐步推理"为前提；
- **与上下文工程**：策略提示词属于静态前缀，与动态内容分离后可配合 Prompt Caching 复用；
- **与结构化输出**：Few-Shot 给格式示例、JSON Schema 给结构约束，都是从"输入侧"提高输出的可用性。
