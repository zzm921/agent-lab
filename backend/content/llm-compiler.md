---
id: llm-compiler
name: LLMCompiler 范式
shortDesc: 把计划编译成 DAG，无依赖步骤并行执行，让 Agent 像编译器一样调度任务。
icon: code-bracket
difficulty: adv
tags: [LLMCompiler, DAG, Parallel, Task-Compiler]
techFilters: []
accent: '#06b6d4'
experience: false
prompts:
  - 查 5 个城市的天气，然后汇总对比。
---
## 概述

LLMCompiler 把"计划 + 调度 + 执行"编译成任务 DAG：并行调度器把无依赖的工具调用同时发出去，显著压低延迟。

## 为什么需要它

ReAct / ReWOO 都是串行工具调用，独立步骤白白排队。LLMCompiler 用依赖分析把可并行步骤并行化。

## 核心思想

LLM 生成计划 → 编译器拆成 DAG（含依赖边）→ 就绪任务并行执行 → 结果汇合后再答。关键在依赖解析与结果拼装。

## 本项目的做法（规划中）

现有范式均串行 / 半串行。规划：新增 DAG 并行调度模式，或给 plan-execute 加并行 executor。

## 收益与边界

- 收益：多工具独立调用时延迟大幅下降；
- 边界：依赖分析复杂、共享状态难、结果顺序需稳定。

## 演进与关联

ReWOO 的并行进阶；与 Graph 的 fan-out / fan-in 天然契合。
