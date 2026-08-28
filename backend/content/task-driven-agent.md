---
id: task-driven-agent
name: 任务驱动自举 Agent
shortDesc: 任务队列自举 + 优先级执行（BabyAGI / AutoGPT 系），让 Agent 自主拆解并持续推进目标。
icon: list
difficulty: adv
tags: [BabyAGI, AutoGPT, Task-Queue, Autonomous]
techFilters: []
accent: '#6366f1'
experience: false
prompts:
  - 帮我制定一份产品发布计划，并逐步推进执行。
---
## 概述

任务驱动 Agent 维护一个任务队列：从目标生成任务 → 按优先级 / 依赖执行 → 执行结果反馈生成新任务（自举）。BabyAGI / AutoGPT 是其代表。

## 为什么需要它

面对长期目标，单轮对话或单循环不够——需要一个"持续经营"的执行实体。

## 核心思想

目标分解 + 任务队列（优先级排序）+ 自举（结果反哺新任务）+ 终止条件。风险：任务膨胀、跑偏、成本失控，需预算与上限。

## 本项目的做法（规划中）

现有范式均为"单次任务响应式"，无任务队列常驻。规划：task queue + 优先级调度 + 预算护栏原型。

## 收益与边界

- 收益：长目标自主推进；
- 边界：跑偏与成本失控，必须配止损。

## 演进与关联

偏"自治度"高的范式；与多智能体协作、Harness 成本护栏结合使用。
