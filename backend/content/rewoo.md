---
id: rewoo
name: ReWOO 范式
shortDesc: Planner / Worker / Solver 三段解耦，用变量占位省去"观察后重规划"，token 高效。
icon: zap
difficulty: adv
tags: [ReWOO, Planner, Worker, Token-Efficiency]
techFilters: []
accent: '#f59e0b'
experience: false
prompts:
  - 帮我查 A 的规格，再对比 B 的价格，汇总成表格。
---
## 概述

ReWOO（Reasoning WithOut Observation）把 ReAct 的单循环拆成 Planner（规划，产出带 #E1 占位的步骤）→ Worker（逐条执行并填观察值）→ Solver（汇总作答）。

## 为什么需要它

ReAct 每观察一次就要重新进一次 LLM，token 和延迟都高。ReWOO 让规划与观察解耦，比 ReAct 更省 token、更快。

## 核心思想

一次规划产出完整步骤；Worker 只执行不推理；变量占位把结果引用回 Solver。代价：计划错了纠偏成本高。

## 本项目的做法（规划中）

现有 react / plan-execute 未覆盖 ReWOO 的变量占位与三段解耦。规划：新增 rewoo 模式（Planner → Worker → Solver 三节点 StateGraph）。

## 收益与边界

- 收益：token 降低、延迟下降；
- 边界：长链计划脆弱、观察值依赖计划正确。

## 演进与关联

介于 ReAct 与 Plan-and-Execute 之间的范式；LLMCompiler 是其并行化进阶。
