---
id: cost-governance
name: 成本与延迟治理
shortDesc: token 预算、Prompt Cache、模型分级与并发管控，让 Agent 跑得起也付得起。
icon: cpu
difficulty: adv
tags: [Cost, Latency, Token-Budget, Optimization]
techFilters: []
accent: '#22c55e'
experience: false
prompts:
  - 这个 Agent 每月大概要花多少钱？
---
## 概述

Agent 是"每次调用都烧钱烧时间"的实体。成本与延迟治理：预算上限、缓存复用、小模型分流、并发与重试管控。

## 为什么需要它

循环 + 多步工具调用让成本随轮数爆炸；不加护栏的 Agent 是"无限预算黑洞"。

## 核心思想

Token 预算（会话级 / 轮次级上限）+ Prompt Cache（前缀复用）+ 模型分级（轻任务用小模型）+ 并行化降延迟 + 观测成本归因。

## 本项目的做法（规划中）

react 已用 ModelCallLimitMiddleware 限轮数，但无统一成本观测与预算护栏。规划：tokens 统计落库 + 会话预算 + 缓存复用策略。

## 收益与边界

- 收益：成本可控、延迟下降；
- 边界：过度压成本伤质量，需分级权衡。

## 演进与关联

Harness 层的执行控制职责；与 observability-eval（成本指标）、llm-gateway（模型路由）配套。
