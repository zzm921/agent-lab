---
id: llm-gateway
name: LLM 网关与模型路由
shortDesc: 统一接入多模型 / 多供应商，按任务与成本自动路由（LiteLLM / OpenRouter 模式）。
icon: network
difficulty: adv
tags: [Gateway, Model-Routing, Multi-Provider, LiteLLM]
techFilters: []
accent: '#0ea5e9'
experience: false
prompts:
  - 简单问题用便宜模型，复杂问题用强模型。
---
## 概述

LLM 网关（LiteLLM / OpenRouter）统一封装多模型、多供应商的接入，提供模型路由、限流、密钥管理与成本统计。

## 为什么需要它

多模型并存（强模型 / 便宜模型 / 多模态）时，散装接入导致切换、密钥、账单、失败重试全部手写。

## 核心思想

统一 API 面 + 路由策略（按任务难度 / 成本 / 供应商优先级）+ 故障转移 + 限流重试 + 用量统计。

## 本项目的做法（规划中）

当前直连 DashScope 单一供应商，无网关抽象。规划：接入网关层，支持多模型路由与故障转移。

## 收益与边界

- 收益：模型可替换、成本可调、稳定；
- 边界：增加一跳延迟、网关本身要运维。

## 演进与关联

Harness 层基础能力；与 cost-governance、model-routing 配套。
