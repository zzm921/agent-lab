---
tags:
  - id: prompt-engineering
    title: 提示词工程
    description: 不改模型，用输入格式 / 示例 / 推理引导把模型能力引出来。
    cards: [prompt-strategy, structured-output]
  - id: context-engineering
    title: 上下文工程
    description: 管理"模型当前看到什么"——窗口规划、信息压缩、缓存与渐进式披露；RAG 是其核心手段之一（见 RAG 标签）。
    cards: [context-mgmt, context-caching]
  - id: rag
    title: RAG · 检索增强生成
    description: 把外部知识接进生成，让回答有据可查；总表导航 + 五代范式演进（naive → advanced → modular → 图谱 → 智能体）+ 离线/在线工程策略 + 专项增强技术。
    cards: [rag, naive-rag, advanced-rag, modular-rag, graph-rag, agentic-rag, rag-variants, offline-processing, online-hybrid-retrieval]
  - id: agent
    title: Agent · 智能体
    description: 让模型会想、会做、会协作、会操作——从单 Agent 推理到多 Agent 编排与计算机操作。
    cards: [react, plan-execute, reflection, multi-agent, computer-use]
  - id: harness
    title: Harness · 强化工程
    description: Agent = Model + Harness；记忆 / 沙箱 / 容错 / 审批 / 观测 / 安全交给外部环境。
    cards: [memory, sandbox, fault-injection, hitl, observability-eval, security]
  - id: protocol
    title: 协议 · Protocol
    description: Agent ↔ 工具、Agent ↔ Agent 的互操作标准；函数调用 → MCP → A2A 的演进。
    cards: [function-calling, mcp, a2a]
---
