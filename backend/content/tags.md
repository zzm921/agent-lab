---
tags:
  - id: agent-engineering
    title: Agent 工程演进
    description: 定义轴——从"怎么说"到"谁运行"的 Prompt → Context → Harness → Loop → Graph 五层瓶颈外移地图；含输出侧与推理增强。
    groups:
      - title: 总览地图
        cards: [agent-engineering]
      - title: Harness 层（环境）
        cards: [cost-governance, llm-gateway, sandbox, fault-injection, hitl, observability-eval, security]
      - title: Context 层（喂什么）
        cards: [context-mgmt, context-caching, memory]
      - title: Prompt 层（怎么说）
        cards: [prompt-strategy, structured-output]
  - id: agent
    title: Agent 范式
    description: ReAct / 计划执行 / 反思修订 / 多智能体 / 计算机操作——四种范式 + 环境操作，回答"怎么跑"；与"Agent 工程演进"（定义轴）标签互补。
    cards: [multimodal-agent, memgpt, task-driven-agent, llm-compiler, rewoo, react, plan-execute, reflection, multi-agent, computer-use]
  - id: rag
    title: RAG 范式与工程
    description: 五代 RAG 范式演进（naive → advanced → modular → 图谱 → 智能体，总表导航打头），叠加离线处理 / 在线混合检索 / 专项增强（Self-RAG / CRAG / HyDE / RAPTOR）等工程策略与插件。
    cards: [rag, text-to-sql, kb-routing, rag-online-eval, naive-rag, advanced-rag, modular-rag, graph-rag, agentic-rag, rag-variants, offline-processing, online-hybrid-retrieval]
  - id: protocol
    title: 协议 · Protocol
    description: Agent ↔ 工具、Agent ↔ Agent 的互操作标准；函数调用 → MCP → A2A 的演进。
    cards: [agent-skills, function-calling, mcp, a2a]
---
