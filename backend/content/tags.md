---
tags:
  - id: agent-engineering
    title: Agent 工程演进
    description: 定义轴——从"怎么说"到"谁运行"的 Prompt → Context → Harness → Loop → Graph 五层瓶颈外移地图；含输出侧与推理增强。
    groups:
      - title: 总览地图
        cards: [agent-engineering]
      - title: Prompt 层（怎么说）
        cards: [prompt-strategy, structured-output]
      - title: Context 层（喂什么）
        cards: [context-mgmt, context-caching, memory]
      - title: Harness 层（环境）
        cards: [cost-governance, llm-gateway, sandbox, fault-injection, hitl, security]
  - id: agent
    title: Agent 范式
    description: ReAct / 计划执行 / 反思修订 / 多智能体 / 计算机操作——四种范式 + 环境操作，回答"怎么跑"；与"Agent 工程演进"（定义轴）标签互补。
    cards: [react, plan-execute, reflection, rewoo, llm-compiler, multi-agent, task-driven-agent, computer-use, multimodal-agent, memgpt]
  - id: rag
    title: RAG 范式与工程
    description: 五代 RAG 范式演进（naive → advanced → modular → 图谱 → 智能体，总表导航打头），叠加离线处理 / 在线混合检索 / 专项增强（Self-RAG / CRAG / HyDE / RAPTOR）等工程策略与插件。
    cards: [rag, naive-rag, advanced-rag, modular-rag, graph-rag, agentic-rag, rag-variants, offline-processing, online-hybrid-retrieval, kb-routing, text-to-sql, rag-eval, rag-online-eval]
  - id: protocol
    title: 协议 · Protocol
    description: Agent ↔ 工具、Agent ↔ Agent 的互操作标准；函数调用 → MCP → A2A 的演进。
    cards: [function-calling, mcp, a2a, agent-skills]
  - id: eval
    title: 评估评测
    description: 从离线确定性断言到线上反馈闭环——L1/L2 分层评测、RAGAS 指标与在线评测闭环，回答"Agent 到底行不行"。
    cards: [agent-eval, rag-eval, rag-online-eval, observability-eval]
  - id: ops
    title: 生产与治理
    description: 让 Agent 在真实环境跑得住——成本与延迟治理、模型网关、沙箱、故障容错、审批门与安全防护（与工程演进 Harness 层互为交叉视图）。
    cards: [cost-governance, llm-gateway, sandbox, fault-injection, hitl, security]
---
