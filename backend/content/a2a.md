---
id: a2a
name: A2A 智能体通信
shortDesc: Agent 间发现、委托与协作的开放协议，跨框架互操作的"普通话"。
icon: arrows-right-left
difficulty: adv
tags: [A2A, Protocol, Agent-Interop, Agent-Card]
techFilters: [MCP]
accent: '#ec4899'
experience: false
---
## 为什么需要它

A2A（Agent-to-Agent Protocol，Google 发起，现归 Linux 基金会 / AAIF 治理）定义 Agent 之间如何发现彼此、委托任务与协作。核心机制：Agent Card 名片声明能力、Task 生命周期状态机（submitted → working → completed / failed）、任务委托与异步回调。与 MCP 互补——MCP 解决"用什么工具干活"，A2A 解决"找谁一起干活"。

## 怎么解决

难点在跨框架互操作与安全——各框架（LangGraph / CrewAI / AutoGen）API 各异；Agent 间信任与鉴权、内部实现对外的"不透明性"边界。业界做法：Agent Card + 标准任务信封 + OAuth 鉴权，仅暴露能力不暴露实现。

## 核心实现

```python
# A2A：Agent Card 名片 + 任务委托
AGENT_CARD = {
    "name": "research-agent",
    "skills": ["文献调研", "竞品分析"],
    "endpoint": "https://api.example.com/a2a",
    "auth": "OAuth2",
}

# 客户端委托任务（JSON-RPC over HTTP + SSE）
task = await a2a_client.start_task(
    card=AGENT_CARD,
    message={"type": "text",
             "text": "调研 2026 年 RAG 主流范式"},
    push_notifications=True,
)
# task.status: submitted → working → completed | failed | input-required
```

## 收益与边界

- Agent Card 名片发现 + 能力协商
- 任务生命周期状态机，委托 / 流式 / 异步回调
- 与 MCP 互补协作，构成 2026 双协议栈
