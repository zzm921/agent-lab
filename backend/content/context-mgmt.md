---
id: context-mgmt
name: 上下文管理与压缩
shortDesc: 上下文窗口有限，通过摘要压缩 / 滚动 / 阈值管理，让长对话不"忘事"、不超限。
icon: cube
difficulty: int
tags: [Context-Engineering, Compaction, Window-Management]
techFilters: [LangGraph]
accent: '#84cc16'
experience: false
---
## 为什么需要它

上下文窗口是模型当前可见的"工作记忆"，容量有限。上下文管理在窗口内做取舍：接近上限时用 Compaction（结构化摘要）压缩历史、滚动替换旧消息、按阈值自动触发。摘要需保留用户目标、已确认事实、失败原因等关键信息。它与记忆（跨会话）、RAG（外部知识）分工互补——会话内靠管理、跨会话靠记忆、领域知识靠检索。

## 怎么解决

难点是压缩不丢关键信息——粗暴清空会让 Agent 在长任务中"忘记初心"；压缩策略还要与 Prompt Cache 的前缀缓存协同，避免改动靠前内容导致缓存失效、成本上升。业界做法是分层压缩流水线（截断 → 微压缩 → 上下文折叠），原始数据保留以便回滚。

## 核心实现

```python
# Compaction：接近窗口上限时压缩为结构化摘要
def maybe_compact(history, tokens, threshold=CONTEXT_THRESHOLD):
    if tokens < threshold:
        return history, tokens
    summary = llm.summarize(history, keep_fields={
        "goal": ...,          # 用户真实目标
        "decided": ...,       # 已确认的事实与决策
        "failed": ...,        # 已尝试方案与失败原因
        "todo": ...,          # 未完成事项与下一步
    })
    return [summary], len(summary)
```

## 收益与边界

- Compaction 结构化摘要替代粗暴清空，长对话不"失忆"
- 阈值触发式压缩，接近窗口上限自动收敛
- 与记忆、RAG 明确分工：会话内 / 跨会话 / 外部知识
