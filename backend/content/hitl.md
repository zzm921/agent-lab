---
id: hitl
name: 审批门（人在回路）
shortDesc: 执行前审批机制，关键决策暂停等人工确认，安全与效率的平衡。
icon: check
difficulty: int
completeLevel: 95
tags: [HITL, Approval, Safety, Guardrails]
techFilters: [LangGraph]
accent: '#22d3a8'
---
## 为什么需要它

人在回路（Human-in-the-Loop）是 Harness 的"自主权边界"：Agent 在执行高风险操作前暂停，等待人工审批。审批通过才继续执行，被拒绝则终止或调整方案。平台实现：审批策略 always 时所有工具调用需审批，高危工具（如命令执行）无论策略如何都强制 HITL。

## 怎么解决

难点在于 LangGraph 的中断（interrupt）机制集成——如何在状态机执行中优雅暂停、保存检查点、等待外部信号后恢复。我利用 LangGraph 的 interrupt 功能在 action 节点前设置检查点，前端接收审批请求并回传决策。

## 核心实现

```python
# 人在回路审批节点
def approval_node(state):
    """在执行工具前暂停，等待人工审批"""
    next_tool = state["pending_tool_call"]

    # 触发中断，等待外部输入
    return interrupt({
        "type": "approval_required",
        "tool_name": next_tool["name"],
        "tool_args": next_tool["args"],
        "reason": "高风险操作需要审批",
    })

# 恢复执行
graph = build_graph()
config = {"configurable": {"thread_id": thread_id}}
result = graph.invoke(None, config)  # 从断点恢复
```

## 收益与边界

- 基于 LangGraph interrupt 原生实现，非 hack
- 支持按工具风险等级配置审批策略
- 审批状态持久化，服务重启不丢失待审任务
