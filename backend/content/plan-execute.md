---
id: plan-execute
name: 计划执行
shortDesc: 先拆解任务为子步骤，再逐个执行，复杂任务的成功率大幅提升。
icon: list
difficulty: int
completeLevel: 90
tags: [Planning, Decomposition, Task-Graph, Agent]
techFilters: [LangGraph]
accent: '#38bdf8'
mode: plan_execute
---
## 为什么需要它

计划执行（Plan-and-Execute）与 ReAct 的"边走边看"不同：先让 Agent 把复杂任务拆解为有序的子任务列表，再逐个执行并根据结果动态调整计划。计划可审查、可干预，适合多步骤、依赖关系复杂的任务。

## 怎么解决

难点在于计划的动态调整——执行中发现子任务不可行时如何重新规划，以及子任务之间的结果传递和依赖管理。我用了"计划器-执行器-重规划器"三段式，每完成一个子任务就评估是否需要调整剩余计划。

## 核心实现

```python
# 计划执行状态机
class PlanState(TypedDict):
    task: str
    plan: List[SubTask]
    current_idx: int
    results: Dict[str, str]
    completed: List[str]

def replan_node(state):
    """根据已完成结果，重新规划剩余任务"""
    remaining = get_remaining_tasks(state)
    new_plan = planner.replan(
        state.task, state.results, remaining
    )
    return {"plan": new_plan, "current_idx": 0}
```

## 收益与边界

- 动态重规划：执行中遇到阻碍自动调整计划
- 子任务结果缓存与传递，避免重复计算
- 计划树可视化，让用户看到任务拆解过程
