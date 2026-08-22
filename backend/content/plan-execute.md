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
## 概述

计划执行（Plan-and-Execute）与 ReAct 的「边走边看」不同：先让 Agent 把复杂任务拆解为有序的子任务列表，再逐个执行并根据结果动态调整计划。计划可审查、可干预，适合多步骤、依赖关系复杂的任务。

一句话：**先想清楚怎么做，再一步步做，边走边修正路线**。

## 为什么需要

- ReAct「边走边看」路径不确定，复杂任务容易绕路、迷失方向；
- 多步骤任务需要整体视角：先拆解能让 Agent 看清全貌，避免只见树木不见森林；
- 计划本身可审查、可干预：用户能看到任务拆解过程，发现问题可在执行前或执行中介入；
- 步骤失败时可以基于已完成进度**局部重规划**，而非整段重来。

## 通用设计思路

用「**计划器 → 执行器 → 重规划器**」三段式，每完成一个子步骤就评估是否需要调整剩余计划：

1. **计划器**：把任务拆解为 2–N 个有序子步骤。步骤粒度要可控——太粗执行时无从下手，太细则计划本身成本过高；
2. **执行器**：对当前步骤做一次模型调用，可调用工具；产出完整回答则本步完成、推进到下一步；
3. **重规划器**：某步执行失败（如工具报错）时，基于「原任务 + 已完成步骤 + 失败原因」重新生成剩余计划，覆盖旧计划；
4. **终止条件**：全部步骤完成、或重规划次数达上限、或累计模型调用/工具回合数达上限（防止某一步内反复请求工具导致死循环）。

通用要点：

- **结果传递**：已完成步骤的记录要带入后续上下文，避免重复计算；
- **失败恢复**：步骤失败不等于任务失败，优先重规划而非直接终止；
- **成本控制**：每一步都有独立的模型调用，计划粒度与重规划上限要按成本承受力设定。

## 本项目的做法

本项目用 LangGraph `StateGraph` 原生编排四节点循环：

```
planner → executor ⇄ tools →（失败且未超重规划上限）→ replanner → executor
                              →（完成 / 达轮数上限 / 超重规划上限）→ END
```

### 节点（伪代码）

```python
async def planner(state):
    task = 最近一条用户消息
    text = llm(PLAN_PROMPT + task)
    steps = parse_steps(text)             # 去编号/行首符号，得到步骤列表
    emit({ type: "plan", steps, current_step: 0, status: "created" })
    return { plan: steps, current_step: 0, past_steps: [], replans: 0 }

async def executor(state):
    steps += 1                            # 累计模型调用/工具回合数
    if steps > max_steps:                 # 轮数上限：防单步内反复请求工具死循环
        return { steps, stopped: "max_steps" }
    # 把「当前计划第 k/N 步」拼入 system prompt，让模型聚焦本步
    msg = stream_model_call(llm, messages, emit, tools,
                            system_prompt=base + step_hint(state))
    if msg 含工具调用:
        return { messages: [msg], steps } # 路由到 tools，执行后回到本节点
    # 本步完成：推进 current_step，记录已完成步骤，发射 plan running/done
    return { messages: [msg], current_step: idx+1, past_steps += 已完成第 k 步, steps }

async def replanner(state):
    context = f"原任务：{task}\n已完成：{progress or '（无）'}"
    steps = parse_steps(llm(REPLAN_PROMPT + context))
    emit({ type: "plan", steps, current_step: 0, status: "created" })
    return { plan: steps, current_step: 0, replans+1 }

def should_replan(state):
    if stopped == "max_steps": return END     # 达轮数上限直接结束
    if 末条消息含工具调用: return "tools"
    if current_step >= len(plan): return END  # 全部步骤完成
    if step_failed 且 replans < max_replans: return "replan"   # 失败 → 重规划
    return "continue"                          # 否则继续下一步
```

### 事件流

```
plan(created) → [tool_start/tool_end（工具回合）] → plan(running → done)
  →（步骤失败）plan(created，重规划) → … → done
```

### 防死循环

| 机制 | 作用 |
|------|------|
| **max_steps** | executor 累计模型调用/工具回合数上限，防「单步内反复请求工具」死循环 |
| **max_replans** | 重规划次数上限（取 `max_iterations // 2`），防「失败 → 重规划 → 再失败」空转 |
| **步骤完成判定** | 无工具调用即认为本步完成并推进，不依赖模型自报完成 |

### 与通用设计的对应关系

| 通用设计 | 本项目做法 |
|---------|-----------|
| 计划器 | planner 节点，`_PLAN_PROMPT` 拆解 + `_parse_steps` 解析 |
| 执行器 | executor 节点，单步流式模型调用 + 工具循环 |
| 重规划器 | replanner 节点，`_REPLAN_PROMPT` 基于已完成步骤重生成 |
| 终止条件 | `should_replan`：完成 / max_steps / max_replans 三路 |
| 结果传递 | `past_steps` 记录已完成步骤带入上下文 |
| 工具执行 | 复用共享 `make_tools_node`（事件 + HITL + 异常兜底） |

## 收益与边界

- 计划可审查、可干预：`plan` 事件下发完整步骤与当前进度，用户能看到任务拆解过程
- 动态重规划：步骤失败时基于已完成进度与失败原因重建剩余计划，而非从头再来
- 结果传递：`past_steps` 记录已完成步骤，避免重复计算
- 复用共享 `make_tools_node`：工具事件 + HITL 审批 + 异常兜底，失败写 `step_failed`
- 边界：计划质量依赖规划器 prompt；拆解过细会增加模型调用成本，过粗则失去计划意义

## 测试覆盖

`backend/tests/test_modes.py` 覆盖：正常计划执行、步骤失败触发重规划、max_steps 轮数上限（单步内反复请求工具被拦截）。

