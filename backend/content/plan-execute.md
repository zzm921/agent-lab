---
id: plan-execute
featured: true
name: Plan-and-Execute
shortDesc: 先拆解任务为子步骤，再逐个执行，复杂任务的成功率大幅提升。
icon: list
difficulty: int
completeLevel: 90
tags: [Planning, Decomposition, Task-Graph, Agent]
techFilters: [LangGraph]
accent: '#38bdf8'
mode: plan_execute
enabledTools: [rag]
prompts:
  - 帮我策划一次周末短途旅行：先拆解预算、交通、住宿、行程等子任务，再逐个给出建议。
  - 拆解「从零学习 Python 数据分析」的学习路径，分步骤给出每周计划。
  - 帮我规划明天的工作：先列出任务清单，再按重要与紧急排序，给出执行顺序。
  - 帮我把「申请出差报销」的完整流程整理出来，并列出每一步需要的材料
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
    text = llm(PLAN_PROMPT + task)            # 一次 ainvoke 返回完整规划文本（非流式）
    if 正文第一行 == "DIRECT":                  # 无需计划（简单问候 / 一句话问答）：直接回复
        return { todo: [], past_steps: [], replans: 0, step_failed: False }   # 不发 plan 事件
    todo = parse_todo(正文)                    # 每行一个子任务，行首 [行号] 标注依赖 → [{id, desc, deps, status}]
    emit({ type: "plan", items: todo, current_step: 0, status: "created" })
    return { todo, past_steps: [], replans: 0, step_failed: False }

async def executor(state):
    steps += 1                            # 累计模型调用/工具回合数
    if steps > max_steps:                 # 轮数上限：防单步内反复请求工具死循环
        return { steps, stopped: "max_steps" }
    if not todo:                          # DIRECT：无子任务，直接流式回复后结束
        msg = stream_model_call(llm, messages, emit, tools, system_prompt=base)
        return { messages: [msg], steps }
    # step_hint 标注「当前子任务 tN/M」+ 是否最终步骤：非最终步骤只输出简短结论，最终步骤输出完整方案
    msg = stream_model_call(llm, messages, emit, tools,
                            system_prompt=base + step_hint(state))
    if msg 含工具调用:
        return { messages: [msg], steps } # 路由到 tools，执行后回到本节点
    # 本项完成：置 done（失败则置 failed），记录进度，发射 plan running/done
    todo[当前项].status = "failed" if failed else "done"
    emit({ type: "plan", items: todo, current_step: 下一个未完成项, status: "done" if 全部完成 else "running" })
    return { messages: [msg], todo, past_steps += 第 k 项完成/失败, steps }

async def replanner(state):
    # 只重写「未完成 / 失败」的剩余项，保留已完成项（不整盘推翻）
    context = f"原任务：{task}\n已完成：{progress or '（无）'}\n剩余待办：{剩余项描述}"
    new_items = parse_todo(llm(REPLAN_PROMPT + context), start=len(已完成项))
    new_todo = 已完成项 + new_items       # 已完成项 id 保持不变，新项从 t{N+1} 续编
    emit({ type: "plan", items: new_todo, current_step: 0, status: "created" })
    return { todo: new_todo, replans+1 }

def should_replan(state):
    if stopped == "max_steps": return END     # 达轮数上限直接结束
    if 末条消息含工具调用: return "tools"
    if 存在 failed 项 且 replans < max_replans: return "replan"   # 失败 → 重规划
    if 无 pending 项: return END              # 全部完成（或全部失败且不再重规划）
    return "continue"                          # 否则继续下一个子任务
```

### 事件流

```
（规划中）前端「正在拆解任务，生成执行计划…」占位卡片
（无需计划）→ 无 plan 事件，executor 直接流式回复
（需要计划）plan(created, items 全 pending) → [tool_start/tool_end（工具回合）] → plan(running，逐项 done)
  →（某项 failed）plan(created，重规划：保留已完成项) → … → plan(done，全 done)
```

### 输出约束（防重复输出）

executor 每步都会流式输出结果，模型若不约束会在中间步骤就输出完整最终方案，导致同一轮内重复。因此 `step_hint` 明确标注当前步是否为最终步骤：

- **非最终步骤**：只输出本步简短结论（关键数据/决定，2-4 行），不输出完整最终方案、不重复已输出内容；
- **最终步骤**：整合此前所有步骤结论，输出完整的最终方案。

### 防死循环

| 机制 | 作用 |
|------|------|
| **max_steps** | executor 累计模型调用/工具回合数上限，防「单步内反复请求工具」死循环 |
| **max_replans** | 重规划次数上限（取 `max_iterations // 2`），防「失败 → 重规划 → 再失败」空转 |
| **步骤完成判定** | 无工具调用即认为本步完成并推进，不依赖模型自报完成 |
| **澄清轮次豁免** | `ask_user` 提问/回答不计入 max_steps（末条为澄清回复 ToolMessage 时跳过计数），避免因向用户确认信息消耗执行预算导致任务提前终止 |

### 缺信息时的澄清引导

executor 执行前先判断所需信息是否已具备，按以下框架处理：

- **事实类信息**（可通过外部工具核实的内容）→ 主动调用相应工具核实后再输出结论，不依赖模型记忆臆测；
- **私有信息**（仅用户掌握、工具无法获取的内容）→ 通过一次 `ask_user` 调用（`questions` 一次性列出全部缺失项，不要逐项往返提问）礼貌澄清；
- **澄清回复为「跳过未回答」或信息仍不完整** → 直接基于已有信息给出建议并如实标注假设与所缺信息，不要重复追问同一问题；
- **全程不得编造或臆测未核实的事实**。

### 与通用设计的对应关系

| 通用设计 | 本项目做法 |
|---------|-----------|
| 计划器 | planner 节点，`_PLAN_PROMPT` 先判 DIRECT 再拆解（2-5 项、适度粒度）+ `_parse_todo` 解析 |
| 执行器 | executor 节点，单步流式模型调用 + 工具循环，按最终/非最终步骤约束输出 |
| 重规划器 | replanner 节点，`_REPLAN_PROMPT` 基于已完成步骤重生成 |
| 终止条件 | `should_replan`：完成 / max_steps / max_replans 三路 |
| 结果传递 | `past_steps` 记录已完成步骤带入上下文 |
| 工具执行 | 复用共享 `make_tools_node`（事件 + HITL + 异常兜底） |

## 收益与边界

- 计划可审查、可干预：`plan` 事件下发完整子任务与当前进度，用户能看到任务拆解过程
- 动态重规划：步骤失败时基于已完成进度与失败原因重建剩余计划，而非从头再来
- 无需计划直接回答：简单问候/一句话问答走 DIRECT，不产生计划开销
- 规划中占位：规划期间前端显示「正在拆解任务…」占位，规划器一次返回后整体下发完整 todo 清单（协议确定、可回放、可评测）
- 结果传递：`past_steps` 记录已完成步骤，避免重复计算
- 复用共享 `make_tools_node`：工具事件 + HITL 审批 + 异常兜底，失败写 `step_failed`
- 边界：计划质量依赖规划器 prompt；拆解过细会增加模型调用成本，过粗则失去计划意义（prompt 已约束适度粒度）

## 测试覆盖

`backend/tests/test_modes.py` 覆盖：正常计划执行、DIRECT 无需计划直接回复、步骤失败触发重规划、max_steps 轮数上限（单步内反复请求工具被拦截）、todo 状态流转与依赖解析。

