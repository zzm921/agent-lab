---
id: multi-agent
featured: true
name: 多智能体编排
shortDesc: Orchestrator 统一拆解调度子任务，通用 subagent 按角色标注协作完成复杂任务。
icon: network
difficulty: adv
completeLevel: 85
tags: [Orchestrator, Multi-Agent, Coordination, Agent]
techFilters: [LangGraph, MCP]
accent: '#7c5cff'
mode: multi_agent
enabledTools: [rag]
prompts:
  - 你是项目经理：把「上线一个 AI 助手网站」拆给研究员、开发者、测试员三个角色，分派任务并汇总执行方案。
  - 让「策划师 + 文案 + 设计师」三个角色协作，为新品咖啡出一份上市营销方案。
  - 新员工入职需要准备哪些材料？请分角色给出清单
---
## 概述

多智能体编排（Multi-Agent Orchestration）采用「编排者 + 工人」模式：一个**编排者（Orchestrator）**接收任务，拆解后以任务单分派给**通用 subagent（Worker）**执行，每个子任务带角色标注引导执行侧重，结果交回编排者汇总——必要时再分派一轮，直至产出最终答案。

一句话：**一个全能但容易过载的「超人」，不如一个「指挥官」统一拆解调度、一个「通用执行者」分工跑任务**。

## 为什么需要

- **单 Agent 能力过载**：让一个模型同时承担检索、计算、写作、质检，上下文互相污染，长任务容易顾此失彼；
- **任务级专精引导**：每个子任务带 role 标注（计算 / 分析 / 自定义角色），引导 subagent 执行侧重，输出更稳定；
- **并行分派降延迟**：无依赖的独立子任务在同一次分派中并行执行，总耗时从「累加」变为「取最长」；
- **结果可追踪**：每个任务产出什么、由哪个角色执行，按 task id 维度可见可回溯，比单 Agent 黑盒更可审计。

## 通用设计思路

核心组件：

| 组件 | 职责 |
|------|------|
| **编排者 Orchestrator** | 接收任务 → 分析拆解 → 规划分派 → 汇总 → 判断是否再分派 |
| **通用 subagent（Worker）** | 单一通用执行者，按任务单执行子任务；`role` 字段仅作角色标注，不实例化多个 Agent |
| **任务单 TaskTicket** | 分派协议：`{id, role, task, context?, deps?}`，编排者与 Worker 之间的标准化接口 |
| **调度器** | 按任务单依赖深度分层：同层并行（受并发护栏限制）、层间串行 |

> 任务单、调度、状态追踪、结果归位与汇总收敛的完整机制见 [task-system.md](task-system.md)——多智能体是任务系统的一种「编排者 + 子任务分派」形态。

### 编排循环（analyze → dispatch → synthesize → decide）

```
任务 → 编排者 analyze（拆解 + role 标注） → 并行/串行 dispatch → subagent 执行
     → 编排者 synthesize（汇总结果） → decide（信息足够？→ 再分派 or 收尾）
```

- **analyze**：编排者把任务拆成子任务，并为每个子任务标注角色（compute / analyze / 自定义）；
- **dispatch**：按依赖关系一次并行（或串行）派发任务单；
- **synthesize**：收集各任务结果，整合为阶段性答案；
- **decide**：答案是否完整——不完整则再分派一轮补齐缺口，完整则输出最终答案。

## 关键设计点

### 1. 角色标注（role 字段）

执行者统一为**单一通用 subagent**，不按角色实例化多个 Agent。任务单的 `role` 字段仅作**执行侧重标注**：提示 subagent 该子任务偏向数值计算（compute）、逻辑分析（analyze）还是自定义角色（研究员 / 开发者 / 测试员等）。角色不决定「谁执行」，只影响「怎么执行」。

示例角色标注：

| role 标注 | 执行侧重 | 说明 |
|------|------|------|
| compute | 数值计算 | 提示 subagent 优先调用计算工具 |
| analyze | 逻辑分析 | 直接推理分析给出结论 |
| 自定义（研究员 / 开发者 / 测试员等） | 按名字语义侧重 | 仅提示性，不改变执行者 |

### 2. 结构化任务单（TaskTicket）

分派不传裸字符串，而是结构化任务单 `{id, role, task, context?, deps?}`：

- **id**：任务唯一标识，事件流据此串联「谁干了什么」；
- **role**：角色标注（compute / analyze / 自定义），引导 subagent 执行侧重，不参与路由；
- **task**：可独立完成的子任务描述；
- **context**：可选参考上下文；
- **deps**：依赖的前置任务 id 列表——无依赖可并行，有依赖须串行。

（todo 拆解、任务单字段详解、调度波次、状态追踪与汇总收敛见 [task-system.md](task-system.md)）

### 3. 并行 / 串行调度

- **并行**：多个无依赖的子任务在同一次分派中一起发出，同一 subagent 内并发执行，结果按 task id 归位；
- **串行**：有依赖的子任务按依赖深度分层，层间串行执行并注入前置结果；前置任务尚未完成时返回结构化错误，由编排者修正后重新分派。

### 4. 上下文隔离

每个子任务在 subagent 内拥有**独立的一次执行上下文**：只看到自己的任务单与注入的前置结果，互不干扰——避免「研究员找到的资料」污染「测试员的判断」。

### 5. HITL 收敛到编排者

Worker 是子代理、不持有可恢复的会话（无 checkpointer），因此**提问与审批统一收敛到编排者层**：

- Worker 不应直接向用户提问（澄清由编排者统一发起 ask_user）；
- 工具审批在编排者的工具执行层统一弹窗，Worker 内部工具调用不打断用户。

### 6. 双层轮数护栏

- **编排者层**：模型调用/工具回合数超上限 → 强制结束（防编排者反复分派空转）；
- **Worker 层**：每个子任务执行也设轮数上限（防单个任务内部死循环）。

### 7. 编排阶段事件（可观测）

前端按阶段逐步展示：

```
orchestrator analyze → agent_event(dispatch, worker=t1) → worker_running(thinking/message)
  → agent_event(done, worker=t1) → orchestrator synthesize →（decide 再分派）→ … → done
```

`agent_event` 的 `status` 字段：`dispatch / running / done / failed`；`running` 时 `stage` 为 `thinking / message`（worker 执行过程流式增量），Worker 名 + task id 全程可见。

## 推荐 Prompt（示例）

### ① 编排者系统提示（任务单 + role 标注）

```
你是多智能体编排者（Orchestrator）。你的职责是接收用户任务，拆解后分派给
通用 subagent（Worker），最后整合它们的产出给出完整答案。你不是执行者，不亲自
完成子任务，只负责「拆解 → 分派 → 汇总 → 决定是否再分派」。

任务单角色标注（role 字段，仅提示执行侧重，不切换执行者）：
compute=数值计算 / analyze=逻辑分析 / 自定义角色（研究员、开发者、测试员等）。

分派规范：
1. 能拆成多个独立子任务的，一次调用并行分派（tasks 列表一次列全），
   不要逐个一问一答多次往返；
2. 有依赖关系的子任务（前一步产出是后一步输入）用 deps 字段注明前置任务 id，
   Worker 会按依赖顺序自动执行并注入前置结果；
3. 每个子任务必须是可独立完成的最小单元；
4. 信息不足时，把「向用户确认关键信息」作为一步，通过 ask_user 统一澄清；
5. Worker 返回错误时，可调整任务措辞后重新分派（同一任务不重复派发）；
6. 汇总时必须区分「Worker 已产出的结果」与「你的推断」，不要编造 Worker 没给的数据；
7. 最终答案由你整合所有结果后直接输出，不得把总结/汇总类任务派给 Worker。
```

### ② Worker 系统提示（单一通用 subagent）

```
你是通用 Worker（subagent），负责执行编排者派发的单个子任务。
需要数值计算时调用计算工具，否则直接进行逻辑分析并给出结论。
若工具调用失败或返回错误，先修正参数或换一种方式重试，不要直接说工具不可用。
只完成派给你的单个子任务，不做全盘汇总或最终总结，汇总由编排者负责。
```

### ③ 汇总（synthesize）引导

```
请汇总以下 Worker 的产出，形成完整答案：
- 结构清晰：按子任务分节组织，标注每节来源任务；
- 交叉验证：若多个任务结论冲突，说明冲突并给出你的判断依据；
- 如实标注：任务未覆盖的部分明确说明，不要用自身知识编造补齐。
```

### ④ 再分派决策（decide）

```
基于当前已汇总的结果判断：
1. 用户任务是否已完整回答？是 → 输出 {"continue": false}；
2. 若仍缺关键信息，且该信息可通过 subagent 获得（缺口依赖前置产出 / 尚未覆盖）
   → 输出 {"continue": true, "next_tasks": [任务单列表]}；
3. 不得重复分派已由 Worker 完成并产出的任务。
输出必须严格是 JSON，不要输出任何其他文字。
```

## 本项目的做法

本项目把多智能体模式实现为「编排者 `create_agent` + 通用 subagent + Worker 工具化分派」，与 react / plan_execute / reflection 并列（侧边栏可切换）：

- **模式构建**：编排者与 subagent 均用 `create_agent` 构建，subagent 经 `convert_runnable_to_tool` 包装为编排者工具，入参为任务单；
- **单一通用 subagent**：不创建多个角色 Agent，只构建一个通用 worker subagent（带 calculator）；任务单 `role` 字段作为角色标注（compute=数值计算 / analyze=逻辑分析 / 自定义角色），执行者统一；
- **任务单分派**：Worker 工具入参为 `{"tasks": [{"id", "role", "task", "context"}]}`，一次可派多个子任务；任务单内按依赖深度分层并行执行（同层任务并发、层间串行），事件按 task id 归位（任务单与调度的通用机制见 [task-system.md](task-system.md)）；
- **编排阶段事件**：中间件按 worker 工具名匹配，发射 `agent_event`（dispatch / done + worker 名 + task id）；Worker 执行过程透传 thinking / message 中间事件，前端逐步展示；
- **HITL 收敛**：Worker 无 checkpointer 不触发中断；提问（ask_user）与工具审批统一收敛到编排者层；
- **双层护栏**：编排者与 Worker 各自挂轮数上限，防止编排者反复分派与单 Worker 内部死循环；
- **事件流**：

```
orchestrator → agent_event(dispatch, worker=worker, task_id=t1) → worker_running(thinking/message)
  → agent_event(done, worker=worker, task_id=t1, result) → orchestrator(synthesize)
  →（decide：缺口再分派）→ … → done
```

## 收益与边界

**收益**

- 任务级上下文隔离：每个子任务独立执行上下文，互不污染；
- 并行分派降延迟：独立子任务同一 subagent 内并发执行；
- 可插拔替换：任务单协议标准化后，换执行者 = 换一个 subagent；
- 可观测可审计：分派/执行/汇总全程按 task id 可见。

**边界 / 局限**

- 编排成本高：每轮分派与汇总都有额外模型调用，简单任务用多智能体是浪费；
- 任务间信息传递依赖编排者汇总：跨任务复杂依赖会放大编排轮次与上下文长度；
- 分派质量依赖编排者 prompt：拆解过细或 role 标注不当会直接降低整体效果；
- 并行分派只适用于「可独立」的子任务，耦合任务仍需串行。

## 演进方向

- **动态角色创建**：由编排者按任务现场声明新角色，实例化为独立执行体（当前 role 仅为标注，执行者统一）；
- **群聊 / 辩论模式**：Worker 之间直接对话交锋，而不是全部经由编排者中转；
- **图任务编排**：把任务依赖建模为 DAG，由调度器一次编排并行波次（见 [llm-compiler.md](llm-compiler.md) 的并行 DAG 编译思路）。
