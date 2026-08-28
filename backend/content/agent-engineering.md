---
id: agent-engineering
name: Agent 工程演进
shortDesc: 从"怎么说"到"谁运行"——Prompt → Context → Harness → Loop → Graph 五层工程瓶颈外移地图。
icon: layers
difficulty: adv
completeLevel: 100
tags: [Agent-Engineering, Prompt-Engineering, Context-Engineering, Harness-Engineering, Loop-Engineering]
techFilters: [LangGraph, LangChain]
accent: '#6366f1'
prompts:
  - 对比一下：同样的任务，ReAct 循环、Plan-and-Execute、Reflection 三种模式分别会怎么执行？分别适合什么场景？
  - 讲清楚"上下文工程"和"提示词工程"的区别，并说明 RAG 检索、记忆、工具选择分别属于 Agent 工程的哪一层。
  - 从 Prompt 到 Graph，Agent 工程的五层演进分别解决了什么问题？结合本项目的 runner、harness、modes 实现说明。
---
## 概述

Agent 工程有一组高频词：**Prompt 工程、Context 工程、Harness 工程、Loop 工程、Graph 工程**。它们常被放在同一层比较，于是越讲越抽象。本卡片把它们整理成一条由内向外、层层嵌套的**工程演进路径**——它们不是"新概念取代旧概念"的换代链，而是同一套 Agent 系统里五个不同的关注层次，每层回答一个不同的问题：

| 层 | 回答的问题 | 一句话记忆 |
|---|---|---|
| Prompt 工程 | 一次模型调用怎么"说清楚" | 管"怎么说" |
| Context 工程 | 每一步该给模型"什么信息" | 管"喂什么" |
| Harness 工程 | 模型在"什么环境"里工作 | = 环境 |
| Loop 工程 | 干完一轮之后"怎么办" | = 反馈 |
| Graph 工程 | 下一步"允许谁运行" | = 流程 |

需要特别区分**两条轴线**：

- **工程层（定义）**：上表的五层，回答"瓶颈在哪、该优化哪一层"，是关注层次的**定义**；
- **Agent 范式（模式）**：react / plan_execute / reflection / multi_agent，回答"怎么跑"，是另一条轴线上的**具体模式**，不是工程层的定义——它们都实现为 Graph 上不同的节点/边拓扑（见"演进与关联"）。

## 为什么需要它

演进本质是**工程瓶颈随模型能力提升不断外移**：

1. 早期模型指令跟随弱，措辞一变结果就变 → 焦点在 **Prompt**（怎么说）；
2. 模型能读懂复杂指令后，瓶颈移到"该喂什么、何时喂、如何压缩" → 焦点在 **Context**（喂什么）；
3. 模型开始连续调用工具，焦点又移到"承载一次 Agent 运行的脚手架" → **Harness**（环境）；
4. 需要围绕目标反复触发、验证、持续执行 → **Loop**（反馈）；
5. 多个 Agent 要分工协作、分支要显式可控 → **Graph**（流程）。

旧问题不会消失，只会**被模型吸收、被框架固化、或变成平台默认能力**。因此判断一个系统"该优化哪一层"，比"写一个更长的提示词"更重要——这五个层次就是定位故障、划分控制对象的工程视角。

## 五层核心思想

### Prompt 工程：怎么说

- 关注单次模型调用里的指令表达：system prompt、few-shot 示例、CoT 逐步思考、角色/边界/成功标准。
- 它是**一次性手工雕琢**，是 Context 工程的子集——信息给全之后，"如何组织成最终 prompt"仍是 Prompt 的事。

### Context 工程：喂什么

- 不只 prompt，而是"模型要做好这件事需要知道什么"——检索（RAG）、记忆、工具列表、状态、格式。
- 核心不是"塞得更多"，而是**管理有限的注意力预算**：什么常驻、什么按需加载（渐进式披露）、什么压缩、什么丢弃。
- Prompt 从此由**静态模板**变为**动态拼装**。

### Harness 工程：环境

- 模型之外的运行环境。业界定义直白：**Agent = Model + Harness**——从架构图里把模型拿掉，剩下的基本都属于 Harness。
- 六大组成：上下文组装、行动入口（工具/MCP/浏览器/Shell/DB）、持久化（检查点/会话/记忆）、执行控制（超时/预算/审批门/轮数上限）、安全边界（沙箱/权限/密钥）、可观测性（trace/日志/评测）。
- 判断法：Agent 缺能力、丢状态、权限过大、无法审计、换环境表现不一致 → **先修 Harness，别怪模型**。

### Loop 工程：反馈

- 关注一次 Agent 运行之外的闭环：谁触发下一轮、完成条件、谁验证、失败重试还是交给人、状态如何跨会话延续。
- ReAct 的 `模型 → 行动 → 观察 → 模型` 是最内层的小循环；Loop 工程在此基础上增加**证据与终止机制**。
- 关键原则：**不要围绕"信心"循环，要围绕"证据"循环**——"Agent 说自己完成了"不是停止条件，"测试通过 / schema 校验 / 审批人批准"才是。

### Graph 工程：流程

- 关注当前节点结束后**哪个组件被允许运行**——节点、边、条件分支、并行、汇合、状态迁移、受控循环。
- 把"谁先谁后"写成**明规则**，而不是藏在模型输出里。
- 关键认知：**Graph 不是 Loop 的下一代**——Graph 解决"任务路径是否显式"，Graph 的节点内部仍可运行 Agent Loop，Harness 里也可运行 Graph。三者互相嵌套，只是三种不同的工程抓手。

## 本项目的做法

本项目五层均有落点，且按"Harness 包 Graph、Graph 节点内跑 Loop"的方式嵌套：

| 层 | 本项目实现 |
|---|---|
| Prompt 工程 | [runner.py](file:///c:/Users/ASUS/Desktop/workspace/my-agent/backend/app/agents/runner.py) 的 `STRATEGY_PROMPTS`（standard / few_shot / cot）、`TOOL_RETRY_HINT` 工具重试规范、`_augment_query` 按 `generation_mode`（direct/citation/comparison）拼装的指令段 |
| Context 工程 | `_augment_query` 把检索命中 + 来源清单 + generation_mode 指令动态注入用户消息；modular RAG 的指代消解、HyDE 假想文档、context_compress 语义去重、跨轮 seed 复用（`_last_hits`）、answerability 判定不足时强制追问 |
| Harness 工程 | [harness.py](file:///c:/Users/ASUS/Desktop/workspace/my-agent/backend/app/agents/harness.py) 的 AgentHarness（审批策略 / 资源上限 / 止损 / 统计 / 工具计数）、[tools_builder.py](file:///c:/Users/ASUS/Desktop/workspace/my-agent/backend/app/agents/tools_builder.py)、middleware 层（`StreamEventsMiddleware` 事件流、`ModelCallLimitMiddleware` 轮数上限、`WorkerEventsMiddleware` 子代理事件） |
| Loop 工程 | 反馈与终止机制：`ModelCallLimitMiddleware` 强制轮数上限、reflection 的评审通过判定（`passed`）、plan_execute 的步骤完成 / 失败重规划判定、HITL 审批门——都是"证据 + 停止条件"的具体落点 |
| Graph 工程 | [runner.py](file:///c:/Users/ASUS/Desktop/workspace/my-agent/backend/app/agents/runner.py) 统一图运行器（StateGraph + checkpointer + HITL `interrupt`/`resume` + 条件边）；四种范式均编译为图，由 runner 统一 `ainvoke` |

**Agent 范式（另一条轴线，modes/）**——本项目四种范式都是 Graph 上的不同拓扑，不是工程层的定义：

| 范式 | 拓扑 | 对应工程层实现 |
|---|---|---|
| react | `create_agent` 内建「模型 ⇄ 工具」循环 | Loop 层的小循环 + Harness 层中间件护栏 |
| plan_execute | StateGraph：planner → executor ⇄ tools → replanner | Graph 层（显式计划流）+ Loop 层（步骤验证 / 重规划） |
| reflection | StateGraph：generator ⇄ tools → critic → 条件循环 | Graph 层（显式评审流）+ Loop 层（评审证据 + 终止） |
| multi_agent | `create_agent` 编排者 + compute/analyze 子代理 | Graph 层（编排拓扑） |

嵌套关系在本项目中清晰可见：`AgentRunner` 用 Graph（StateGraph + checkpointer）承载**四种 Agent 范式**；其中 react 是模型 ⇄ 工具的小循环，plan_execute / reflection 是带条件边的显式流；AgentHarness 与 middleware 包在最外层提供护栏与可观测性。

## 收益与边界

**收益**

- 一套视角统一解释全部 Agent 工程名词，故障能快速定位到具体层；
- 五层解耦、各自演进：改 Prompt 不动图、加中间件不碰模式、换模型不丢 Harness；
- "先判断该优化哪一层"的思维，避免盲目堆提示词或过早引入复杂编排。

**边界 / 局限**

- 五层边界并非物理隔离，实践中互相嵌套，定位故障仍需结合运行现场；
- Graph 不是万能终点，**范式选型看场景**：步骤可预知用 Plan-and-Execute、探索型用 ReAct、质量优先用 Reflection——**场景匹配才是核心**；
- 每加一层（评审、重试、多智能体）都增加成本与延迟，只有失败代价高于验证代价时才值得。

## 演进与关联

这两条轴线是正交的，最容易被混淆，本卡片的核心区分如下：

```
工程层（定义 · 关注瓶颈在哪）          Agent 范式（模式 · 怎么跑）
                                        ┌───────────────────────────────┐
Prompt 工程（怎么说）                    │  react                         │
   ↓                                    │  plan_execute                  │
Context 工程（喂什么）                    │  reflection                    │
   ↓                                    │  multi_agent                   │
Harness 工程（环境）                      └───────────────┬───────────────┘
   ↓                                                   │ 都是 Graph 上的
Loop 工程（反馈：证据 + 终止）                             │ 不同节点/边拓扑
   ↓                                                   ▼
Graph 工程（流程：节点 / 边 / 状态） ←──── 共同基础设施（LangGraph StateGraph）
```

- **与 Agent 模式（范式）**：react / plan_execute / reflection / multi_agent 是**另一条轴线的 Agent 范式**，不是工程层定义——它们都编译为 Graph 上不同的节点/边拓扑（modes/ 下的实现），Graph 是它们的共同基础设施；
- **与协议演进**：行动环节的工具调用对应 function calling → MCP → A2A 的协议演进线；
- **与 RAG**：检索、重排、压缩是 Context 工程的核心手段，Agentic RAG 则把"要不要检索"变成 Agent 的自主决策。
