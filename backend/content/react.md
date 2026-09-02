---
id: react
name: ReAct 边想边做
shortDesc: 观察 → 思考 → 行动 → 观察，让 Agent 像人一样边想边做、每一步都可追溯。
icon: brain
difficulty: int
completeLevel: 100
tags: [ReAct, Reasoning, LangGraph, Agent]
techFilters: [LangGraph]
accent: '#f59e0b'
mode: react
strategy: cot
enabledTools: [rag, calculator]
prompts:
  - 帮我计算 2 的 10 次方，再告诉我今天的日期。
  - 帮我计算 (137 × 0.85 − 20) ÷ 3，保留两位小数，并说明计算步骤。
  - 今天是几号？顺便帮我算 2026 年 1 月 1 日距今过了多少天。
  - 请说明我司加班费的支付标准
---
## 概述

ReAct（Reasoning + Acting）是 Agent 的底层引擎——「思考」与「行动」交替进行：模型输出推理 → 决定调用哪个工具 → 观察工具返回结果 → 再思考再行动，直至得出结论。它第一次让 LLM 学会「先想清楚再动手，动手后看结果」，是所有后续 Agent 模式（计划执行、反思修订、多智能体）的基座。

## 为什么需要

在 ReAct 出现之前，LLM 的能力边界固定在「单次生成」上：你问一句，它答一句，答完即止。遇到需要外部信息的任务（查天气、查数据库、跑代码、访问网页），模型只能靠训练时的记忆作答——而记忆会过期、会编造。

ReAct 要解决的正是这个根本矛盾：**模型的智力（推理）与模型的局限（无实时信息、无法行动）之间的断层**。

- 痛点 1：模型知识有截止时间，训练后无法获知新信息；
- 痛点 2：模型只能输出文字，无法真正「做事」（执行代码 / 调接口 / 操作文件）；
- 痛点 3：即使强行让模型调工具，一次调用失败就整段崩溃，缺乏「观察 → 修正」机制。

## 核心思想：观察 → 思考 → 行动

把 Agent 组织成一个**闭环状态机**：

| 环节 | 模型在做什么 | 是否调用工具 |
|------|------------|------------|
| 思考 Thought | 分析当前观察结果，决定下一步 | 否 |
| 行动 Action | 发起一个工具调用（如 `search("...")`） | 是 |
| 观察 Observation | 读取工具返回结果，写回上下文 | 否 |

工程实现需要解决四个通用问题：

1. **循环编排**：用状态机显式建模三态跳转，而不是靠 Prompt 暗示。可用图框架（如 LangGraph `StateGraph` 的节点 + 条件边）或自研 while 循环，天然支持循环、分支与中断；
2. **结构化行动**：模型以结构化 JSON（Function Calling）声明要调用的工具与参数，避免靠文本解析的不稳定；
3. **终止条件**：设置最大迭代次数与最大工具调用次数，防止死循环与无限烧 token；
4. **可中断性**：每一步思考都落盘，支持中途打断、人工审批后再继续。

## 两种写法：传统 Thought-Action vs 现代 ReAct

### 传统写法：Thought-Action（Prompt 驱动）

ReAct 论文提出的原始形态：模型按固定文本格式循环输出，框架靠**正则/字符串解析**驱动循环：

```python
prompt = f"""可用工具：search(query)、calculator(expr)。
请严格按格式输出：
Thought: 你的思考
Action: 工具名
Action Input: 参数

{之前的 Thought / Action / Observation 历史}
当前问题：{question}
"""
text = llm(prompt)                     # 模型输出一大段文本

if "Action Input:" in text:
    action = 正则提取 Action 行            # 解析工具名
    args   = 正则提取 Action Input 行      # 解析参数
    obs    = execute_tool(action, args)
    history += text + f"\nObservation: {obs}\n"
    # 回到循环顶部继续生成，直到模型不再输出 Action
else:
    return 提取最终答案（无 Action 即回答）
```

特点与问题：

- 依赖 Prompt 约束格式，模型偶尔**格式漂移**（漏写 / 写错标签），需解析层正则兜底；
- 工具参数是**非结构化文本**，复杂参数（嵌套 JSON、列表）难以可靠传递；
- 思考与动作混在同一个文本流里，无法精确拆分成独立的流式事件。

### 现代写法：Function Calling + 框架循环

模型以**结构化 JSON** 声明要调用的工具与参数（原生 Function Calling），框架识别到 `tool_calls` 就执行、把结果作为新消息放回上下文，循环交给框架内建，无需手写解析：

```python
def build_react_agent(llm, tools, emit, settings, checkpointer, harness):
    return create_agent(
        model=llm,
        tools=list(tools),
        middleware=[
            # 事件流 + HITL：逐 token 下发 thinking/message，工具执行发 tool_start/tool_end，
            # 需审批的工具调用经 interrupt 暂停等人确认
            StreamEventsMiddleware(emit, harness=harness),
            # 轮数上限：单轮模型调用（思考/工具回合）超 max_steps 即抛错，运行器转为 done
            ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="error"),
        ],
        checkpointer=checkpointer,  # 落盘状态，支持中断/恢复
    )
```

### 两种写法对比

| 维度 | 传统 Thought-Action | 现代 ReAct |
|------|--------------------|-----------|
| 动作声明 | 文本行（Action / Action Input） | 结构化 JSON（Function Calling `tool_calls`） |
| 循环驱动 | 框架手动解析文本 + 拼接历史 | 框架内建（识别 tool_calls → 执行 → 回填上下文） |
| 参数传递 | 非结构化文本，易错 | 结构化类型，精确可靠 |
| 思考可见性 | 思考与动作混在文本流 | `reasoning_content` 独立字段 → thinking 事件 |
| 容错 | 依赖正则解析，格式漂移需兜底 | 原生协议，几乎零漂移 |
| 可观测性 | 难拆分成独立事件 | 逐 token 事件（thinking / message / tool_*） |
| 护栏 | 需要手写校验 | 中间件统一注入（HITL / 熔断 / 轮数上限） |

## 本项目的做法

本项目用 LangChain `create_agent` 内建「模型 ⇄ 工具」循环，用两个中间件补齐事件流与护栏（即上文"现代写法"的完整实现）：

事件循环（隐式，由 create_agent 驱动）：

```
模型 → thinking 增量 /（请求工具则）tool_start → HITL 审批 → tool_end
    → 观察结果 → 模型再思考 …… 直至产出无工具调用的最终答案（message）→ done
```

关键细节：

- `StreamEventsMiddleware`：模型输出按 `reasoning_content → thinking` 事件、`content → message` 事件逐 token 下发；工具调用前经护栏层（审批 / 熔断 / 次数上限 / 重试上限）把关，「想干」不等于「能随便干」；
- `ModelCallLimitMiddleware`：达到轮数上限（`max_steps`）即强制终止并转为 done 而非 error，防止无限烧 token。

与通用要求的对应关系：

| 通用要求 | 本项目做法 |
|---------|-----------|
| 循环编排 | `create_agent` 内建「模型 ⇄ 工具」循环 |
| 结构化行动 | 原生 Function Calling（`tool_calls` JSON） |
| 终止条件 | `ModelCallLimitMiddleware` 轮数上限 `max_steps` |
| 可中断性 | checkpointer 落盘 + `interrupt` HITL 审批 |
| 事件可观测 | `StreamEventsMiddleware` 逐 token 下发 thinking / message / tool_* |

## 收益与边界

**收益**

- 每一步思考显式可观测，Agent 的黑盒变成可追溯的白盒；
- 支持中途中断与人工干预，天然可审批、可审计；
- 循环次数与工具调用次数可控，防止失控。

**边界 / 局限**

- ReAct 是「边走边看」，路径不确定，复杂任务容易绕路或陷入局部循环；
- 单 Agent 单线程，处理多步骤、依赖复杂的任务不如 Plan-and-Execute 直观；
- 依赖模型自身的推理质量，小模型在长链条上容易「迷失方向」。

## 演进与关联

ReAct 是 Agent 演进的**第一个完整闭环**（2022，Princeton + Google Research）：

```
ReAct（2022）—— 底层引擎
  ├─→ Plan-and-Execute（2023）      先规划后执行，弥补「绕路」
  ├─→ Reflexion / Reflection（2023） 草稿-批评-修订，弥补「答不好」
  └─→ Multi-Agent（2024）           角色分工协作，弥补「单线程」
```

- **向上**：更高级的 Agent 模式都建立在 ReAct 的「观察-思考-行动」循环之上；
- **向外**：这个循环要跑得稳，依赖 Harness 层（记忆 / 沙箱 / 审批 / 容错）的支撑；
- **向协议**：行动环节的工具调用，正是 Function Calling → MCP → A2A 这条协议演进线的落点。

