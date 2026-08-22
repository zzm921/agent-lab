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
prompt: 帮我计算 2 的 10 次方，再告诉我今天的日期。
---
ReAct（Reasoning + Acting）是 Agent 的底层引擎——"思考"与"行动"交替进行：模型输出推理 → 决定调用哪个工具 → 观察工具返回结果 → 再思考再行动，直至得出结论。它第一次让 LLM 学会「先想清楚再动手，动手后看结果」，是所有后续 Agent 框架（计划执行、反思修订、多智能体）的基座。

## 为什么需要它

在 ReAct 出现之前，LLM 的能力边界固定在「单次生成」上：你问一句，它答一句，答完即止。遇到需要外部信息的任务（查天气、查数据库、跑代码、访问网页），模型只能靠训练时的记忆作答——而记忆会过期、会编造。

ReAct 要解决的正是这个根本矛盾：**模型的智力（推理）与模型的局限（无实时信息、无法行动）之间的断层**。它把「推理」和「行动」接进同一个循环，让模型不仅能想，还能动手，动手后还能根据结果修正自己的想法。

- 痛点 1：模型知识有截止时间，训练后无法获知新信息；
- 痛点 2：模型只能输出文字，无法真正「做事」（执行代码 / 调接口 / 操作文件）；
- 痛点 3：即使强行让模型调工具，一次调用失败就整段崩溃，缺乏「观察 → 修正」机制。

## 怎么解决

ReAct 的核心是把 Agent 组织成一个 **观察 → 思考 → 行动 → 观察** 的闭环状态机：

| 环节 | 模型在做什么 | 是否调用工具 |
|------|------------|------------|
| 思考 Thought | 分析当前观察结果，决定下一步 | 否 |
| 行动 Action | 发起一个工具调用（如 `search("...")`） | 是 |
| 观察 Observation | 读取工具返回结果，写回上下文 | 否 |

工程实现需要解决四个问题：

1. **循环编排**：用状态机显式建模三态跳转，而不是靠 Prompt 暗示。我用 LangGraph 的 `StateGraph` 定义节点（think / act / observe）与条件边，天然支持循环、分支与中断；
2. **结构化行动**：模型以结构化 JSON（Function Calling）声明要调用的工具与参数，避免靠文本解析的不稳定；
3. **终止条件**：设置最大迭代次数与最大工具调用次数，防止死循环与无限烧 token；
4. **可中断性**：每一步思考都落盘，支持中途打断、人工审批后再继续。

## 核心实现

```python
# LangGraph 状态机：ReAct 三态闭环
def build_react_graph():
    graph = StateGraph(AgentState)

    graph.add_node("think", think_node)      # 思考：产出 Thought + Action(JSON)
    graph.add_node("act", action_node)       # 行动：执行工具调用
    graph.add_node("observe", observe_node)  # 观察：把结果写回状态

    graph.set_entry_point("think")
    graph.add_edge("think", "act")
    graph.add_conditional_edges(
        "act",
        should_continue,      # 判断：还要继续 → observe；否则 → END
        {"observe": "observe", "end": END},
    )
    graph.add_edge("observe", "think")

    return graph.compile()
```

关键细节：

- `should_continue` 同时检查「是否已得到最终答案」与「是否超过迭代上限」，二者任一满足即终止；
- 行动节点在执行前会经过 Harness 护栏层（审批 / 熔断 / 工具白名单），保证「想干」不等于「能随便干」。

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

## 参考链接

- [ReAct: Synergizing Reasoning and Acting in Language Models（原论文）](https://arxiv.org/abs/2210.03629)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)


