---
id: function-calling
name: 函数调用
shortDesc: 模型以结构化 JSON 直接发起工具调用；各家私有格式，是 MCP 的前身。
icon: cpu
difficulty: beg
tags: [Function-Calling, Tool-Calling, JSON, Protocol]
techFilters: [MCP]
accent: '#6366f1'
enabledTools: [rag, calculator]
prompts:
  - 出差 6 天，连续出差按每日 100 元餐补，餐补一共多少钱？
---
## 概述

Function Calling 让模型在生成文本的同时，以**结构化 JSON** 声明要调用的工具与参数，程序据此执行并把结果回填给模型。它是「模型 → 工具」交互的第一个标准形态，解决了模型训练后知识停滞的实时性问题，也是 Agent 执行能力（ReAct 循环里的「行动」环节）的底层触发机制。

## 为什么需要

- 模型知识有截止时间，无法获知实时信息（天气、行情、数据库）；
- 模型只能输出文字，无法真正「做事」（执行代码 / 调接口 / 操作文件）；
- 早期靠文本格式（Thought/Action）解析工具调用不稳定，需要原生、结构化的协议。

痛点：每家供应商格式私有——M 个模型 × N 个工具需 M×N 次对接，且函数定义格式各异（OpenAI / Claude / Gemini）。

## 通用设计思路

### 完整调用流程

```
1. 开发者定义工具 Schema（name / description / parameters）
2. 用户提问："北京今天天气怎么样？"
3. 模型决策需要调工具 → 输出 tool_calls:
   [{"id": "call_1", "type": "function",
     "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}]
4. 应用系统执行工具，拿到结果："北京 26℃ 晴"
5. 把结果作为 role:"tool" 的消息返回给模型
6. 模型基于工具结果生成最终回答："北京今天 26℃，晴天。"
```

### 关键工程要点

- **工具描述质量**：决定约 80% 的调用准确率——描述要含功能、参数含义、适用场景、返回格式、示例；
- **参数 Schema 设计**：字段越少越可靠；用枚举替代自由文本；嵌套不超过 2 层；必填字段明确；
- **并行调用**：现代模型支持一次响应请求多个工具调用，提升效率；
- **tool_choice**：控制模型行为（auto 自主决定 / none 不调用 / 指定函数强制调用）；
- **结构化输出**：以 Pydantic / JSON Schema 严格约束返回，避免格式漂移。

### 演进

```
1.0 早期：文本格式解析（Thought/Action）
2.0 2023.06：OpenAI 原生 Function Calling，结构化 tool_calls
3.0 2024：各家跟进（Anthropic Tool Use、Gemini Function Calling），支持并行调用
4.0 2024–2025：Structured Outputs / JSON Schema 严格约束，MCP 标准化工具接入
```

## 本项目的做法

本项目以 DashScope（阿里云）原生 SDK 为模型层，做「各家私有格式 ↔ LangChain 统一格式」的适配：

- **工具注册**：`_tool_to_ds` 把 BaseTool / dict / Pydantic 模型类统一转为 DashScope tools 条目（OpenAI 风格 function schema）；
- **绑定副本**：`bind_tools` 返回绑定后的新副本（不修改原实例），避免工具 / 结构化输出污染共享 llm；
- **结果归一**：`_to_lc_tool_calls` 把 DashScope 完整 tool_calls 转为 LangChain `AIMessage.tool_calls`（arguments 字符串解析为 dict）。

伪代码：

```python
# 1) 工具注册：BaseTool / Pydantic 模型类 → 供应商 function schema
def _tool_to_ds(tool):
    if 是 Pydantic 模型类:                       # with_structured_output 场景
        return { "type": "function",
                 "function": { "name": 类名, "description": 描述,
                               "parameters": 模型 JSON Schema } }
    return { "type": "function",
             "function": { "name": tool.name, "description": tool.description,
                           "parameters": tool.args or {} } }

# 2) bind_tools：在副本上保存工具，返回新实例（不污染共享 llm）
def bind_tools(self, tools, **kwargs):
    bound = self.model_copy(deep=True)
    bound._tools = [_tool_to_ds(t) for t in tools]
    bound._tool_choice = kwargs.get("tool_choice", "auto")   # any → auto 映射
    return bound

# 3) 供应商 tool_calls → LangChain 统一格式（arguments 解析为 dict）
def _to_lc_tool_calls(ds_tool_calls):
    return [{ "name": fn.name,
              "args": json.loads(fn.arguments),     # 失败兜底为空 dict
              "id": tc.id, "type": "tool_call" }
            for tc in ds_tool_calls]
```

### 与通用设计的对应关系

| 通用设计 | 本项目做法 |
|---------|-----------|
| 工具 Schema | `_tool_to_ds` 统一转 OpenAI 风格 function schema |
| tool_choice | `bind_tools(tool_choice=...)`（any → auto 映射） |
| 并行调用 | 逐个解析 `tool_calls` 列表，全部回填为 tool 消息 |
| 结构化输出 | `with_structured_output` 以 Pydantic 模型类绑定 |
| 结果回填 | `_to_lc_tool_calls` 归一为 `AIMessage.tool_calls` |

## 收益与边界

- 结构化 JSON 工具调用，链路可编程、可校验
- 解决实时性问题：天气、行情等动态数据
- 私有格式 → 被 MCP 标准化取代的演进起点
- 边界：各家私有格式仍需适配层；工具描述质量仍是调用准确率的最大变量

