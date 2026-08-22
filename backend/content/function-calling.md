---
id: function-calling
name: 函数调用
shortDesc: 模型以结构化 JSON 直接发起工具调用；各家私有格式，是 MCP 的前身。
icon: cpu
difficulty: beg
tags: [Function-Calling, Tool-Calling, JSON, Protocol]
techFilters: [MCP]
accent: '#6366f1'
experience: false
---
## 为什么需要它

Function Calling 让模型在生成文本的同时，以结构化格式声明要调用的工具与参数，程序据此执行并回填结果。它是"模型 → 工具"交互的第一个标准形态，解决了模型训练后知识停滞的实时性问题；但每家供应商格式私有——M 个模型 × N 个工具需 M×N 次对接。

## 怎么解决

难点在跨模型一致性——OpenAI / Claude / Gemini 的函数定义格式各异，多模型需各自适配；函数描述质量直接影响选对工具的准确率。业界走向：以 MCP 统一工具层，Function Calling 退居为底层触发机制。

## 核心实现

```python
# OpenAI Function Calling 格式
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的实时天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string",
                         "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}]
# 模型返回 tool_calls → 程序执行 → 回填 tool 消息
```

## 收益与边界

- 结构化 JSON 工具调用，链路可编程可校验
- 解决实时性问题：天气、行情等动态数据
- 私有格式 → 被 MCP 标准化取代的演进起点
