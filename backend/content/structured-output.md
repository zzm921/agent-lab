---
id: structured-output
name: 结构化输出与推理增强
shortDesc: JSON Schema 约束输出格式，配合思维链 / 自洽性 / 思维树，让输出可被程序消费、推理更可靠。
icon: code-bracket
difficulty: beg
tags: [Structured-Output, JSON-Schema, CoT, Self-Consistency, Tree-of-Thoughts]
techFilters: [LangGraph]
accent: '#10b981'
experience: false
---
## 概述

结构化输出与推理增强是"输出侧"的两件互补的事：

- **结构化输出**：用 JSON Schema / 函数声明**约束模型输出的格式**，让结果可直接被程序消费与校验；
- **推理增强**：在"模型怎么想"上做文章——CoT 逐步推理、Self-Consistency 多次采样投票、Tree-of-Thoughts 多路径搜索，让复杂问题的答案更稳。

一句话：结构化输出管**输出长什么样**，推理增强管**答案准不准**。

## 为什么需要它

- **自由文本不可被程序消费**：下游业务要取字段、做强类型校验，纯文本只能靠脆弱的解析；
- **模型对复杂问题"跳步"**：直接作答看似流畅，实则推理过程缺失、错误率高；
- **单次采样有随机性**：关键决策不能赌运气，需要多次采样取更稳的结果；
- **约束与自由度要平衡**：约束过度压制模型能力，约束不足则产出非法 JSON。

## 结构化输出：让结果可被程序消费

### 三种实现方式

| 方式 | 原理 | 适用 |
|------|------|------|
| JSON Mode / Response Format | 请求声明 `response_format={"type": "json_object"}`，模型按 schema 输出 JSON | 通用结构化结果 |
| Function Calling / 工具约束 | 把输出声明为一次"工具调用"，参数由 schema 校验 | 与工具调用天然衔接 |
| Schema 校验 + 失败重试 | 输出用 Pydantic 校验，失败自动重试修正 | 生产环境兜底 |

### 核心模式：约束结构、不约束内容

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)
    need_human: bool = False

# JSON Mode + 运行时校验 + 失败重试
raw = llm.chat(task, response_format={"type": "json_object"})
for _ in range(3):
    try:
        return Answer.model_validate(json.loads(raw))
    except ValidationError:
        raw = llm.chat(task, hint="修正为合法 JSON")
```

原则：**schema 只约束字段与类型，不约束内容的写法**——既拿到程序可用的结构，又不压制模型的表达能力。

## 推理增强：让答案更可靠

| 方法 | 原理 | 成本 | 适用 |
|------|------|------|------|
| CoT（思维链） | 先写推理步骤，再给结论 | 低 | 数学 / 逻辑 / 多步推理 |
| Self-Consistency（自洽性） | 多次采样 + 多数投票 | 中（N 倍采样） | 答案有明确正确项的任务 |
| Tree-of-Thoughts（思维树） | 多路径搜索 + 评估回溯 | 高（树状展开） | 探索型 / 组合型难题 |

```python
# Self-Consistency：多次采样，多数投票
samples = [llm.complete(task, cot=True) for _ in range(5)]
return Counter(samples).most_common(1)[0][0]

# Tree-of-Thoughts：多条路径并行探索，择优回溯
for step in range(max_depth):
    candidates = [llm.expand(node) for node in frontier]
    frontier = [c for c in candidates if llm.evaluate(c) > threshold]
```

三者的关系是**递进**的：CoT 是"想一遍"，Self-Consistency 是"想多遍再投票"，ToT 是"分叉多条路再挑"。

## 本项目的做法

项目在 LLM 层（`app/llm/dashscope_chat.py`）实现了对结构化输出的原生支持：

```python
# Pydantic 模型类 → JSON Schema → function tool（with_structured_output 兼容）
def _tool_to_ds(tool: BaseTool | dict | type) -> dict:
    if isinstance(tool, dict):
        return tool
    if isinstance(tool, type) and issubclass(tool, BaseModel):
        schema = tool.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": schema.get("title") or tool.__name__,
                "description": schema.get("description") or "",
                "parameters": schema,
            },
        }
    ...

# bind_tools 返回新副本，结构化输出工具不会污染共享 LLM 实例
def bind_tools(self, tools, **kwargs):
    bound = self.model_copy(deep=True)
    bound._tools = [_tool_to_ds(t) for t in tools]
    ...
```

关键设计：

- **模型类即 Schema**：`with_structured_output` 以 Pydantic 模型类调用 `bind_tools`，`model_json_schema()` 自动生成 JSON Schema，无需手写约束；
- **副本绑定防污染**：`bind_tools` 返回 `model_copy(deep=True)` 的新副本——否则如 reflection 评审器把 `CritiqueResult` 绑到生成器上，会让生成器误以为存在该工具；
- **推理增强**：模型开启思考（`enable_thinking`）时，`reasoning_content` 与 `content` 分离，推理过程以 thinking 事件流式下发；CoT 策略在 runner 的 `STRATEGY_PROMPTS` 中一键开启。

## 收益与边界

**收益**

- 输出可被程序直接消费，进入业务链路前先做强类型校验；
- 校验失败自动重试，非法 JSON 有兜底，不会带进下游；
- 推理增强让复杂问题答案更稳，推理过程可观测、可检查。

**边界 / 局限**

- 结构化约束占用一部分模型自由度，字段极多时模型易出错，仍需人工兜底；
- Self-Consistency / ToT 多次采样有成本与延迟，简单任务不值得；
- 推理增强降低的是随机错误，**消除不了幻觉**——它只是让模型"更稳地错"的概率变小。

## 演进与关联

```
提示词策略（怎么问）
   │
   ├─→ CoT（最基础的推理增强）
   │      ├─→ Self-Consistency（多次采样投票）
   │      └─→ Tree-of-Thoughts（多路径搜索）
   │
   └─→ 结构化输出（怎么让输出可用）
          └─→ Function Calling → MCP → A2A（输出即工具调用的协议演进）
```

- **与提示词策略**：CoT 既是推理增强方法，也是提示词策略之一（见提示词工程标签）；
- **与协议线**：结构化输出的"函数声明"，正是 Function Calling → MCP → A2A 这条协议演进线的落点；
- **与 Harness**：解析失败重试、失败率统计等由护栏层统一管理，结构化输出只需聚焦"约束格式"本身。
