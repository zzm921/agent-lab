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
## 为什么需要它

结构化输出用 JSON Schema / 函数声明约束模型输出格式，让结果可直接被程序消费与校验；推理增强则在"怎么想"上做文章——CoT 逐步推理、Self-Consistency 多次采样投票、Tree-of-Thoughts 多路径搜索，让复杂问题的答案更稳。

## 怎么解决

难点在于约束与自由度的平衡——过度约束压制模型能力，约束不足则产出非法 JSON；Self-Consistency 的多次采样有成本，ToT 的路径搜索深度需要控制。业界做法：JSON Mode + Pydantic 运行时校验，解析失败自动重试回退。

## 核心实现

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

# 推理增强：Self-Consistency 多次采样投票
samples = [llm.complete(task, cot=True) for _ in range(5)]
return Counter(samples).most_common(1)[0][0]
```

## 收益与边界

- JSON Schema + 运行时校验，输出可直接进业务链路
- Self-Consistency 多次采样投票，答案更稳定
- Tree-of-Thoughts 多路径搜索，突破单路径上限
