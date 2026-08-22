---
id: prompt-strategy
name: 提示词策略
shortDesc: Zero / One / Few-shot 与思维链一键切换对比，直观感受提示策略对输出的影响。
icon: sparkles
difficulty: beg
completeLevel: 100
tags: [Prompt-Engineering, Few-Shot, CoT]
techFilters: [FastAPI]
accent: '#f59e0b'
---
## 为什么需要它

提示词工程是 Agent 能力的基础——不改模型，只改输入。零样本（Zero-shot）直接提问，少样本（Few-shot）给出示例让模型照着模式回答，思维链（Chain-of-Thought）引导逐步推理再作答。不同的策略对输出质量影响显著，本平台支持一键切换直观对比。

## 怎么解决

核心工作在于为每种策略设计高质量的 prompt 模板，并确保切换时不影响其他模块。我用模板继承的方式组织——基础模板定义结构和角色，各策略模板通过填充不同的示例与推理指令来扩展，新增策略零侵入。

## 核心实现

```python
# 提示词策略模板系统
class PromptStrategy(Enum):
    STANDARD = "standard"
    FEW_SHOT = "few_shot"
    COT = "cot"

def build_prompt(strategy: PromptStrategy, task: str):
    base = load_template("base.txt")

    if strategy == PromptStrategy.STANDARD:
        examples = ""
        thinking_instruction = "直接回答问题。"
    elif strategy == PromptStrategy.FEW_SHOT:
        examples = load_template("few_shot_examples.txt")
        thinking_instruction = "参考示例回答。"
    elif strategy == PromptStrategy.COT:
        thinking_instruction = (
            "先逐步推理，再给出答案。"
            "用'思考：'标记推理过程。"
        )

    return base.format(
        task=task,
        examples=examples,
        thinking_instruction=thinking_instruction,
    )
```

## 收益与边界

- Zero / One / Few-shot + CoT 四种策略一键切换对比
- 模板继承式设计，新增策略零侵入
- 内置高质量示例，开箱即用
