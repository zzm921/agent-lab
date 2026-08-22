---
id: reflection
name: 反思修订机制
shortDesc: 草稿—批评—修订三阶段迭代输出，自我批判让答案质量跃迁一个档次。
icon: refresh
difficulty: adv
completeLevel: 95
tags: [Self-Critique, Iteration, Quality, Agent]
techFilters: [LangGraph]
accent: '#ef4444'
mode: reflection
---
## 为什么需要它

反思修订（Reflection / Self-Critique）让 Agent 先产出初稿，再以批评者视角审视不足，最后基于批评意见进行修订。经过多轮迭代，输出质量显著高于单次生成，是 Reflexion 等机制的工程化落地。

## 怎么解决

难点在于如何让"批评"真正有建设性而非泛泛而谈，以及如何防止修订循环发散。我设计了专门的批评 prompt 模板，要求按维度评分并给出具体修改建议；同时引入修订轮次上限和质量评分阈值，达标即停止。

## 核心实现

```python
# 反思修订工作流
async def reflection_workflow(state):
    draft = await generate_draft(state.task)

    for i in range(max_iterations):
        critique = await critique_output(
            draft, state.task, criteria
        )

        if critique.overall_score >= quality_threshold:
            break  # 达标即停止

        draft = await revise_output(
            draft, critique.suggestions, state.task
        )

    return {"result": draft, "iterations": i + 1}
```

## 收益与边界

- 结构化批评维度：准确性、完整性、清晰度、逻辑性
- 质量评分阈值控制，达标即终止，避免无效迭代
- 每轮修订前后对比可追溯，质量提升肉眼可见
