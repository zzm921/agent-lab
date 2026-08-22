---
id: observability-eval
name: 可观测性与评估
shortDesc: Trace 全链路追踪 + 指标监控 + 离线评测集，让 Agent 从黑盒变为可调试可优化。
icon: chart-bar
difficulty: int
tags: [Observability, Tracing, Evaluation, LangSmith, RAGAS]
techFilters: [FastAPI]
accent: '#8b5cf6'
experience: false
---
## 为什么需要它

Agent 自主执行让"复盘"变得困难。可观测性用 Tracing 记录每一步思考、工具调用与中间结果，配合 Token / 成本监控、工具失败率统计、循环异常检测定位问题；评估（Evals）用离线回归测试集持续打分，任何 Prompt / 模型 / Harness 调整都先过评测再上线。代表工具：LangSmith、LangFuse、Arize Phoenix、RAGAS、Braintrust。

## 怎么解决

难点在链路与质量的量化——LLM 输出无固定标准，需定义评估维度（准确性 / 相关性 / 忠实度）；离线评测集需持续回流真实失败样本。业界做法：把 Prompt 当代码做版本管理，评测集接入 CI/CD 回归。

## 核心实现

```python
# 评估闭环：离线评测集 + Trace 采样 + 失败样本回流
def evaluate(agent, eval_set):
    for case in eval_set:
        trace = agent.run_traced(case.query)   # 记录每一步
        score = RAGAS({
            "faithfulness": faithfulness(case, trace.answer),
            "relevancy": relevancy(case.query, trace.answer),
        })
        if score < threshold:
            collect(case)                      # 失败样本回流训练集
```

## 收益与边界

- 全链路 Trace：思考 / 工具 / 中间结果全程可视化
- 指标监控：Token 成本、失败率、循环异常自动告警
- 离线评测集 + CI 回归，调整不回归
