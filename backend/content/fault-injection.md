---
id: fault-injection
name: 容错·熔断·故障注入
shortDesc: 两层重试 + 熔断短路 + 13 类故障注入，测试 Agent 在异常情况下的鲁棒性。
icon: shield
difficulty: adv
completeLevel: 80
tags: [Resilience, Testing, Chaos-Engineering, Circuit-Breaker]
techFilters: [LangGraph, MCP]
accent: '#ef4444'
enabledTools: [calculator]
faults:
  calculator: timeout
prompt: 帮我计算 128 × 37，并验证结果的正确性。
---
故障注入是混沌工程思想在 Agent 上的落地：主动向工具调用注入「失败」（超时、5xx、限流、参数错误……），验证 Agent 在异常情况下是崩溃、无限重试还是优雅降级。它把「鲁棒性」从口头承诺变成可度量的工程指标。

## 为什么需要它

Agent 的本质是「自主执行」——模型在循环里不断调用外部工具。一旦某个工具失败，自主循环可能陷入三种糟糕局面：

- **崩溃传播**：一次工具报错让整个任务链断裂，前面的工作全部白费；
- **无限重试**：模型对瞬时错误盲目重试，白白烧掉 token 还放大下游压力；
- **静默失败**：Agent 假装成功继续推进，把错误结果带进最终答案（最危险）。

这些问题在「一切正常」时不会暴露，只有故障时刻才显现。而真实世界里，超时、限流、5xx、参数写错每天都在发生。**不能等到上线后由真实事故来测试鲁棒性**——故障注入让这些场景可以被主动、可重复地演练。

## 怎么解决

核心是两条设计决策。

**决策 1：给失败分门别类，决定「谁来重试」**

不是所有失败都该用同一种方式处理。平台把故障分成两类，分别走不同重试路径：

| 分类 | 典型故障 | 重试走向 | 理由 |
|------|---------|---------|------|
| retryable（瞬时） | timeout / conn_reset / dns / 429 / 5xx | 工具层透明重试 | 与参数无关，换个时间点大概率成功 |
| permanent（参数/业务） | 4xx / 余额不足 / 业务报错 | 交给模型思考后重试 | 同参数重试必败，盲试是浪费 |

**决策 2：熔断短路，防止雪崩**

同一会话内「同一工具 + 同一参数」连续失败达到阈值后，熔断该调用（half-open 冷却后放行一次探测）。模型换参数重试视为新调用始终放行——既保护下游，又不堵死模型修正参数的出路。

**配套：两层重试上限**

- 工具层：瞬时错误自动重试（指数退避），把偶发抖动消化在内部；
- Agent 层：同一工具连续失败达到上限后，明确提示模型「换一个工具」。

## 核心实现

```python
# 故障注入钩子：执行前短路为失败，计入失败计数
spec = harness.fault_spec(tool_name)          # 命中返回 {mode, message, retryable}
if spec is not None:
    if spec["retryable"]:
        raise RetryableToolError(spec["message"])   # 工具层透明重试
    return spec["message"]                            # 返回给模型思考后重试

# 熔断判定：同参数重复失败达阈值 → OPEN 短路
if not harness.circuit_allows(session_id, tool_name, args):
    return error_event("circuit_open, cooling down")
```

故障目录（13 类）：

- retryable：`timeout` `conn_reset` `dns` `http_429` `http_500` `http_502` `http_503`
- permanent：`error` `business` `http_400` `http_401` `http_403` `http_404`

## 收益与边界

**收益**

- 两层重试各取所长：瞬时错误透明消化、参数错误交给模型思考，避免盲试；
- 熔断 + half-open 探测，防止工具雪崩式失败拖垮会话；
- 13 类故障注入可一键开启，容错能力可验证、可演示、可回归。

**边界 / 局限**

- 故障注入只模拟「工具层失败」，覆盖不了模型幻觉、检索质量差等「不报错但答错」的情况；
- 熔断只针对同参数重复失败，模型若持续换着花样失败，仍需要 Agent 层上限兜底；
- 注入本身也是要控制的——生产环境不应开启。

## 演进与关联

容错设计是 Harness（Agent = Model + Harness）六大组件之一，与其它护栏协同：

```
审批门（先拦高风险）→ 沙箱（隔离执行）→ 工具调用 → 故障注入 / 两层重试 / 熔断
                                                       ↘ 失败太多 → 可观测告警 / 审计
```

- **与 HITL 协作**：高危工具强制审批，把「失败面」挡在进入之前；
- **与可观测协作**：每次失败都计入工具失败率统计，可观测层据此告警；
- **理念来源**：Chaos Engineering（Netflix 混沌工程）——「故障不是意外，是常态，要主动演练」。

## 参考链接

- [Netflix Chaos Engineering 原则](https://principlesofchaos.org/)
- [LangGraph 容错与重试机制](https://langchain-ai.github.io/langgraph/concepts/functional_api/#retries)
