---
id: security
name: 安全防护
shortDesc: 三层 Guardrails、Prompt 注入防御、权限沙箱与数据脱敏，守住 Agent 的安全边界。
icon: lock
difficulty: adv
tags: [Security, Guardrails, Prompt-Injection, RBAC]
techFilters: [MCP, FastAPI]
accent: '#f43f5e'
experience: false
---
## 为什么需要它

Agent 从"说了什么"扩展到"做了什么"，风险面同步扩大。安全防护覆盖：输入 / 输出 / 工具三层 Guardrails（过滤越狱与恶意诱导、校验输出、工具白名单）；Prompt 注入防御（区分可信指令与不可信外部内容）；最小权限 RBAC；敏感操作审计日志与数据脱敏。代表实践：OpenAI Guardrails、LangGraph HITL、OWASP LLM Top 10。

## 怎么解决

难点在"防得住"与"不误伤"的平衡——过于严格的护栏降低 Agent 效能；注入攻击可通过间接渠道（网页内容、工具返回、记忆投毒）绕过单点过滤。业界做法：多道防御纵深 + 来源可信分级 + 最小权限 + 全链路审计。

## 核心实现

```python
# 三层 Guardrails：输入 → 工具 → 输出
async def agent_with_guardrails(user_input):
    input_ok = await input_guard.validate(user_input)  # 越狱/注入过滤
    if not input_ok:
        return polite_refusal()

    tool_call = await llm.choose_tool(user_input)
    if tool_call.name not in TOOL_WHITELIST:           # 工具白名单
        return block("tool not allowed")

    result = await tool_call.execute(scope=minimal_rbac)
    return await output_guard.scan(result)             # 输出复核 + 脱敏
```

## 收益与边界

- 输入 / 输出 / 工具三层 Guardrails 纵深防御
- Prompt 注入与记忆投毒防御（来源可信分级）
- 最小权限 RBAC + 敏感操作审计与数据脱敏
