---
id: hitl
featured: true
name: 审批门（人在回路）
shortDesc: 执行前审批机制，关键决策暂停等人工确认，安全与效率的平衡。
icon: check
difficulty: int
completeLevel: 95
tags: [HITL, Approval, Safety, Guardrails]
techFilters: [LangGraph]
accent: '#22d3a8'
---
## 概述

人在回路（Human-in-the-Loop，HITL）是 Harness 的「自主权边界」：Agent 在执行高风险操作前**暂停**，等待人工审批。审批通过才继续执行，被拒绝则终止或调整方案。它的意义不是让人类做所有事，而是在正确的时机让人类做关键决策。

## 为什么需要

Agent 的自主权越大，风险也越大。在关键决策点、高风险操作、结果不确定时，需要人工把关：

- 一次误操作（删库、付款、发信）可能造成不可逆损失，光靠提示词约束不可靠；
- Agent 自主循环可能失控，需要人类随时叫停或接管；
- 合规要求：高危操作必须有「人」的签字确认记录。

HITL 解决的是信任问题——「机器敢做」与「人类放心」之间的平衡。

## 通用设计思路

### 介入模式

| 模式 | 时机 | 说明 |
|------|------|------|
| 事前审批 | 执行高风险操作前 | 操作前暂停，等待人工确认（删文件、发送、支付） |
| 事中监督 | 执行过程中 | 人类随时查看进度、干预、暂停、接管 |
| 事后审核 | 任务完成后 | 审核结果确认后才生效（合同、代码 PR） |
| 异常升级 | 遇到无法解决的问题 | 自动升级给人工，而不是无限重试或强行回答 |
| 协作编辑 | 生成初稿后 | 人类在 Agent 输出基础上编辑修改 |

### 技术实现要素

- **检查点 + 中断恢复**：Agent 在关键节点暂停，保存完整状态（checkpoint），等人工输入后用同一检查点恢复；
- **条件分支**：高风险操作走人工审批节点，低风险直接执行；
- **权限分级**：不同操作不同风险等级 → 自动执行 / 通知即可 / 需审批 / 禁止；
- **超时处理**：人工长时间未响应时自动降级（跳过 / 默认拒绝 / 升级给更高权限）。

## 本项目的做法

本项目用 LangGraph 原生 `interrupt` + checkpointer 实现审批门，走「暂停 → 事件通知 → 人工决策 → 恢复」闭环：

```
模型请求工具 → interrupt 暂停（保存检查点）→ 后端发 approval_request 事件
  → 前端弹窗 → POST /api/approve（approve / reject / modify）
  → 用同一 checkpointer 重建图，Command(resume=...) 恢复执行
```

### 审批策略判定

```python
# 审批策略：approval_policy = always | never（经 config['configurable'] 传入）
# 高危工具无论策略如何都强制 HITL
ALWAYS_APPROVE_TOOLS = {"run_command"}   # 命令执行等不可逆高危操作

def should_approve(approval_policy, tool_name):
    return approval_policy == "always" or tool_name in ALWAYS_APPROVE_TOOLS
```

### 中断与恢复（伪代码）

```python
# tools 节点：任一本步工具需要审批即整批暂停
if any(should_approve(policy, c.name) for c in calls):
    payload = [{ "name": c.name, "args": c.args, "id": c.id } for c in calls]
    decision = interrupt({ "tool_calls": payload })   # 暂停，等待人工输入

# 人工决策分流
if decision.action == "reject":
    # 注入「用户拒绝」ToolMessage，模型需改用其它方式或询问用户
    return ToolMessage("用户拒绝了该工具调用，请改用其它方式或询问用户。")
if decision.action == "modify":
    args = decision.modified_args                 # 覆写工具参数后再执行

# 恢复：Runner 检测到 interrupt 后不再抛异常，
# 用同一 checkpointer 重建图，Command(resume=decision) 从断点继续
```

### 关键细节

- **审批与会话映射**：`approval_id → {session, interrupt_ids}`（同一 superstep 可能有多个 interrupt），恢复时据此 resume；
- **恢复接口**：`POST /api/approve` 携带 `decision` 与 `modified_args` → `runner.resume(...)`；
- **强制 HITL**：`run_command` 位于 `ALWAYS_APPROVE_TOOLS`，无论策略如何都必须人工确认命令本身；
- **多代理收敛**：子代理不持有 checkpointer、不触发 HITL，审批统一收敛到编排者层；
- **止损**：`POST /api/stop` 取消进行中的后台图任务，及时停掉执行、节省 token。

### 澄清提问（ask_user）：HITL 的第二类中断

除了「审批工具调用」，HITL 还有一种常见场景：Agent 缺少**只有用户能提供的信息**（工具无法获取、模型不应臆测的私有信息）。此时不应凭空编造，而应暂停向用户澄清。本项目用 `ask_user` 工具触发第二类中断：

```
模型调 ask_user(questions=[{question, options}, ...]) → interrupt 暂停
  → 后端发 ask_user_request 事件（一次携带全部 questions）
  → 前端回复卡片（逐题点选 / 输入 / 跳过）→ POST /api/approve（decision=reply + answers）
  → 恢复后把「Q1 <问题文本>：<回答> / Q2 <问题文本>：用户跳过未回答」
    格式化为 ToolMessage 返回给模型继续执行
```

设计要点：

- **一次问全**：`questions` 列表一次列出所有缺失项（每项含问题文本与候选 `options`），避免一问一答多次往返；系统提示明确要求模型批量提问；
- **强制中断**：`ask_user` 无论审批策略（always / never）如何都强制中断——澄清不是审批，但同样需要等待用户输入（`do_approval = name == "ask_user" or should_approve(...)`）；
- **跳过处理**：用户可逐题跳过 / 无法回答，恢复后逐题标记「用户跳过未回答」；模型收到后应基于其余已确认信息继续推进，不重复追问同一问题；
- **轮次豁免**：澄清交互（提问 + 回答）不计入 Agent 轮数上限，避免因提问消耗执行预算导致任务提前终止（react 用 `AskFreeCallLimitMiddleware` 判断「模型输出仅含 ask_user 调用则不计轮」；plan_execute / reflection 用 `is_ask_reply_msg` 识别澄清回复 ToolMessage 后跳过计数）；
- **参数归一化**：模型经常把 `questions` / `options` 以多种形态返回——JSON 字符串、真实换行包裹的数组文本、双重转义（内容含字面 `\"`/`\n`）、整体被序列化成字符串的 arguments、尾部带杂质等；后端统一归一化为结构化列表后再进中断（先解析外层、再剥离一层转义、strip 杂质后 `json.loads`），保证卡片渲染与回答格式化对齐。若解析后为空则不中断、返回结构化错误让模型重新组织参数，并打印 `[ask_user] 参数无效` 诊断日志（含实际 args）供排查；
- **截断兜底（双保险）**：模型输出偶发被截断、`questions` 数组缺闭合括号（以 `}` 结尾缺 `]`）导致解析失败时，自动补 `]` 再解析一次；同时后端已移除各 LLM 场景的 `max_tokens` 限制——「源头不限长」防截断 + 「补全再解析」兜底，两层保障缺一不可；
- **子代理不提问**：multi-agent worker 无 checkpointer、不触发澄清中断，提问收敛到主代理。

与「工具审批」的对比：

| 维度 | 工具审批 | 澄清提问（ask_user） |
|------|---------|---------------------|
| 触发时机 | 工具执行前 | Agent 缺用户私有信息 |
| 人工决策 | 批准 / 拒绝 / 修改参数 | 逐题回答 / 跳过 |
| 恢复结果 | 执行或拒绝工具 | 问答文本注入模型上下文 |
| 轮次计数 | 计入轮数 | 豁免（不计入） |
| 依赖审批策略 | 是（always 或强制工具） | 否（始终中断） |

### 与通用设计的对应关系

| 通用设计 | 本项目做法 |
|---------|-----------|
| 事前审批 | 工具执行前 `interrupt` 暂停 |
| 检查点 + 中断恢复 | LangGraph checkpointer + `Command(resume=...)` |
| 条件分支 | `should_approve`：always 或强制 HITL 工具才审批 |
| 权限分级 | `approval_policy`（always/never）+ `ALWAYS_APPROVE_TOOLS` 高危清单 |
| 审批界面 | `approval_request` 事件 → 前端弹窗 → `/api/approve` |
| 事中监督 / 止损 | `POST /api/stop` 取消运行 |

## 收益与边界

- 基于 LangGraph interrupt 原生实现，非 hack
- 支持按工具风险等级配置审批策略
- 审批状态持久化，服务重启不丢失待审任务
- 边界：HITL 依赖人工响应，长时间无响应任务会挂起，需超时策略兜底；审批只覆盖工具调用层，不覆盖「模型自身输出风险」的审核

