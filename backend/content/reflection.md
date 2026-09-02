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
enabledTools: [rag]
prompts:
  - 写一段 200 字的产品宣传文案：先给初稿，再自我批评，最后给出修订版。
  - 为「AI 智能体平台」写 5 条价值主张：先列初稿，指出每条不足并修订。
  - 用通俗语言解释 RAG：先写初稿，再从「读者是否听得懂」角度批评并修订。
  - 请详细说明差旅报销和日常报销的区别
---
## 概述

反思修订（Reflection / Self-Critique）让 Agent 先产出初稿，再以批评者视角审视不足，最后基于批评意见进行修订。经过多轮迭代，输出质量显著高于单次生成，是 Reflexion、Self-Refine 等机制的工程化落地。

核心原则：**先出结果、再回头审视**——让模型先专注产出，再切换成「批评者」视角挑毛病，而不是边写边自我怀疑。

## 为什么需要

- 单次生成的答案往往「一眼看不到盲区」，模型不知道自己的输出哪里不够好；
- 直接让模型「再想想」通常只是重复原话，缺乏真正的批判性审视；
- 多轮「草稿 → 批评 → 修订」能显著提升完整性、准确性，是自我改进最轻量的手段。

难点在于让「批评」真正有建设性而非泛泛而谈，以及防止修订循环发散。

## 通用设计思路

反思修订可拆成几个可独立设计的环节：

### 1. 生成环节

让模型产出一版完整答案（草稿）。生成与自我批判尽量分两次调用完成——先专注产出，再切换成「批评者」视角审视。

### 2. 评审环节（核心设计点）

评审要有建设性，不能只说「不够好」。主流做法：

- **结构化评分**：按维度打分（准确性 / 完整性 / 清晰度 / 逻辑性），可量化、可设达标阈值；
- **自由文本意见**：直接输出具体改进点，更贴近模型自然表达，但需用关键词 / 启发式判定是否通过；
- **进阶变体**：用**独立评审模型**（LLM-as-a-judge）避免「既当运动员又当裁判」的自我偏好；或把失败反思写成语言化记忆（Reflexion）供后续复用。

无论哪种，评审 prompt 都要要求「指出具体缺陷 + 给出可落地修改建议」，而不是泛泛评价。

### 3. 修订环节

把「原始问题 + 上一版答案 + 评审意见」拼给模型，让它产出优化后的**完整**回答。修订应是完整重写而非局部打补丁，否则迭代难以收敛。

### 4. 循环与终止（防止发散）

- **质量达标即停**：评分 ≥ 阈值，或评审输出 PASS 标记；
- **最大迭代上限**：无论是否达标都强制停止（兜底）；
- **一致性收敛**：相邻两轮输出几乎无变化时停止，避免无意义空转；
- 每轮成本是「生成 + 评审」两次模型调用，迭代上限要按成本承受力设定。

### 5. 可选增强

- 修订阶段允许调用工具 / 检索补充信息，让修订「有的放矢」；
- 保留每轮草稿与评审意见，形成可追溯的迭代记录。

## 本项目的做法

本项目用 LangGraph `StateGraph` 原生编排三节点循环，评审采用**流式自由文本**（`critique` 增量事件实时下发，生成到一半即可展示）：

```
START → generator ⇄ tools → critic →（评审通过 / 达最大迭代 / 达轮数上限）→ END
```

### 图与状态

```python
State: messages, draft, critique, iteration, max_iter, steps, stopped, passed, step_failed

builder = StateGraph(State)
START → generator
generator ⇄ tools（模型带工具调用时）
generator → critic（产出完整答案后）
critic → generator（未通过且未达 max_iter）| END（通过 / 达 max_iter）
```

### 生成节点（草稿 / 修订）

```python
async def generator(state):
    fresh   = 最近一条消息是 user          # 多轮会话：重置循环状态，避免残留 iteration/passed
    revising = iteration > 0 且 critique    # 已有评审反馈 → 修订阶段；否则草稿阶段

    steps += 1
    if steps > max_steps:                  # 轮数上限兜底（防工具死循环）
        return { stopped: "max_steps" }

    if 上一步不是工具回合:
        prompt = REVISE(问题, 草稿, 评审意见) if revising else GENERATOR(问题)
        追加 HumanMessage(prompt)

    # 流式生成：修订稿发 revise 事件，首稿发 message 事件
    msg = stream_model_call(..., output_event="revise" if revising else "message")

    if msg 含工具调用:
        return { messages: [msg], steps }  # 路由到 tools 执行后回到本节点
    if not revising:
        emit({ type: "reflect", stage: "draft" })
    return { messages: [msg], draft: msg, iteration+1, passed: False, steps }
```

### 评审节点（流式自由文本）

```python
async def critic(state):
    # 思考 → thinking 事件；评审文本 → critique 增量事件（实时下发评审过程）
    msg = stream_model_call(llm, [HumanMessage(f"原始问题：{query}\n草稿：{draft}\n\n输出评审结论")],
                            emit, system_prompt=CRITIC_SYSTEM, output_event="critique")
    return { critique: msg, passed: judge_text(msg) }

def judge_text(text):
    # 通过判定：空 / "无" / 以【PASS】或 PASS 开头 → 通过；显式含【FAIL】/【不通过】 → 不通过
    if 空 or text == "无": return True
    if "【FAIL】" in text or "【不通过】" in text: return False
    return 首行以 "【PASS】" 或 "PASS" 开头
```

### 条件路由与终止

```python
after_generator(state):
    if stopped == "max_steps": return END          # 达轮数上限直接结束
    return "tools" if 末条消息含工具调用 else "critic"

should_continue(state):
    return END if (passed 或 iteration >= max_iter) else "generator"
```

### 关键事件流

```
message（首稿）→ reflect(stage=draft) → critique 增量（评审过程）
  →（未通过）revise 增量（修订稿）→ critique 增量 ……
  →（通过 / 达上限）done
工具调用穿插：tool_start / tool_end（含 HITL 审批）
```

### 防死循环三道闸

| 机制 | 作用 |
|------|------|
| **max_iter** | 评审不通过时最多迭代的生成轮数，达上限强制结束 |
| **max_steps** | 累计模型调用/工具回合数上限，防「反复请求工具从不产出草稿」 |
| **recursion_limit** | LangGraph 递归深度兜底（Runner config 注入） |

### 与通用设计的对应关系

| 通用设计 | 本项目做法 |
|---------|-----------|
| 生成环节 | generator 节点，草稿 / 修订二阶段 |
| 评审环节 | critic 节点，流式自由文本 + `judge_text` 关键词判定 |
| 修订环节 | 未通过时回到 generator 以 `_REVISE_PROMPT` 重写 |
| 质量达标即停 | 评审以【PASS】/ 无开头判定通过 |
| 最大迭代上限 | `max_iter` 兜底 |
| 修订可调工具 | generator ⇄ tools 循环，工具结果回填后继续生成 |

## 收益与边界

- 全链路流式：草稿、评审意见、修订稿均逐 token 下发，评审过程实时可见
- 自由文本评审替代结构化评分：更贴近模型自然输出，通过判定容错高
- 三道终止机制兜底，循环永不发散
- 边界：评审质量依赖 critic prompt；`【PASS】` 关键词判定属于启发式，极端输出需依赖 max_iter 兜底

