---
id: memgpt
name: MemGPT / 分层记忆 Agent
shortDesc: 借鉴 OS 虚拟内存，把上下文当作"分页内存"换入换出，实现长程记忆与复杂任务。
icon: history
difficulty: adv
tags: [MemGPT, Memory, Paging, Long-Horizon]
techFilters: []
accent: '#8b5cf6'
experience: false
prompts:
  - 做一个需要连续 10 轮维护状态的复杂任务。
---
## 概述

MemGPT（Letta）借鉴 OS 虚拟内存：上下文是"RAM"，外部记忆是"磁盘"，由函数调用把记忆分页换入换出，让 Agent 处理长程、复杂任务而不失忆。

## 为什么需要它

长对话、长任务会撑爆上下文或丢失早期事实；MemGPT 主动管理"哪些内容在上下文、哪些在外存"。

## 核心思想

核心上下文（工作记忆）+ 外部记忆（档案 / 会话 / 工具文档分层）由 LLM 自我管理换页；配睡眠压缩归档。

## 本项目的做法（规划中）

现有 memory 卡已做"跨轮向量记忆"，但无 MemGPT 的主动分层换页。规划：参考其分层记忆 + 归档压缩，增强长任务。

## 收益与边界

- 收益：长程任务稳定、上下文可控；
- 边界：换页策略复杂、成本上升、易翻车需回滚。

## 演进与关联

memory（跨轮记忆）的进阶形态；与 context-mgmt（压缩）互补。
