---
id: agent-skills
name: Agent Skills 技能封装
shortDesc: 把可复用的专业能力打包成"技能"（含提示词、脚本与说明），按需注入上下文（AGENTS.md / Claude Skills）。
icon: cube
difficulty: adv
tags: [Skills, AGENTS-MD, Reusable, Packaging]
techFilters: []
accent: '#f97316'
experience: false
prompts:
  - 让 Agent 学会处理财务报销这类专业流程。
---
## 概述

Agent Skills（Anthropic 2025 提出）把可复用的专业能力打包成独立"技能包"：PROMPT.md（触发说明）+ 示例 + 脚本，按需注入上下文，与项目代码解耦。

## 为什么需要它

能力沉淀在单条 prompt 里不可复用、难维护；技能包让"会做某类事"成为可安装、可组合的模块。

## 核心思想

技能 = 元数据（触发场景）+ 指令（怎么做）+ 资源（示例 / 脚本）；Agent 按任务自动选择并注入相关技能，避免全量塞进上下文。

## 本项目的做法（规划中）

能力集中在 system prompt 与 tools_builder。规划：拆出技能目录（frontmatter + prompt + resources），运行时按需装配。

## 收益与边界

- 收益：能力复用、上下文省 token、可协作迭代；
- 边界：技能粒度设计、冲突与版本管理。

## 演进与关联

介于 Protocol（MCP）与 Prompt 之间的一层"可复用指令资产"；可视为工具层的组织化。
