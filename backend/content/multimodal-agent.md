---
id: multimodal-agent
name: 多模态 Agent
shortDesc: 让 Agent 能"看"——图像 / UI / 图表理解与视觉操作（OCR、截图、屏幕交互）。
icon: monitor
difficulty: adv
tags: [Multimodal, Vision, OCR, GUI-Agent]
techFilters: []
accent: '#fb7185'
experience: false
prompts:
  - 看这张截图，告诉我界面哪里有问题。
---
## 概述

多模态 Agent 把视觉（图像、UI 截图、图表）纳入理解与决策，支持看文档、读图表、操作界面（GUI Agent）。

## 为什么需要它

大量信息以图像 / 截图 / 图表承载，纯文本 Agent 看不见；视觉能力让"看 → 想 → 动"闭环成立。

## 核心思想

视觉输入（截图 / 图）+ 理解（OCR / 图表解析）+ 决策 + 动作（点击 / 输入）。平台已统一用多模态模型 qwen3.5-flash，具备基础视觉输入。

## 本项目的做法（规划中）

底层模型已支持多模态，但能力卡与流程未落地视觉场景。规划：新增图像理解 / 截图问答 / UI 操作工具链。

## 收益与边界

- 收益：信息面扩大、可处理文档图表；
- 边界：视觉 token 成本高、GUI 稳定性差、隐私风险。

## 演进与关联

与 computer-use（桌面自动化）协同；Context 层补充"视觉上下文"。
