---
id: rag-online-eval
name: 线上 RAG 自动评测闭环
shortDesc: 在线采集 → 用户反馈 → 定期回流评测 → 失败样本回流，让"线上是不是变好了"有数据回答。
icon: chart-bar
difficulty: adv
tags: [Evaluation, Online-Eval, Feedback-Loop, Regression]
techFilters: []
accent: '#ec4899'
experience: false
prompts:
  - 上线后怎么知道这次改动变好了还是变差了？
  - 把今天的失败回答加入回归集。
---
## 概述

在 L1 确定性回归 / L2 手写 judge / L3 RAGAS 三层离线评测之上，补齐线上真实流量的评估闭环，回答"改任何模块 / prompt / 模型后，线上是不是变好了"。

## 为什么需要它

现状线上只跑不评：query / 检索 / 路由 / 答案 / 耗时未落库，无点赞点踩、无指标、失败样本不回流——"改没改好"没有数据。

## 核心思想

四段闭环：在线采集 → 用户反馈 → 定期回流评测 → 失败样本回流；影子评分与告警列为后续增强。

## 本项目的做法（规划中）

方案在 backend/rag/docs/线上RAG自动评测方案（在线评估闭环）.md。规划：采样落库、前端点赞 / 点踩、失败样本并入 eval_set 回归，复用三层离线评测脚本。

## 收益与边界

- 收益：改动能量化回归，质量可追踪；
- 边界：采样成本、隐私脱敏、样本标注质量。

## 演进与关联

observability-eval 的线上延伸；与 Harness 可观测性、CI 回归门禁互补。
