---
id: text-to-sql
name: Text-to-SQL 结构化查询
shortDesc: 把自然语言转成 SQL 查库表，覆盖统计 / 排名 / 跨表聚合类问题，检索从"文本语义"走向"结构计算"。
icon: database
difficulty: adv
tags: [Text-to-SQL, NL2SQL, Structured-Query, Schema]
techFilters: []
accent: '#10b981'
experience: false
prompts:
  - 上月各部门报销总额排名前五有哪些？
  - 系统里一共接入了多少个知识库？
---
## 概述

文本语义检索答不了"上月报销总额""部门人数排名"这类聚合计算问题。Text-to-SQL 把自然语言转成 SQL 查库表，覆盖统计 / 排名 / 跨表聚合类问题，让检索从"文本语义"走向"结构计算"。

## 为什么需要它

RAG 检索的是"语义相似"，SQL 算的是"结构事实"。统计类问题语义检索只能"猜答案"，SQL 能"算答案"。

## 核心思想

Schema 注入 → SQL 生成 → 安全校验（只读 / 白名单 / 权限）→ 执行 → 结果转述。关键在护栏：只读连接、LIMIT 约束、失败时兜底转文本检索。

## 本项目的做法（规划中）

modular-rag 明示暂未实现结构化查询（边界清楚）。规划：接入库表 Schema + 只读 SQL 执行器 + 权限管控，作为 Agent 工具或 RAG 的补充能力。

## 收益与边界

- 收益：统计 / 聚合 / 排名类问题从"不可答"到"精确答"；
- 边界：schema 依赖、权限审计成本、生成质量不稳定，需兜底。

## 演进与关联

与多知识库路由（D2）、function-calling 互补；是 RAG 从"检索"走向"计算"的关键一步。
