---
id: rag-variants
name: RAG 专项增强技术
shortDesc: 可叠加在任意基础 RAG 上的增强插件：Self-RAG、CRAG、HyDE、RAPTOR 等。
icon: sparkles
difficulty: adv
tags: [RAG, Self-RAG, CRAG, HyDE, RAPTOR]
techFilters: []
accent: '#f472b6'
---
## 概述

以下技术**不是独立的 RAG 代际**，而是可叠加在任意基础 RAG（Naive / Advanced / Modular）上的「增强插件」：针对某一具体痛点（检索不准、覆盖不全、幻觉、多轮、跨模态、结构化数据）做定向修补。选型时应先确定基础范式，再按痛点叠加对应插件。

## 1. Hybrid RAG（混合检索增强）

- **核心思想**：向量 + 关键词（+ 图谱）多路召回，互相补短；
- **实现方式**：多路并行检索（向量语义 + BM25 关键词，可选加图谱检索）→ RRF 融合去重 → Rerank 精排；
- **适用场景**：大多数生产场景的通用基线；
- **收益/边界**：召回率显著提升；仍无法处理跨文档实体关联。

## 2. HyDE（假设文档嵌入）

- **核心思想**：先让 LLM 生成一段「假设答案」，再用它做检索——假设答案比原问题更接近文档的表达方式；
- **实现方式**：LLM 生成假想文档 → 假想文档向量化 → 用其向量检索 → 把真实检索结果喂回真实作答；
- **适用场景**：查询与语料「问法」不一致、零样本难以改写时；
- **收益/边界**：显著缓解「问法漂移」；多一次 LLM 调用，成本与延迟略增。

## 3. Multi-Query（多查询扩展）

- **核心思想**：一个问题拆成多个子查询分别检索，覆盖不同表述；
- **实现方式**：LLM 生成 N 个改写查询 → 多路召回 → 去重合并 → 拼上下文；
- **适用场景**：查询意图多样、单次检索覆盖不全；
- **收益/边界**：召回更全；检索次数成倍增加，需控制 N 与去重质量。

## 4. Self-RAG（自反思检索）

- **核心思想**：让模型用**反思 token** 自主决策「要不要检索、检索片段相不相关、答案有没有文档支撑」；
- **实现方式**：微调带反思 token 的模型 → 按 token 决定检索与生成路径 → 无支撑不硬答；
- **适用场景**：防幻觉、按需检索（常识问题不查库）；
- **收益/边界**：按需检索省 token、答案有据可依；依赖带反思 token 的微调模型，落地成本高。

## 5. CRAG（纠正式检索）

- **核心思想**：检索质量差时**自动降级**到网络搜索等兜底，而不是硬用低质量结果；
- **实现方式**：质量评估器对检索结果分级 → 高质量直接用 / 中质量修正后重检 / 低质量走网络检索兜底；
- **适用场景**：知识库覆盖不全、答案可能过时；
- **收益/边界**：显著提升覆盖与时效；引入评估器与兜底检索，链路更重。

## 6. RAPTOR（递归摘要树）

- **核心思想**：递归聚类 + 摘要，构建「树状分层索引」，既看全局概览也看局部细节；
- **实现方式**：自底向上对 chunk 聚类 → 生成各层摘要 → 检索时多粒度取（可结合 Graph 思路）；
- **适用场景**：文档量大、既需全局主题又需局部细节；
- **收益/边界**：多粒度理解优于平铺切块；建树成本较高。

## 7. Conversational RAG（对话式 RAG）

- **核心思想**：在多轮对话中维护会话历史与引用上下文，检索与问答都考虑上文；
- **实现方式**：会话级检索 + 历史状态记忆 + 引用上下文传递；
- **适用场景**：客服、对话式知识问答；
- **收益/边界**：多轮体验连贯；需处理历史压缩与指代消解。

## 8. Multimodal RAG（多模态 RAG）

- **核心思想**：图文表跨模态统一向量空间，图表内容也能被检索；
- **实现方式**：多模态 embedding + 跨模态检索 + 多模态生成；
- **适用场景**：含图表/图片/表格的文档问答；
- **收益/边界**：补齐非文本信息；多模态 embedding 对齐成本高。

## 9. Text-to-SQL RAG（结构化数据检索）

- **核心思想**：自然语言转 SQL 直接查结构化数据库，把结果回填上下文；
- **实现方式**：LLM 生成 SQL → 执行 → 结果（或摘要）拼进 prompt → 作答；
- **适用场景**：业务数据库即问即查、BI 报表；
- **收益/边界**：结构化查询精确；依赖 schema 描述与 SQL 正确性，需防注入。

## 选型建议

| 痛点 | 推荐插件 |
|------|---------|
| 检索召回不全、问法漂移 | HyDE / Multi-Query / Hybrid RAG |
| 上下文噪声大、污染 | Hybrid RAG + Rerank / 上下文压缩 |
| 幻觉、答案无据 | Self-RAG / CRAG |
| 知识库覆盖不全 | CRAG（网络兜底） |
| 文档量大、要全局概览 | RAPTOR / Graph RAG |
| 多轮对话场景 | Conversational RAG |
| 文档含图表 | Multimodal RAG |
| 结构化数据库 | Text-to-SQL RAG |

## 参考链接

- **Self-RAG**：Asai et al., 2023,《Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection》(arXiv:2310.11511)
- **CRAG**：Yan et al., 2024,《Corrective Retrieval Augmented Generation》(arXiv:2401.15884)
- **HyDE**：Gao et al., 2022,《Precise Zero-Shot Dense Retrieval without Relevance Labels》(arXiv:2212.10496)
- **RAPTOR**：Sarthi et al., 2024,《Recursive Abstractive Processing for Tree-Organized Retrieval》(arXiv:2401.18059)
- **RRF 融合**：Cormack et al., 2009,《Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods》
