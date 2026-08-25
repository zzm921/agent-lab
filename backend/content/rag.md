---
id: rag
name: 检索增强生成
shortDesc: RAG 知识总表：五代范式演进路线 + 各类型 RAG 导航总览。
icon: database
difficulty: int
completeLevel: 85
tags: [RAG, Qdrant, Embedding, Hybrid-Search, Rerank]
techFilters: [Qdrant, FastAPI]
accent: '#38bdf8'
enabledTools: [rag]
prompts:
  - 用 RAG 检索知识库回答：什么是检索增强生成？它解决什么问题？
  - 检索「多智能体」相关文档，总结三种主流编排方式。
---
## 概述

RAG（Retrieval-Augmented Generation，检索增强生成）把**外部知识库**接进大模型生成流程：回答问题时先检索最相关的文档片段，再把片段作为上下文交给模型作答。一句话：**先查资料、再写答案**——让模型从「闭卷硬背」变成「开卷考试」。

本文档是 RAG 知识体系的**总表与导航**，承担两件事：

1. 讲清楚 RAG 的**技术演进路线**——五代范式为什么一步步长成这样；
2. 给出一张**全类型总表**，每个类型对应一份独立详解文档。

## 为什么需要

大模型的根本局限是「只记得训练时见过的知识」：

- **知识过期**：模型有知识截止日期，训练后无法获知新信息；
- **私有数据不可见**：企业内部文档、个人资料从未进入训练集；
- **幻觉**：模型对不知道的事会「一本正经地编」，且难以给出依据；
- **微调代价高**：每次知识更新都重新训练不现实。

RAG 的本质优势：**知识可更新、答案可溯源、无需重训模型、成本可控**——是企业知识库场景的事实标准。与替代方案（微调 / 传统搜索 / 纯长上下文）的取舍详见各分文档与通用知识部分。

## 技术演进路线（五代范式）

RAG 自 2020 年 Lewis 奠基（arXiv:2005.11401）起，演进本质是：**把「外部知识 → 检索 → 生成」这条链条不断解耦、模块化、结构化、自主化**。

```
Naive RAG ──→ Advanced RAG ──→ Modular RAG ──→ Graph RAG ──→ Agentic RAG
(2020 奠基)   (补丁式优化)      (模块化组合)     (结构化关系)   (自主决策)
```

每一代解决的，都是上一代留下的「盲区」：

| 代际 | 核心特征 | 解决上一代的什么盲区 |
|------|---------|---------------------|
| **Naive RAG** | 固定三步：切块 → 向量检索 → 生成 | 奠基形态，把「检索即记忆」跑通 |
| **Advanced RAG** | 检索前（改写/意图分类）+ 检索后（重排/压缩）优化 | Naive 的召回噪声多、无关键词命中、上下文被污染 |
| **Modular RAG** | 拆成可插拔模块+算子，出现路由/调度/融合 | Advanced 仍是「整条管道」，组件不可复用、不可组合 |
| **Graph RAG** | 实体-关系图谱，图索引→图检索→图增强生成 | 平面向量的「关系盲区」：跨文档关联、多跳推理 |
| **Agentic RAG** | Agent 自主决定何时检索、检索几次、够不够、何时停 | 前三代受「先检索后生成」束缚，无法按需自适应 |

**演进主线一句话**：从「被动管道」走向「主动系统」——前四代逐步把检索链路的每一环做深做结构化，Agentic 则把整条链路交给 Agent 自主决策。

> 选型原则：**不是选最先进，而是选最匹配当前数据规模、问题复杂度与延迟要求的那一个**。数据小、问题单一、要快 → Naive/Advanced；要跨实体理解 → Graph；问题复杂多步、覆盖不全 → Agentic。

## RAG 类型总表（导航）

| 类型 | 一句话定位 | 独立详解文档 |
|------|-----------|-------------|
| **Naive RAG（朴素）** | 极简基线，固定三步，一切 RAG 的原点 | [naive-rag.md](naive-rag.md) |
| **Advanced RAG（高级）** | 检索前后全链路优化，工业界主流 | [advanced-rag.md](advanced-rag.md) |
| **Modular RAG（模块化）** | 模块可插拔、可路由，按需动态组合 | [modular-rag.md](modular-rag.md) |
| **Graph RAG（图谱）** | 实体-关系图谱 + 多跳推理 | [graph-rag.md](graph-rag.md) |
| **Agentic RAG（智能体）** | Agent 自主决策检索策略 | [agentic-rag.md](agentic-rag.md) |
| **专项增强技术** | 可叠加的插件：Self-RAG / CRAG / HyDE / RAPTOR 等 | [rag-variants.md](rag-variants.md) |

## 通用管线速览

无论哪种 RAG，底层都共享一条「离线建库 + 在线问答」管线：

```
离线（建库）：加载 → 分块 → 向量化 → 索引入库
在线（问答）：查询理解 → 检索 → 增强(重排/压缩) → 生成 → 评估反馈
```

七个环节的关键知识点（分块策略、Embedding、向量库、混合检索、Rerank、生成约束、两段评估）在**分文档的「为什么需要/核心设计」中按类型展开**，此处不再重复。

两个横向工程主题单独成文：

- **离线数据处理（建库）**：复杂文档解析、热门解析工具、分块与存储策略——详见 [offline-processing.md](offline-processing.md)；
- **在线混合检索策略（问答）**：向量 + BM25 双路召回、RRF/加权/重排融合——详见 [online-hybrid-retrieval.md](online-hybrid-retrieval.md)。

## 本项目的做法

本框架把 RAG 做成**多方案可选框架**：同一份内置语料（虚构「科创公司员工行政、考勤、福利与差旅全管理制度」）按方案写入各自独立的集合（`knowledge_{scheme_id}`），侧边栏可随时切换方案，直观对比检索与回答差异。当前已落地 **naive / advanced / modular** 三方案，graph / agentic 在同一框架上规划扩展。

- **naive（朴素 RAG）**：固定切块 + 纯稠密向量语义检索（DashScope `text-embedding-v3`），最简基线——详见 [naive-rag.md](naive-rag.md)；
- **advanced（高级 RAG）**：语义分块 + Query 重写（Multi-Query）+ 混合检索（稠密 + 稀疏 `text-sparse-embedding-v1`，Qdrant 内置 RRF 融合）+ Rerank 精排，补足关键词盲区——详见 [advanced-rag.md](advanced-rag.md)；
- **modular（模块化 RAG）**：前置语义路由（结构化路由决策）→ 执行计划 → 动态编排模块执行；含查询分解、上下文压缩与多跳（规划-执行-验证）迭代检索——详见 [modular-rag.md](modular-rag.md)。

存储层以「存储后端」抽象解耦，Qdrant 与 Elasticsearch 均已实现（advanced / modular 默认跨后端多路召回）；未配置后端或连接失败时回退内存检索，离线/测试可跑通。

## 收益与边界

**收益**

- 纯语义召回：长文本固定切块后按语义相似度召回最相关片段（naive 方案）；
- 方案可扩展：同一语料下逐代叠加 advanced / modular / graph / agentic，检索与回答差异一目了然；
- 存储解耦：换向量库（如接 ES）只改后端实现，RAG 方案层不动；
- 知识可更新、答案可溯源，减少幻觉。

**边界 / 局限**

- **检索决定上限**：检索质量差则「垃圾进、垃圾出」，再强的模型也救不回；
- **关系盲区**：向量+关键词无法处理跨文档实体关联与多跳——需 Graph RAG；
- **固定管道**：假设「所有问题都检索一次、结果都可用」——复杂问题需 Agentic RAG；
- **权限与合规**：强隔离场景依赖元数据过滤；
- **缺少评估闭环**：没有独立的检索/生成质量评估，优化无从下手。

## 演进与关联

RAG 是「上下文工程」的核心手段之一，与 Agent 演进线深度交汇：

- **向 Agent 延伸**：Agentic RAG 把检索封装成工具，融入 ReAct 循环（见 [agentic-rag.md](agentic-rag.md)）；
- **向图谱延伸**：Graph RAG 用实体关系补足向量检索的「关系盲区」（见 [graph-rag.md](graph-rag.md)）；
- **向 Harness 延伸**：RAG 的检索质量评估、权限过滤、数据治理依赖观测与安全组件支撑。

## 参考链接

- **RAG 奠基论文**：Lewis et al., 2020,《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》(arXiv:2005.11401)
- **RAG 综述（三代范式划分）**：Gao et al., 2024,《Retrieval-Augmented Generation for Large Language Models: A Survey》(arXiv:2312.10997)
- **RAG 最佳实践**：Wang et al., 2024,《Demystifying RAG: A Comprehensive Guide to Retrieval-Augmented Generation》
- **GraphRAG**：Microsoft Edge et al., 2024,《From Local to Global: A Graph RAG Approach to Query-Focused Summarization》(arXiv:2404.16130)
- **Self-RAG**：Asai et al., 2023,《Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection》(arXiv:2310.11511)
- **CRAG**：Yan et al., 2024,《Corrective Retrieval Augmented Generation》(arXiv:2401.15884)
- **HyDE**：Gao et al., 2022,《Precise Zero-Shot Dense Retrieval without Relevance Labels》(arXiv:2212.10496)
- **RAPTOR**：Sarthi et al., 2024,《Recursive Abstractive Processing for Tree-Organized Retrieval》(arXiv:2401.18059)
- **RAG 技术演进综述文章**：[从 NaiveRAG 到 AgenticRAG 的范式演进](https://cloud.tencent.cn/developer/article/2714559) / [RAG 全体系技术分类](https://blog.csdn.net/qq_37325859/article/details/163163446)
