---
id: online-hybrid-retrieval
name: 在线混合检索策略
shortDesc: 向量+BM25 双路召回与 RRF/加权/重排融合策略，在线问答的召回质量核心。
icon: route
difficulty: int
tags: [RAG, Hybrid-Search, RRF, Rerank, Retrieval]
techFilters: [Qdrant]
accent: '#0ea5e9'
---
## 概述

在线混合检索是问答阶段的核心召回策略：把**稠密向量检索（语义）**与**稀疏关键词检索（BM25）**两条互补信号**并行召回**，再用 RRF / 加权 / 重排等策略**融合**，得到最终喂给生成模型的高质量上下文。它是 Advanced RAG 检索质量的工程核心（详见 [advanced-rag.md](advanced-rag.md)）。

```
Query → 查询改写 → 双路召回（各取 top-50~100）→ 融合（RRF/加权）
  → Rerank 精排 → 上下文压缩/选择 → 拼装 → LLM
```

## 为什么需要（两条信号的盲区互补）

- **向量（稠密）**：擅长语义泛化——「怎么退个税」能召回「个人所得税退税流程」；但对**专有名词、错误码、型号、人名**等精确字符串失准（embedding 几乎无法区分未见过的新字符串）；
- **BM25（稀疏）**：精确匹配强——错误码/型号一打即中；但**不懂语义**，问法一改就漏（「年假」vs「休假」）；
- 混合检索 = **「懂语义」+「认关键字」**，Recall@k 通常显著高于单路。

## 两条召回信号

| 信号 | 原理 | 擅长 | 弱项 |
|------|------|------|------|
| **稠密（Dense）** | embedding 编码 + 余弦相似度最近邻 | 近义、口语化、模糊概念、跨语言 | 未见过的精确字符串、长尾实体 |
| **稀疏（Sparse）** | BM25 词频-逆文档频率打分 | 专有名词 / ID / 错误码 / 型号 / 短查询 | 语义泛化弱 |

> 进阶：SPLADE 等**学习型稀疏表示**可提升稀疏信号的泛化能力，兼顾关键词与语义。

## 查询侧处理（检索前）

- **查询改写**：把口语化/含混问题改写成更贴近语料的表达；
- **Multi-Query / HyDE**：多子查询扩展、假设文档嵌入增强召回（详见 [rag-variants.md](rag-variants.md)）。

## 融合策略（核心）

| 策略 | 做法 | 特点 |
|------|------|------|
| **RRF（倒数排名融合）** | 按排名而非分数融合：每路文档计 `1/(k+rank)` 求和 | 无需分数校准、量纲无关、工程最稳（k 常取 60，10-60 区间调试） |
| **加权求和（Weighted Sum）** | 两路分数归一化后线性组合：`α·norm(dense) + (1-α)·norm(sparse)` | 可调权重（α 常 0.3-0.7）；**归一化必做**，否则量纲不同无法相加 |
| **统一重排（Rerank）** | 两路各取 top-n（50-100）合并去重 → Cross-Encoder 精排 → 取 top 2-4 | 准确率最优，延迟略高 |

**多级检索架构（工程推荐）**：初检要求「广」（大 Top-N，避免早期截断丢答案），精排要求「准」（小 Top-K）。RRF 是最稳的融合起点，专有名词多的场景可改用加权融合，最终用重排兜底——形成「召回 → 融合 → 重排 → 评测」闭环。

## 工程实现路径

- **路径 A（ES 原生）**：Elasticsearch 的 `dense_vector` 存向量 + `text` 字段做 BM25，retriever 内用 RRF 一条查询搞定；
- **路径 B（向量库 + 外置 BM25）**：Qdrant/Milvus 管向量召回，BM25 用 Lucene/ES/Whoosh 独立跑，应用层用 RRF 或加权融合两份 top-n（**本项目**）；
- **路径 C（统一重排）**：两路各取 top-n 合并去重后，送 Cross-Encoder 重排器（如 bge-reranker-large）重新打分，取最相关 2-4 段送入生成。

## 关键参数与取舍

- **RRF 的 k**：60 为经验默认；k 越大排名差异被稀释得越平，k 越小头部权重越重；
- **加权 α**：语义类查询升 α，专名类查询降 α，用验证集调；
- **各路 top-n**：宁多勿少（50-100），避免早期截断丢答案；
- **归一化**：加权求和前必须 min-max / z-score，否则不同量纲的分数无法相加；
- **元数据过滤**：检索前按权限/部门/时间过滤，减少误召回与越权。

## 评估

- **Recall@k**：前 k 个结果是否含正确答案（检索召回质量）；
- **nDCG / MRR**：排序质量；
- **Faithfulness**：答案是否忠实于检索片段（生成质量，可用 RAGAS / TruLens 自动化）；
- 对比实验设三组：纯向量 / 纯 BM25 / 混合，通常混合组 Recall@10 领先 5-15 个百分点（随数据分布变化）。

## 本项目的做法

- **双路召回**：Qdrant 向量语义检索 + BM25 关键词检索并行；
- **融合**：RRF（倒数排名融合）合并去重；
- **精排**：Reranker 对融合候选二次打分，把真正相关的片段顶到 Top-K；
- **生成**：拼装高相关上下文 + 用户问题，交给 LLM 作答。

## 收益与边界

**收益**

- 语义与关键词互补，召回率显著提升；
- RRF 融合免分数校准、稳定易用；
- 多级检索兼顾召回广度与排序精度。

**边界 / 局限**

- 混合检索本质仍是「文本匹配」，无法处理跨文档实体关联与多跳推理（需 [graph-rag.md](graph-rag.md)）；
- 每层优化叠加延迟，实时性场景需在效果与速度间权衡；
- 查询改写 / 重排依赖模型质量，需评估闭环定位瓶颈。

## 参考链接

- **RRF 融合**：Cormack et al., 2009,《Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods》
- **Qdrant Hybrid Search / Universal Query API**：官方文档（RRF 融合、多级检索示例）
- **Elastic 混合检索实验（RRF 与加权参数化）**：[Improving information retrieval in the Elastic Stack: Hybrid retrieval](https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid)
- **混合检索工程实践**：[向量召回与关键词召回的融合（RRF / 加权 / 统一重排路径）](https://cloud.tencent.cn/developer/article/2728960)
