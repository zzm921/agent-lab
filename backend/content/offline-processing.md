---
id: offline-processing
name: 离线数据处理
shortDesc: 复杂文档解析、分块与存储策略——离线建库决定 RAG 质量的天花板。
icon: archive
difficulty: adv
tags: [RAG, Document-Parsing, Chunking, Vector-DB, Storage]
techFilters: [Qdrant]
accent: '#10b981'
---
## 概述

离线数据处理（建库）是 RAG 的**第一道工序**：把杂乱的企业文档解析、清洗、分块、向量化，变成可检索的索引。它决定 RAG 质量的天花板——**「垃圾进、垃圾出」（GIGO）**：检索与生成再强，也救不回解析失真的上游。

```
加载 → 解析（含 OCR / 表格 / 公式提取）→ 清洗（去页眉页脚/水印/目录）
  → 分块 → 向量化 → 索引入库（含元数据）
```

## 为什么需要

- 企业文档形态复杂：PDF / Word / PPT / 表格 / 公式 / 扫描件 / 图片；默认加载器（如 PyPDFLoader）只能抽正文文本——**表格错乱、公式丢失、扫描件空白**，问答质量必然崩塌；
- 分块策略直接决定检索精度：chunk 太大噪声多、太小语义割裂；
- 存储策略决定在线检索的速度与精度：向量库、ANN 索引、混合存储的选型直接影响 Recall 与延迟。

## 复杂文档处理（重点）

### 复杂文档的类型

多栏排版 PDF、扫描件（需 OCR）、表格、公式、图片、代码片段、多页报告、合同。它们需要「**视觉感知**」——理解布局、跨栏阅读顺序、表格结构，而非简单抽字。

### 解析工具选型（热门工具）

| 工具 | 出品/定位 | 强项 | 注意点 |
|------|----------|------|--------|
| **MinerU** | 上海 AI Lab，VLM 加持 | 公式/表格/多栏识别业界领先 | 需 GPU、计算重 |
| **LlamaParse** | LlamaIndex 官方 | 深度集成 LlamaIndex 生态 | 云端 API、需 Key |
| **Docling** | IBM Research | 完全离线、格式最全、表格强 | 企业本地首选 |
| **Unstructured** | RAG 专用预处理 | 50+ 格式、语义分块领先 | 云端 API、速度慢 |
| **PyMuPDF（pymupdf4llm）** | Artifex | 极速 PDF→Markdown、零依赖 | 仅 PDF、无 OCR |
| **PaddleOCR** | 中文 OCR | 扫描件/图片文字识别 | 需与解析配合 |
| **Marker / DoclingAI / DeepDoc** | 公式 / 表格 / 中文专项 | 按场景定向攻坚 | 各有侧重、代价不同 |

**分层选型思路**：通用格式用 Unstructured → Word 用 MarkItDown、简单 PDF 用 PyMuPDF → 含公式/图表/扫描件用 MinerU / Marker → 表格用 DoclingAI → 中文场景考虑 DeepDoc / PaddleOCR。也可按「文件类型 + 复杂度检测」动态路由到最合适的解析器。

### 复杂内容的工程化处理

- **表格**：结构化提取为 Markdown 表格（保留行列），必要时再让 LLM 生成摘要；
- **公式**：识别为 LaTeX 表示，保留可检索的语义；
- **图片 / 扫描件**：OCR 出文字（或 AI 生成图片描述）参与检索；**三重索引范式**——「检索文本 = 原文 + AI 图片描述，LLM 输入 = 纯原文避免污染，渲染 = 图片 URI」；
- **多栏布局**：按阅读顺序还原文本流，避免跨栏错乱。

### 分块策略（与检索精度强相关）

- **固定长度切分**：按字符/token 硬切，简单但易把语义割裂；
- **语义切分**：按句子语义相似度合并，保证单块语义完整；
- **标题层级分块**：按章节/段落/列表天然结构切割，适配结构化文档；
- **父子分层分块**：小 chunk 精确定位 + 父 chunk 补全上下文——兼顾「精准」与「完整」（本项目采用）；
- 重叠窗口 15%-20%，过滤页眉页脚 / 目录 / 水印 / 重复备注。

## 热门存储策略

### 向量库选型

| 方案 | 定位 | 适用 |
|------|------|------|
| **Qdrant** | 高性能专用向量库（Rust） | 中大规模、混合检索友好（本项目） |
| **Milvus** | 分布式向量库 | 超大规模、高并发生产 |
| **FAISS** | Meta 开源库，嵌入式 | 原型/中小规模（千万内） |
| **pgvector** | PostgreSQL 插件 | 已用 PG、数据中等、不想引新组件 |
| **Chroma / Weaviate / Pinecone** | 轻量 / 自托管 / 托管 | 按部署偏好与规模选择 |
| **ES dense_vector / AnalyticDB** | 检索/数据库生态融合 | 已有 ES 技术栈或需向量+结构化融合 |

**选型三要素**：数据规模、延迟要求、现有技术栈。粗略阈值：百万级用 FAISS / pgvector；千万到亿级用 Milvus / Qdrant；重延迟场景选 HNSW 索引。

### ANN 索引类型（决定检索速度与召回）

| 索引 | 原理 | 特点 |
|------|------|------|
| **Flat（暴力）** | 逐条计算距离 | 100% 准确，仅万级内或作评测基线 |
| **IVF（聚类）** | 先聚类再在最近簇检索 | nlist / nprobe 权衡召回与速度 |
| **HNSW（图）** | 层级导航图 | 在线低延迟首选；M / efConstruction / efSearch 调参，内存占用高 |
| **PQ（乘积量化）** | 向量切段压缩 | 显存/内存降一个数量级，召回略损，适合亿级 |

### 混合存储架构

- **向量库 + 倒排索引（BM25）**：向量管语义、BM25 管关键词，在线阶段融合（详见 [online-hybrid-retrieval.md](online-hybrid-retrieval.md)）；
- **对象存储 + 向量库**：原文/图片存对象存储（OSS/S3），向量库存索引与引用，本地不落盘；
- **关系库 + 向量库**：结构化元数据（权限/部门/时间）存关系库，先过滤再检索；
- **图谱 + 向量**：实体关系建图，双通道召回（见 [graph-rag.md](graph-rag.md)）；
- **元数据策略**：文档级/chunk 级元数据（来源、作者、时间、权限）支撑在线过滤与答案溯源。

### 更新与维护

- 去重：本项目已实现**归一化文本 SHA256 精确去重 + bottom-k sketch 近似去重**（Jaccard ≥ 0.85 判重，保留 mtime 最新版，旧版标 `superseded` 可追溯）；
- 版本化 / 增量入库（内容 hash 台账，只重算变更文档）为规划项，触发条件与方案见《后续规划》；
- 文档变更触发重解析、重向量化，保证知识库不过期。

## 本项目的做法

前置处理管线已落地于 `app/rag/preprocess/`（入口 `scripts/ingest_documents.py --input data/docs`），单文档四态状态（`ok` / `superseded` / `quarantined` / `dlq`），任何单档异常不阻塞整批：

```
扫描（按 mtime 旧→新）→ sniff 格式识别（magic bytes + 加密/损坏前置拦截）
  → complexity 复杂度路由（扫描页占比 > 50% 或图片 → OCR；否则快路径）
  → 解析：md/html/txt、docx（标题层级+表格扁平化）、文本 PDF（块坐标排序）、扫描件 OCR（qwen3.5-flash 多模态，200 DPI 逐页渲染）
  → 清洗五阶段：归一化（NFKC/零宽/断行合并）→ 页眉页脚移除（跨页重复度 > 60%）→ 乱码拦截（� > 3% / mojibake > 5%）→ 质量评分（≥70 入库 / 50-69 隔离 / <50 DLQ）
  → 跨文档去重（SHA256 精确 + bottom-k 近似）
  → 报告 data/ingest/report.json + DLQ 归档 data/ingest/dlq/ → RagManager.ingest_all() 各方案幂等入库
```

- **解析层**：sniffer 以字节头嗅探优先于扩展名（防 `.txt` 伪装 PDF）；复杂文档（扫描件/图片）走 qwen3.5-flash 多模态 OCR，DashScope 非 200 转可操作中文报错并重试；
- **清洗层**：normalizer / boilerplate / garble / quality / dedup 五模块独立纯函数，阈值均为模块级常量可调；
- **分块**：父子分层切分——小 chunk 精确定位 + 父 chunk 补全上下文；
- **存储**：Qdrant 向量库（默认，向量索引）+ BM25 倒排（关键词），混合检索在在线阶段融合；后端经「存储后端」抽象可切换到 Elasticsearch（`dense_vector` kNN + `text` BM25），详见 [advanced-rag.md](advanced-rag.md) §3——内置语料规模小，切换后端几乎无感知，属扩展性储备；
- **元数据**：保留来源信息，支撑答案可溯源到原文。

**配套文档**（长期维护，改代码必须同步更新）：

- [复杂情况应对手册](../app/rag/docs/RAG建库文档处理-复杂情况应对手册.md)——13 类复杂情况逐条应对 + 阈值速查表；
- [后续规划](../app/rag/docs/RAG建库文档处理-后续规划.md)——压缩包递归、五级复杂度评分、多栏版面分析、增量更新等 9 项扩展的挂点与验收标准。

## 收益与边界

**收益**

- 解析质量直接放大下游效果，是性价比最高的优化点；
- 分层切分兼顾「找得准」与「读得全」；
- 混合存储支撑语义 + 关键词双路召回，在线检索下限高。

**边界 / 局限**

- 重型解析工具（VLM / OCR）成本高、速度慢，需按文档复杂度分级处理；
- 存储选型无万能解，按规模 / 延迟 / 技术栈权衡；
- 解析是需持续迭代的独立子系统，需离线质量评估闭环（解析正确率、分块召回）支撑。

## 参考链接

- **解析工具**：MinerU、LlamaParse、Docling、Unstructured、PyMuPDF（官方文档与仓库）
- **向量库**：Qdrant、Milvus、FAISS、pgvector（官方文档）
- **文档解析与工具对比**：[2026 顶尖文档解析工具对比（MinerU/LlamaParse/Docling/Unstructured/PyMuPDF）](https://blog.51cto.com/u_16213686/14592429)
- **向量库选型**：[向量数据库选型指南：RAG 场景下的性能对比与调优策略](https://cloud.tencent.cn/developer/article/2726511)
