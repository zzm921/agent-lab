# RAG 建库文档处理 — 复杂情况应对手册

> 适用范围：`app/rag/preprocess/` 建库前置处理管线。
> 定位：**长期维护文档**——阈值、规则、解析器变更时必须同步更新本手册。
> 配套阅读：《企业级 RAG 离线流水线技术方案》（设计依据）、《RAG建库文档处理-后续规划》（扩展路线）。

---

## 一、处理流程总览

```
data/docs/ 扫描（旧→新按 mtime 排序）
  → [sniff]        格式识别：magic bytes 优先 + 扩展名兜底；加密/损坏 PDF 前置拦截
  → [complexity]   复杂度路由：扫描页占比 > 50% 或图片 → OCR；否则快路径
  → [parse]        解析：md/html/txt、docx（标题层级+表格）、文本 PDF（块坐标排序）、扫描件 OCR
  → [clean 阶段1]  归一化：NFKC / 零宽与控制字符 / 空白折叠 / 断行合并
  → [clean 阶段2]  页眉页脚：跨页重复度 > 60% 移除 + 纯页码模式
  → [clean 阶段3]  乱码拦截：� 占比 > 3% / mojibake 特征 > 5% / 无有效文字 → DLQ
  → [clean 阶段5]  质量评分（0-100）：≥70 入库 / 50-69 隔离 / <50 DLQ
  → [clean 阶段4]  跨文档去重：SHA256 精确 + bottom-k 近似（Jaccard ≥ 0.85），保留 mtime 最新
  → [report]       data/ingest/report.json + DLQ 归档 data/ingest/dlq/
  → [ingest]       干净文本 → RagManager.ingest_all() → 各方案集合幂等入库
```

单文档任何阶段异常均捕获为 `DocReport(status=dlq)`，**不阻塞整批**。

### 状态四态

| 状态 | 含义 | 后续处理 |
|------|------|---------|
| `ok` | 清洗通过 | 进入入库候选 |
| `superseded` | 与其他文档内容重复（保留最新版） | 不入库，文件保留可追溯 |
| `quarantined` | 质量分 50-69 | 不入主索引，文件保留 |
| `dlq` | 失败（乱码/损坏/OCR 失败/极低质） | 复制到 `data/ingest/dlq/` + `<文件名>.error.txt` 说明 |

---

## 二、复杂情况逐条应对（13 类）

每节固定结构：**现象/根因 → 判定 → 解决 → 结果 → 示例**。

### 1. 编码错误（GBK 被 UTF-8 误解码）

- **现象**：`æ–‡ä»¶ç®¡ç†` 式 mojibake，或大量 `�`；BM25 全miss，向量成为随机噪声。
- **判定**：解析前 `charset-normalizer` 检测编码（text_parser.decode）；清洗兜底（garble.py）：`�` 占比 > **3%**，或 mojibake 特征字符占比 > **5%**。
- **解决**：按检测编码正确解码；检测失败但乱码率超标 → 拒绝入库。
- **结果**：正确解码 → ok；仍乱码 → dlq（reason 注明「编码错误」）。
- **示例**：`tests/fixtures/preprocess/garbled.txt` → dlq；`gbk_legacy.txt`（e2e 现场生成）→ 正确解码 ok。

### 2. 扩展名伪装（`.txt` 实为 PDF）

- **判定**：sniffer.py 字节头嗅探（`%PDF-` / `PK\x03\x04` / `\xff\xd8\xff` / `\x89PNG\r\n\x1a\n`）优先于扩展名；扩展名仅用于文本类细分（md/html/txt）。
- **解决**：按真实 MIME 路由解析器。
- **结果**：真实 PDF 走 PDF 管线；伪装文件若损坏 → dlq。
- **示例**：`fake_pdf.txt`（%PDF 开头）被识别为 PDF → 解析失败（损坏）→ dlq，而非读出二进制乱码入库。

### 3. 扫描型 PDF / 图片（无文本层）

- **判定**：complexity.py 逐页统计可提取字符，`< 50 字/页` 记为扫描页；扫描页占比 > **50%** 或全文 < 50 字 → 路由 `ocr`；图片文件直接 `ocr`。
- **解决**：OCR 路径（ocr_parser.py）——PDF 每页渲染 200 DPI PNG → `app/llm/multimodal.ocr_image()` 调 qwen3.5-flash（MultiModalConversation 端点）识别；单页失败记 warning 跳过，不弃整档。
- **结果**：识别文本进清洗层；全部页 OCR 失败 → dlq。
- **示例**：e2e `scan.pdf`（空白页）路由 ocr → mock OCR 文本入库。

### 4. 加密 / 损坏文件

- **判定**：sniffer.check_pdf_openable——`fitz.open` 异常判损坏；`doc.needs_pass` 判加密。
- **解决**：解析前拦截，错误信息中文可操作（「PDF 已加密（需要密码），请先解密后再入库」）。
- **结果**：dlq，不进解析管道。
- **示例**：e2e 生成 AES-256 加密 PDF → dlq。

### 5. 页眉页脚 / 页码 / 水印

- **现象**：每页重复噪声使 BM25 虚高（「内部机密」重复 50 次）、向量被拉偏。
- **判定**：boilerplate.py——分页文档（≥3 页）取每页首/尾各 2 行（先剔除纯页码行），统计跨页出现占比，> **60%** 判为页眉页脚；纯页码模式（`第X页/共X页`、`- X -`、`Page X of Y`、`X / Y`）无条件移除。
- **解决**：全文档匹配移除。**不做数字归一化**——否则正文条款行「第X条…」会被误判为同型噪声。
- **结果**：噪声行删除；**只出现一次的签名/审批行不误删**。
- **示例**：e2e `policy.pdf` 清洗后不含「内部机密」与「共 3 页」。

### 6. 断行 / 断段（词被硬换行切断）

- **现象**：PDF 硬换行把「规章制度」切成「规章/制度」，语义匹配失效。
- **判定/解决**：normalizer.py——行尾无句末标点（`。！？；：.!?;:`）且次行非结构起始（`#` 标题 / `|` 表格 / 列表 / `（一）` / `1、`）→ 合并；上一行为标题/表格行不吸收次行。
- **结果**：段落完整、词不被切断。
- **示例**：「员工严重违反规章\n制度的，可以解除合同。」→ 合并为一行。

### 7. 表格结构丢失

- **解决**：docx_parser——按 body 顺序遍历（段落与表格交错不乱序），表格转 Markdown 表头行 + 每行「列名1: 值1 | 列名2: 值2」扁平化（提高关键词召回）；PDF 表格按文本块输出。
- **结果**：表格以行文本形式入库，检索「张三的薪资」能命中对应行。
- **示例**：e2e `meeting.docx` 表格 → 「会议室: A101 | 容量: 12 人」。

### 8. 精确 / 近似重复文档（多版本并存）

- **现象**：v1/v2/最终版并存，LLM 同时看到新旧限额，回答自相矛盾。
- **判定**：dedup.py——精确：归一化文本 SHA256；近似：5-gram shingle bottom-k（k=128）Jaccard 估计 ≥ **0.85**。
- **解决**：文件按 mtime 旧→新处理，**保留最新版本**，旧版标 `superseded`（不删除，合规可追溯）；链式覆盖（v1→v2→v3 最终都指向 v3）。
- **结果**：入库列表只含每个内容族最新版。
- **示例**：`dup_v1.md`（旧）superseded、`dup_v2.md`（新）ok。

### 9. 碎片 / 低质量文档（日志片段）

- **现象**：时间戳+日志级别行，无完整句；混入主索引污染通用问答。
- **判定**：quality.py 质量分 = 0.35×体量(500字封顶) + 0.30×完整句占比 + 0.20×句长适中性(10-100字) + 0.15×(1-噪声行占比)，时间戳/日志行计为噪声。
- **解决**：score < **70** 不入主索引；< **50** 进 DLQ。不物理删除——隔离/淘汰文档在专门场景（排障日志检索）仍有价值。
- **示例**：`log_fragment.txt` → dlq（quality）。

### 10. 全空文档

- **判定**：garble.py——strip 后为空。
- **结果**：dlq（「提取文本为空」）。空结果比错误结果安全。

### 11. OCR 调用失败 / 服务异常

- **判定**：multimodal.py——DashScope 非 200（含 url error=模型端点不匹配、401=Key 无效），转 **actionable 中文报错**。
- **解决**：重试 1 次；仍失败抛 `OcrError` → 单文档 dlq，批次继续。
- **约束**：多模态模型必须走 `MultiModalConversation.call()`（项目统一 qwen3.5-flash）；同步调用不得进入事件循环（在线化需 `asyncio.to_thread()`）。

### 12. 单文档异常拖垮整批

- **解决**：pipeline.py 逐文档 try/except——预期异常（DocumentRejected/GarbledDocument/OcrError）取中文 reason；未知异常记录堆栈日志 + 「处理异常」入 DLQ；**批次继续**。
- **示例**：e2e 混合 10 份好坏文档，坏的进 DLQ，好的全部入库。

### 13. 超长 PDF 内存压力

- **解决**：pdf_parser.py 逐页提取、按块（bbox y,x 排序）流式产出元素，不整档驻留；ocr_parser 同样逐页渲染。
- **边界**：当前实现的元素列表仍整体存在 ParsedDocument 中；更大规模（万页级）需演进为 JSONL 落盘流式合并（见《后续规划》）。

---

## 三、阈值速查表（改代码必改此表）

| 阈值 | 值 | 代码落点 |
|------|-----|---------|
| 扫描页字符密度 | < 50 字/页 | complexity.SCANNED_PAGE_CHARS |
| 扫描页占比路由 OCR | > 0.5 | complexity.SCANNED_RATIO |
| 替换符 � 占比 | > 0.03 拦截 | garble.REPLACEMENT_RATIO |
| mojibake 特征占比 | > 0.05 拦截 | garble.MOJIBAKE_RATIO |
| 页眉页脚跨页占比 | > 0.6 移除 | boilerplate._REPEAT_RATIO |
| 每页首尾采样行数 | 各 2 行 | boilerplate._EDGE_LINES |
| 近似去重 Jaccard | ≥ 0.85 | dedup.DUPLICATE_THRESHOLD |
| bottom-k sketch 大小 | 128 | dedup.SKETCH_SIZE |
| shingle 长度 | 5 字符 | dedup.SHINGLE_SIZE |
| 质量分入库线 | ≥ 70 | quality.SCORE_PASS |
| 质量分隔离线 | 50-69 | quality.SCORE_QUARANTINE |
| OCR 重试次数 | 1 | multimodal._OCR_RETRY |
| OCR 渲染 DPI | 200 | ocr_parser._RENDER_ZOOM |

---

## 四、如何新增一类应对（维护指引）

1. 在对应清洗阶段模块或新 stage 中实现判定与处理逻辑；
2. 阈值定义为模块级常量并加入本手册第三节速查表；
3. 为该情况补一个 fixture + 测试用例（`tests/fixtures/preprocess/` 或测试内生成）；
4. 在本手册第二节追加一节（结构保持：现象/判定/解决/结果/示例）；
5. 若涉及新解析格式或新路由，同步更新《后续规划》中的挂点说明。
