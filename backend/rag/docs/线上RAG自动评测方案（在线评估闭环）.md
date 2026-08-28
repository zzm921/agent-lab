# 线上 RAG 自动评测方案（在线评估闭环）

> 目标读者：开发者 / 架构评审。
> 定位：在现有「三层离线评测 + 回归门禁」之上，补齐**线上真实流量的自动评测闭环**，让生产质量从"黑盒"变为"可监控、可回归、可回流"。
> 状态：方案评审稿（未实施）。

---

## 1. 背景与问题

### 1.1 现状盘点

当前项目已有完整的**离线**评测体系：

| 层 | 入口 | 数据 | 检索 | 是否需 Key | 用途 |
|---|---|---|---|---|---|
| L1 确定性回归 | `scripts/eval_modular.py` | 手造语料 `eval/corpus.py` + 26 条 `eval_set.jsonl` | 确定性 BM25 + 注入期望路由 | 否 | CI 回归门禁（`pytest tests/test_eval_regression.py`） |
| L2 手写 judge | `scripts/eval_semantic.py` | 同上 | 同上 | `LLM_API_KEY` | 忠实度/相关性/库外不编造率 |
| L3 RAGAS | `scripts/eval_ragas.py` | 同上 | 同上 | +`EMBEDDING_API_KEY` | 标准库交叉校准 |

### 1.2 缺口

线上真实链路（`POST /api/stream` → `AgentRunner.stream`）**只跑不评**：

- 不采样：query、检索命中、路由决策、答案、耗时，全部不落库；
- 不反馈：前端无点赞/点踩，用户满意度不可见；
- 不回归：真实向量检索 + 真实 LLM 路由的质量，没有任何指标监控；
- 不回流：线上失败样本永远不会进入评测集。

结论：**改任何模块/prompt/模型后，"线上是不是变好了"没有数据回答。**

### 1.3 目标

建立四段闭环：**在线采集 → 用户反馈 → 定期回流评测 → 失败样本回流**，使真实流量持续反哺质量监控。

### 1.4 非目标（本期不做，仅设计）

- 影子评分 / 告警（列为 P2，本文只给设计）；
- A/B 分流、多租户隔离、私有化数据脱敏合规平台化。

---

## 2. 总体架构

```
                     ┌────────────────────────────────────────────────┐
                     │               线上链路（改动最小）                 │
                     │  POST /api/stream → AgentRunner.stream          │
                     │   └─ 采集钩子（不阻塞流式返回）                    │
                     └──────────────────────────┬─────────────────────┘
                                                │ 写 JSONL（每轮一行）
                                                ▼
                                     eval/samples/online_*.jsonl
                                                │
                ┌───────────────────────────────┼──────────────────────┐
                ▼                               ▼                       ▼
      ┌─────────────────┐            ┌────────────────────┐   ┌────────────────────┐
      │ 反馈 API（P0）     │            │ 回流+回归（P1）        │   │ 影子评分（P2，设计）   │
      │ POST /api/feedback│            │ scripts/eval_online.py│   │ 异步 judge 打分      │
      │ 前端 👍/👎 写回    │            │ → online_eval_set     │   │ → faithfulness 漂移   │
      └────────┬─────────┘            │ → online_latest.json  │   │ → 阈值告警           │
               │ 补 vote              │ → cron/CI 定时         │   └────────────────────┘
               ▼                       └────────────────────┘
      eval/samples/online_*.jsonl
```

关键设计决策：**离线评测（回归门禁）与在线评测（质量监控）保持两套口径、互相补充**，见 §5。

---

## 3. P0：在线采集 + 用户反馈

### 3.1 在线采集钩子

**位置**：[app/agents/runner.py](file:///c:/Users/ASUS/Desktop/workspace/my-agent/backend/app/agents/runner.py) 的 `stream()`（约 L195-L271）已经持有本轮全部可采集量，**无需侵入各 RAG 方案**：

| 采集字段 | 来源（runner.stream 内） |
|---|---|
| `session_id` / `mode` / `rag_scheme` / `rag_enabled` | L217 meta |
| `query`（原始问题） | `message` 入参 |
| `effective_message`（指代消解后实际检索 query） | L241-L248 |
| `generation_mode`（direct/citation/comparison） | L243-L244 |
| `insufficient`（检索不足标记） | L254-L255 |
| `retrieved_ids` + `retrieved_texts` | `rag_context["hits"]` L249-L250 |
| `answer` | `_run_graph` 结束后从 `graph.aget_state()` 取末条 AIMessage（L358） |
| `elapsed_ms` | 本轮计时 |

**设计要点**：

1. **不阻塞流式**：采集钩子放在 `stream()` 末尾（拿到 answer 之后），用 `asyncio.to_thread` 或独立后台任务写文件，SSE 已结束、不影响首字节延迟。
2. **结构化 JSONL 落盘**：与 `eval/reports/*.json` 风格一致，按日分文件：

   ```
   eval/samples/online_20260828.jsonl
   {"sample_id": "...", "session_id": "...", "ts": "...", "query": "...", "effective_query": "...",
    "mode": "react", "rag_scheme": "modular", "generation_mode": "citation",
    "insufficient": false, "retrieved_ids": ["c01"], "answer": "...", "elapsed_ms": 812.3,
    "vote": null}
   ```

3. **开关**：`settings.eval_online_enabled` 控制（默认开启，与项目"后端能力默认就绪"约定一致），未配 LLM Key 或未检索（`rag_context is None`）时跳过。

### 3.2 用户反馈 API

**后端**：`POST /api/feedback`

- [app/schemas.py](file:///c:/Users/ASUS/Desktop/workspace/my-agent/backend/app/schemas.py) 新增 `FeedbackRequest`：
  - `session_id`（必填，定位会话）
  - `query`（必填，定位样本行；同一会话可能多轮）
  - `vote`: `"up" | "down"`（必填）
  - `reason`（可选，自由文本）
- [app/api/chat.py](file:///c:/Users/ASUS/Desktop/workspace/my-agent/backend/app/api/chat.py) 新增 `feedback()` 路由：按 `session_id + query` 精确匹配样本行，回填 `vote`/`reason` 字段。不匹配则记录到 `eval/samples/orphan_feedback.jsonl` 供排查。
- 注意与现有配额/限流解耦：反馈不计入每日对话配额。

**前端**：[frontend/src/components/ChatPanel.vue](file:///c:/Users/ASUS/Desktop/workspace/my-agent/frontend/src/components/ChatPanel.vue) 在 RAG 回答（带引用来源 `[1]`/`[2]`）下方渲染 👍/👎 两个按钮：

- 点选后调 `POST /api/feedback`，已选态置灰；
- 仅当本轮存在 RAG 检索（meta 里 `rag_enabled=true` 且命中非空）才展示，避免普通对话刷屏。

### 3.3 P0 验收标准

1. 真实对话一次后，`eval/samples/online_*.jsonl` 出现该轮结构化样本（含 query/命中/answer/elapsed）；
2. 前端对该回答点赞/点踩，样本行 `vote` 被回填；
3. 无检索（寒暄）与未配 Key 场景不落样本；
4. 所有既有测试不回归（`pytest tests/`）。

---

## 4. P1：样本回流 + 真实检索回归

### 4.1 回流脚本 `scripts/eval_online.py`

读取 `eval/samples/online_*.jsonl`，筛选需要回流为评测用例的样本：

| 筛选规则 | 目的 |
|---|---|
| `vote == "down"` | 用户明确不满意 → 必回流 |
| judge 语义分低（见 §6，P1 阶段可直接复用 L2 judge 对 answer 离线打分，**不阻塞线上**） | 模型自评低分 |
| 关键分支抽样（multihop / decompose / out_of_kb） | 覆盖复杂路径 |
| 兜底：按比例随机抽样 | 保量，防止全回流导致评测集膨胀 |

产出一条 `online_eval_set.jsonl`（**独立文件，不污染离线 26 条**），用例格式复用 `eval_set.jsonl` 结构：

```json
{"id": "on_0001", "branch": "simple", "query": "<线上真实问题>",
 "expected": {"retrieval_need": true, "retrieval_mode": "hybrid",
              "complexity": "simple", "generation_mode": "citation"},
 "relevant": ["<真实命中 chunk id>"], "answer_keywords": ["<人工确认关键词>"]}
```

**金标处理**（关键，见 §5）：
- `relevant`：以真实检索 top-k 命中为基础，抽样人工确认（低投入）；
- `retrieval_mode / complexity / generation_mode`：直接沿用线上真实路由决策（已在样本里）；
- `out_of_kb`：judge 判定 grounded=false 且检索确实不相关时人工打标。

### 4.2 真实检索回归报告

新增 `eval/online_runner.py`，与 `eval/runner.py` 的区别：

| | 离线 runner | 在线 runner |
|---|---|---|
| 检索 | 确定性 BM25 + 注入期望路由 | 生产 `RagManager` 真实向量/混合检索 + 真实 LLM 路由 |
| 语料 | 手造 42 条 | 生产 `KNOWLEDGE_CORPUS`（云帆制度全文） |
| 指标 | Recall/Precision/MRR（有金标） | 路由准确率 / keyword_coverage / answerable / judge 语义分 / 耗时 |

产出 `eval/reports/online_latest.json`（meta 里标注 `retrieval: production vector + real LLM router`，与离线报告明确区分）。

### 4.3 触发方式

- **手动**：`python scripts/eval_online.py`（评审/发布前跑）；
- **定时**：CI scheduled / cron，例如每日凌晨跑并对比昨日报告，指标下滑超过阈值即失败（复用 `test_eval_regression.py` 的断言模式，阈值按线上基线另定）。

---

## 5. 数据口径与金标问题的处理

这是本方案最需要讲清楚的设计取舍：

1. **线上没有金标**：Recall/Precision/MRR 依赖"已知正确 chunk"，真实流量没有。因此在线评测**不以 Recall 为核心指标**，而以 `keyword_coverage / answerable / judge 语义分 / 路由准确率 / 用户反馈` 为核心。
2. **线上检索不确定**：真实向量检索结果每轮可能不同（而离线是确定性 BM25）。因此在线报告只能看**趋势/分布**，不能当回归门禁的硬断言——回归门禁仍由离线 L1 承担。
3. **两套口径互相补充，不可直接对比**：
   - 离线 = **机制回归门禁**（改代码前跑，防退化，确定性强）；
   - 在线 = **真实质量监控**（发布后看，反映生产真实表现，噪声大）。
4. **失败样本回流形成闭环**：点踩/低分样本 → `online_eval_set.jsonl` → 下次回归覆盖 → 修正后指标回升，验证修复有效。

---

## 6. P2（设计）：影子评分与告警

- **影子评分**：线上返回答案后，异步用 `rag_judge` 场景 LLM 对 `(query, answer, retrieved_contexts)` 打分（faithfulness/answer_relevance，复用 [eval/semantic.py](file:///c:/Users/ASUS/Desktop/workspace/my-agent/backend/eval/semantic.py) 的 `_judge_prompt`），分数写回样本行。
- **漂移告警**：按周聚合平均 faithfulness/answer_relevance，与上周均值比较，下滑超阈值（建议 ±0.3，按 0~5 分制）触发告警日志 + 报告 `drift` 标记。
- **成本控制**：影子 judge 有 LLM 调用费，按比例采样（默认 30%）可配；`eval_shadow_enabled` 开关，默认关闭。

---

## 7. 配置项（`app/config.py` 新增）

| 配置 | 默认 | 说明 |
|---|---|---|
| `eval_online_enabled` | `true` | 在线采样总开关 |
| `eval_sample_dir` | `eval/samples` | 样本落盘目录 |
| `eval_online_set_path` | `eval/online_eval_set.jsonl` | 回流评测集路径 |
| `eval_shadow_enabled` | `false` | 影子评分开关（P2） |
| `eval_shadow_sample_rate` | `0.3` | 影子评分采样比例 |
| `eval_drift_threshold` | `0.3` | faithfulness 周环比告警阈值 |

同步更新 `backend/.env.example`。

---

## 8. 风险与取舍

| 风险 | 应对 |
|---|---|
| 影子 judge 成本 | 采样比例可配，默认关（P2） |
| 真实 query 可能含隐私 | 采集字段最小化；文档声明按部署环境评估脱敏 |
| 线上/离线口径不同导致误解 | 报告 meta 明确标注检索口径；本文 §5 已界定 |
| 样本文件无限增长 | 按日分文件 + 保留策略（如只留 30 天 / 只回流不保留原文） |
| 反馈按钮误点/滥用 | 每个会话+query 只能投一次；可选并入配额体系 |

---

## 9. 里程碑

| 阶段 | 交付物 | 验收 |
|---|---|---|
| P0 | 采集钩子 + 反馈 API + 前端按钮 + 配置 | §3.3 四项验收通过 |
| P1 | `eval_online.py` + `online_runner.py` + 定时 | 线上样本回流出报告，CI 定时跑不崩 |
| P2 | 影子评分 + 漂移告警 | 周报含 drift 标记，超阈值有日志/报告 |

---

## 10. 相关代码索引

| 模块 | 位置 |
|---|---|
| 线上入口（采集挂载点） | `backend/app/api/chat.py` → `AgentRunner.stream` |
| 采样数据源 | `backend/app/agents/runner.py`（`rag_context`/`generation_mode`/`insufficient`/answer） |
| 离线评测底座（在线 runner 参考） | `backend/eval/runner.py` / `semantic.py` / `ragas_eval.py` |
| 评测集格式 | `backend/eval/eval_set.jsonl` |
| judge prompt（影子评分复用） | `backend/eval/semantic.py::_judge_prompt` |
| 配置 | `backend/app/config.py` |
| 前端反馈按钮位置 | `frontend/src/components/ChatPanel.vue` |
