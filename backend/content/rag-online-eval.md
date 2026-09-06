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

在 L1 确定性回归 / L2 语义评分 / L3 RAGAS 三层离线评测之上，补齐线上真实流量的评估闭环，回答"改任何模块 / prompt / 模型后，线上是不是变好了"。闭环已落地：真实对话自动采样 → 前端点赞 / 点踩 → 反馈回填 → 定期脚本回流评测集。

## 为什么需要它

现状线上只跑不评：query / 检索 / 路由 / 答案 / 耗时未落库，无点赞点踩、无指标、失败样本不回流——"改没改好"没有数据。离线评测集是**人工设计的金标**，覆盖不到真实用户怎么问；线上样本能补上这个盲区，且失败样本回流后可以防止同类问题再上线。

## 核心思想

四段闭环：在线采集 → 用户反馈 → 定期回流评测 → 失败样本回流；影子评分（无用户反馈时自动评分）与告警列为后续增强。

```
真实对话 → 采样落库（按日 JSONL）
              │  runner 收口异步写，不阻塞 SSE
              ▼
         前端 👍/👎 + 原因 ──POST /api/feedback──→ 回填样本行 vote/reason
              │
              ▼
     scripts/eval_online.py 定期回流
     ├─ 回归池（未点踩样本，金标沿用线上行为）
     │   ├─ eval/online_eval_set.jsonl        RAG 域（检索命中）
     │   └─ eval/online_agent_eval_set.jsonl  Agent 域（工具轨迹）
     └─ 修复池（点踩样本，金标置空待人工修正）
         └─ eval/online_fix_set.jsonl → 人工修正后移入回归池 → 参与离线回归门禁
```

## 本项目的做法（已落地）

### 1. 在线采集（采样落库）

- **采集点**：runner 在流式收口（`finally` 块）后异步追加样本行（`asyncio.to_thread`），不阻塞 SSE 返回；
- **落盘形态**：`eval/samples/online_YYYYMMDD.jsonl` 按日分文件，长期保留（与 telemetry 运行记录的 TTL 7 天互补）；
- **采样条件**：仅当本轮存在检索命中（`retrieved_ids` 非空）或有工具调用才落样本——寒暄 / 纯问答跳过，避免噪音；
- **样本字段**：`sample_id / session_id / client_key / ts / version / query / effective_query / capability（rag|agent）/ tool_sequence / mode / rag_scheme / complexity / retrieval_mode / generation_mode / insufficient / retrieved_ids / answer / elapsed_ms / status / stats / vote / reason`；
- **失败静默**：任何采集异常吞掉，绝不影响主对话链路；
- **总开关**：`EVAL_ONLINE_ENABLED`（默认 true）。

### 2. 用户反馈（点赞 / 点踩）

- **展示时机**：前端 FeedbackBar 仅在本轮结束、RAG 启用且存在检索命中时出现，与后端"无命中不落样本"对齐；
- **交互**：👍 有帮助 / 👎 没帮助，差评可补填原因（可选，最多 200 字）；
- **回填**：`POST /api/feedback` 按 `session_id + query` 匹配当日样本行，回填 `vote / reason`；
- **幂等**：对已回填行仅更新 reason，vote 保持首次选择，避免同一轮被反复覆盖；
- **异常兜底**：未匹配到样本行（样本清理 / 本轮未采集 / 参数篡改）记入 `orphan_feedback.jsonl` 供排查，不报错；反馈不计入每日对话配额。

### 3. 定期回流评测（scripts/eval_online.py）

读取全部（或 `--date` 指定日期）在线样本，按能力域分流筛选：

| 池 | 条件 | 去向 | 金标处理 |
|---|---|---|---|
| **回归池 · RAG** | capability=rag（有检索命中）且未点踩 | `online_eval_set.jsonl` | `expected` 沿用线上真实语义路由决策（retrieval_need / retrieval_mode / complexity / generation_mode）；`relevant` 以真实检索 top-k 命中为基础 |
| **回归池 · Agent** | capability=agent（无命中但有工具调用）且未点踩 | `online_agent_eval_set.jsonl` | `mode` 沿用实际架构模式 + `must_call` 沿用线上真实工具轨迹，作行为回归 |
| **修复池** | 点踩（vote=down）样本 | `online_fix_set.jsonl` | **金标一律置空不沿用线上行为**（点踩样本的线上行为可能是错的，原样固化会把坏行为写进评测集）；`observed` 记录观测行为（轨迹 / 命中 / 答案）供人工参照修正 |

**筛选规则**：

1. **关键分支全量**：RAG 域 `multihop / decompose / out_of_kb` 三类关键分支全量回流（在线样本量小，宁全不漏）；
2. **兜底抽样**：其余按 `--ratio` 随机抽样（默认 0.3），`--limit` 封顶防止评测集无限膨胀（超出优先保留关键分支）；
3. **同 query 去重**：优先保留点踩样本（同一问题被多人问时负面信号优先级最高，去重后点踩仍移交修复池）。

**常见用法**（backend/ 目录下）：

```bash
python scripts/eval_online.py                  # 全量样本按默认比例回流两域回归池 + 修复池
python scripts/eval_online.py --date 20260906  # 只回流指定日期的样本
python scripts/eval_online.py --ratio 0.2      # 兜底随机抽样比例（默认 0.3）
python scripts/eval_online.py --limit 50       # 每域回归池用例上限（超出优先保留关键分支）
python scripts/eval_online.py --dry-run        # 只打印筛选统计，不写评测集
```

### 4. 失败样本回流后的闭环

- **回归池用例**：`answer_keywords / reference` 留空待人工复核补全（产出 `origin` 溯源到样本行），补全后即参与 `eval_rag_l1.py` / `eval_agent_l1.py` 离线回归门禁；
- **修复池用例**：需人工修正 `expected / must_call / reference`（置 `label=ready`）后移入对应回归池再参与门禁——这是"线上反馈驱动金标修正"的关键一环。

### 版本溯源

样本与评测报告都携带被测系统版本（`get_app_version()` 取 git 最新提交短哈希，非 git 环境回退 `dev`），支撑 **PR-over-PR** 分数对比：同一评测集跑两个 commit，分数差异即该 PR 带来的真实变化，可回溯到代码版本。

## 收益与边界

- **收益**：改动能量化回归（跑回流集对比分数）；真实失败样本自动沉淀为回归用例；点踩样本反向驱动金标修正——质量可追踪、可回滚；
- **边界**：采样成本（当前按命中/工具调用轻量采样）；隐私脱敏（样本含真实 query / 答案，企业落地需脱敏）；金标可信度依赖"未点踩即默认正确"的假设（故点踩样本单独进修复池人工复核）。

## 演进与关联

- **演进**：影子评分（无人工反馈时用 LLM 自动评分补样本标签）、告警（回流集分数劣化自动告警）、Prompt 版本化（把提示词当代码做版本管理）；
- **关联**：observability-eval（可观测性与评估，运行记录是样本的完整版）、agent-eval（离线评测体系，回流集的消费方）、与 Harness 可观测性、CI 回归门禁互补。
