---
id: rag-eval
name: RAG 离线评测体系
shortDesc: 四层评测：L1 确定性回归（路由/检索/闸门）→ L2 语义评分 → L3 RAGAS 生成质量 → 真实链路评测，指标含上下文精确度与拒答判定。
icon: chart-bar
difficulty: adv
tags: [Evaluation, RAG-Eval, LLM-as-Judge, CI-Gate]
techFilters: []
accent: '#ec4899'
experience: false
prompts:
  - 怎么量化衡量一次 RAG 改动是好是坏？
  - 路由 / 检索 / 生成分别该用什么指标、怎么判？
  - RAG 评测为什么要分确定性层和语义层？
---
## 概述

RAG 评测的核心问题不是"模型强不强"，而是"我的 RAG 链路（路由 → 检索 → 闸门 → 生成）在业务语料上做得对不对"。因此指标必须分层：**行为层（路由 / 检索 / 闸门）全确定性、可进 CI 门禁；语义层（答案质量）用 LLM 评分、衡量真实质量**。语义评分使用 RAGAS 标准指标（RAG 链路的事实源是检索上下文，RAGAS 正是为此设计）；Agent 评测则用自定义 judge 逐指标判卷（见 [agent-eval](agent-eval.md)），两套评测**不共享评分引擎，仅共享方法论**（分层金标 / LLM-as-Judge / 确定性-语义分界 / 失败样本回流）。

```
    路由/检索/闸门               生成与答案
  ├─ L1 确定性回归    确定性断言    ├─ L2 语义评分    LLM
  │   路由准确率/Recall/Precision │   正确性/相关性/忠实度
  │   MRR/关键词覆盖/可答澄清率   │   Context Precision/拒答
  └────────────────────           └─ L3 RAGAS 标准指标
                                    └─ 真实链路评测（modular/agentic）
```

## 为什么这样设计

1. **评测单元是整条链路，不是单次输出**——答案对但检索捞错了分块、路由漏检索、闸门对资料不足照答不误，这类隐藏失败只有评链路才抓得住；
2. **确定性层与语义层必须分界**——检索 / 路由用注入期望决策 + 规则模块（FakeEmbeddings）离线可复现、零 Key，能当 CI 回归门禁；语义层有模型波动、成本高，只做质量报告；
3. **符号层比对不用 LLM**——路由四维决策逐维全等、命中块集合与金标相关块集合比对、关键词子串匹配，全部确定性算法；
4. **拒答是 RAG 特有维度**——库外问题必须"拒绝回答而非编造"，RAGAS 标准指标无此维度，需独立 judge 判定。

## 一、确定性层指标（离线可复现）

| 指标 | 判定方法 | 实现 |
|---|---|---|
| 路由准确率 | 路由四维决策（是否检索 / 检索模式 / 复杂度 / 生成模式）与金标期望逐维全对才算命中 | 注入期望决策跑确定性模块链（`--real-router` 时用真实 LLM 路由另算） |
| Recall 召回率 | \|命中 ∩ 相关分块\| ÷ \|相关分块\|（漏没漏） | 检索 top-k 与金标 `relevant` 集合比对 |
| Precision 精确率 | \|命中 ∩ 相关分块\| ÷ 返回条数（脏不脏） | 同上 |
| MRR | 首个相关分块的 1/位置，无则 0，按用例平均 | 同上 |
| 关键词覆盖 | 任一金标答案关键词出现在命中块文本中 | 子串匹配 |
| 可答率 / 澄清率 | 检索上下文是否足以作答 / 资料不足时是否主动澄清而非硬答 | 充分性闸门（answerability）输出比对 |

> 例（evaluate 路由）：金标 `{"retrieval_need": true, "retrieval_mode": "hybrid", "complexity": "multihop", "generation_mode": "citation"}`，实际决策四维全等才算该用例路由命中，任一维偏斜即失分——防"路由退化成总是检索"。

## 二、语义层指标（LLM 评分）

| 指标 | 通用评测方法 | 事实源 / 实现 |
|---|---|---|
| Faithfulness / Groundedness 忠实度 | **主张级验证**：答案拆成原子断言，逐个判定能否从检索上下文推断；分数 = 被支持断言 ÷ 总断言。防幻觉核心指标 | 检索命中块（RAGAS） |
| Answer Relevance 相关性 | 由答案反向生成问题，与原始提问算嵌入相似度，取平均 | RAGAS |
| Correctness 正确性 | 对照金标 reference：事实语义一致度 + 金标要点覆盖度，双维加权 | `eval_set.jsonl#reference` + RAGAS |
| Context Precision 上下文精确度 | 检索块逐位置检查对金标的支撑，按位置加权——越靠前且相关得分越高 | RAGAS |
| Abstention / Refusal 拒答 | 库外问题必须"拒绝回答而非编造"，独立判定 grounded | 手写 judge（`rag_judge` 场景），库外用例不进 RAGAS 评分 |

> 语义层只在有答案的用例上评分：库外用例期望答案是拒答，标准指标无意义，跳过；生成用 chat 场景、拒答 judge 用 rag_judge 场景、RAGAS 内部 LLM 用 rag_ragas 场景（关闭思考：高频小 JSON 提取，开思考会整批超时）。

## 三、真实链路评测（真实后端 + 真实 LLM）

离线评测注入期望路由、用规则模块隔离 LLM；真实链路评测则打通真实基础设施，回答"线上那套真的能跑且质量达标吗"：

- `eval_rag_real.py --scheme modular`：真实 Qdrant/ES 检索（混合检索 + Rerank + 父块回填）+ 真实 LLM 生成 + RAGAS 评分；
- `eval_rag_real.py --scheme agentic`：智能体式 RAG 全链路，附加 agent 轨迹指标（检索工具选择 / 预算消耗 / 评审纠错轮数）；
- `--fake` 冒烟：真实检索 + 占位生成（验证链路，分数无意义）。

## 四、运行方式（backend/ 目录）

```bash
python scripts/eval_rag_l1.py              # L1 确定性回归：路由/检索/闸门，无需 Key，可挂 CI
python scripts/eval_rag.py                 # L2+L3：LLM 生成 + RAGAS 评分 + 拒答判定（需 Key）
python scripts/eval_rag.py --fake          # 离线冒烟：验证链路可跑通（分数无评测意义）
python scripts/eval_rag_real.py --scheme modular   # 真实链路评测（真实检索 + 真实 LLM）
python scripts/eval_rag_real.py --scheme agentic   # 智能体式 RAG 真实链路
```

## 五、与 Agent 评测的分工

方法不同、事实源不同：RAG 语义评分用 RAGAS 标准指标，事实源是**检索上下文**（答案是否依据检索到的文档说话）；Agent 答案评分用自定义 `agent_judge` 场景逐指标判卷（不复用 RAGAS），Faithfulness 事实源是**工具执行证据**（答案是否依据工具返回值说话）。RAG 独有 Context Precision（检索质量）与拒答判定；Agent 独有工具选择 / 轨迹 / 架构不变量（见 [agent-eval](agent-eval.md)）。线上失败样本回流评测集见 [rag-online-eval](rag-online-eval.md)。

## 关联卡片

agent-eval（Agent 评测体系）、rag-online-eval（线上闭环与失败回流）、modular-rag（被评对象）、agentic-rag（被评对象）、observability-eval（可观测性与评估）。
