# Agentic RAG 架构设计（企业级）

> 对应实现：`app/rag/agentic/` 包（state / tools / roles / orchestrator）+ `app/rag/schemes/agentic.py`（薄适配层）。
> 编排器为 **LangGraph StateGraph 原生编排**（与 `agents/modes` 的 plan_execute / reflection 同款惯例）。
> 姊妹篇：`Modular_RAG_企业级架构详解.md`（modular 方案静态编排）。

## 1. 核心思想

按企业级 Agent 系统标准设计：**角色分离 + 工具注册表 + 预算治理 + 双闭环**，编排与 modular 完全独立。

- **角色分离**：决策拆为 5 个 LLM 角色（Router / Planner / Grader / Corrector / Verifier），各自独立场景、独立规则回退、独立熔断；检索执行不设 LLM 角色——决策在 Planner/Corrector，执行治理在 ToolRegistry（决策/执行分离）；
- **工具注册表**：库内检索能力封装为 4 个工具，注册表白名单、单工具限次、跨波去重、非法卷降级、并行波次五类护栏统一治理；
- **预算治理**：步数 / 纠错轮数 / token / 墙钟超时 / 单工具上限 / 角色熔断，全部在编排层统一施加（角色/工具层不管预算）；
- **双闭环**：
  - 证据评审闭环（**CRAG**）：Grader 逐条证据相关性评分 + 缺口归纳 → 不足则 Corrector 决策纠错波（改写/换卷/换工具）→ 回检索；
  - 答案校验闭环（**Self-RAG**）：Verifier 按事实-证据支持度矩阵判定可答/缺口，预算耗尽仍不足 → 如实上报 clarify。

## 2. 分层结构

```
app/rag/agentic/
├── state.py          # AgentState（共享状态/预算记账）+ TraceEvent（逐事件可观测轨迹）
├── tools.py          # ToolRegistry（工具治理 + 并行波次）+ cross_turn_seed / diversify
├── roles.py          # 五角色：Router/Planner/Grader/Corrector/Verifier（LLM JSON + 规则回退）
├── orchestrator.py   # AgenticOrchestrator：LangGraph StateGraph 编排（6 节点 + 4 条件边）+ 预算治理 + 证据管线

app/rag/schemes/agentic.py   # 薄适配层：RagScheme 接口 ↔ 编排器（指代消解/跨轮 seed 前置）
```

方案层继承 `AdvancedRagScheme` **仅取底层原语**（入库分块 / 重排器 / 父块回填），编排、检索执行、纠错、校验全部自有——静态执行计划被多角色状态机取代，不再受 modular 模块链约束。

## 3. Agentic RAG vs Modular RAG 对照

| 维度 | Modular RAG | Agentic RAG（本方案） |
|---|---|---|
| 控制流 | 静态：语义路由一次定型 → 执行计划模块链 | LangGraph StateGraph 动态状态机：ROUTE → PLAN → RETRIEVE(并行) → GRADE ⇄ CORRECT → VERIFY |
| 决策粒度 | 编译期一次定型全部模块组合 | 运行期分角色决策（路由/规划/评审/纠错/校验各自独立） |
| 证据质量 | 命中即上下文（压缩兜底） | CRAG 逐条评审：无关剔除 + 缺口归纳驱动纠错 |
| 纠错能力 | 有界升级 1 轮（多路→定向→多跳硬编码链） | 预算内纠错回环（默认最多 2 轮），按缺口定向补检索 |
| 答案校验 | answerability 闸门（事后判定） | Self-RAG 支持度矩阵（verify 角色）+ 闸门 |
| 失败处理 | 组件异常上抛/跳过 | 角色 LLM 失败 → 规则回退；连续 2 次失败 → 熔断锁定规则回退 |
| 成本治理 | 固定模块链成本 | 步数/纠错轮数/token/超时/单工具上限多维预算，超限优雅降级 |
| 可观测性 | 行为事件（rewrite/classify/retrieve…） | 逐事件轨迹：角色/思想/动作/参数/命中/时延/token/护栏备注 |
| 前置/后处理 | 指代消解、跨轮 seed、重排、压缩、父块回填 | 同款底层原语（入库分块/重排/压缩/父块回填），编排侧独立组装 |

### 能解决 modular 的什么问题

1. **路由误判即整路白跑**：定向降为 Planner 的提示，Grader 看到证据偏差由 Corrector 当场换卷/换表述/换工具；
2. **升级阶梯只能走一遍**：纠错回环预算内可多轮（默认 2），且每轮针对 missing_facts 定向补检索；
3. **证据质量不可控**：CRAG 评审在进入上下文前剔除无关块、归纳缺口；Self-RAG 校验保证生成侧拿到的是「事实有证据支持」的结论。

### 代价与边界

- 每查询多次角色 LLM 调用，延迟与成本高于 modular（qwen3.5-flash 低参数 + 预算治理控制）；
- LLM 决策抖动靠四层兜底：JSON 解析失败→规则回退、护栏拦截、token/步数预算降级、answerability 闸门兜底 clarify；
- 工具范围**仅限库内检索**（不做写操作、不调外部 API）。

## 4. 角色协议（roles.py）

| 角色 | 职责 | 输出（JSON） | 规则回退 |
|---|---|---|---|
| Router（route） | 边界判断：要不要检索 / 生成策略 / 定向卷提示 | `retrieval_need / generation_mode / target` | 保守检索 + citation |
| Planner（plan） | 事实清单 + 首发多路检索计划 | `facts[] / calls[]` | 单事实 + hybrid |
| Grader（grade，CRAG） | 逐条证据相关性 + 缺口归纳 | `relevant[]（范围内下标）/ missing_facts[]` | 词法共现（2 字词 ≥2 或 score ≥0.5，宽松防误杀） |
| Corrector（correct，CRAG） | 纠错波工具调用（按缺口定向） | `calls[]` | 每缺失事实一路：未用过的定向卷优先，否则 hybrid |
| Verifier（verify，Self-RAG） | 事实-证据支持度矩阵 → 可答性 | `answerable / missing_facts[]` | 词法覆盖判定 |

- 每角色独立 LLM 场景（见 §8）；LLM 输出不可解析/调用异常 → 规则回退并在轨迹记 `note`（熔断计数依据）；
- **熔断**：同一角色连续 2 次决策失败（`_FAIL_STREAK_LIMIT=2`）→ 该角色锁定规则回退，不再消耗 LLM 调用；
- 越界防护：Planner 动作白名单过滤（空计划回退默认 hybrid）、Grader 下标越界剔除、Router 非法 generation_mode 归一为 citation。

## 5. 工具协议（tools.py）

| action | 工具 | 参数 | 说明 |
|---|---|---|---|
| `search` | 纯向量检索 | `query` | 语义相似/同义改写 |
| `hybrid` | 混合检索 | `query` | 语义+关键词（含规模规范词扩展） |
| `volume_search` | 定向卷内检索 | `query`, `volume` | 卷名须在目录内，非法卷名降级全库检索 |
| `multi_hop` | 多跳规划-执行-验证 | `query` | 复用 PlanExecuteRetriever，call_cap 独立收紧为 1 |

ToolRegistry 五类护栏：

1. **白名单拦截**：注册表外动作（如外部 API）直接拦截不执行；
2. **单工具限次**：超 `call_cap` 拦截（note 说明，不执行，仍计入预算计数）；
3. **跨波去重**：同 `(action, query, volume)` 已正常执行过 → 后续波次拦截（同波不去重，允许首发多路撞车）；
4. **非法卷降级**：`volume_search` 卷名不在目录 → 降级全库检索（保留意图）；
5. **并行波次**：一波多路调用 ThreadPoolExecutor 并行执行（`parallel` 上限）。

辅助能力：`diversify`（定向卷结果多样性截断，融合前防同模板块挤占 top 结果）、`cross_turn_seed`（跨轮 seed 闸门：低分/无共现丢弃、限量 5 条）。

## 6. 预算治理（编排层统一施加）

| 预算 | 配置项（config.py） | 默认 | 超限行为 |
|---|---|---|---|
| 步数 | `rag_agent_max_steps` | 8 | 强制结束检索循环（含被护栏拦截的调用） |
| 纠错回环 | `rag_agent_correction_rounds` | 2 | 停止纠错，交 Verifier 终判 |
| 墙钟超时 | `rag_agent_timeout_s` | 90s | 不再发起新的 LLM 决策与工具波次 |
| token 预算 | `rag_agent_token_budget` | 8000 | 后续角色全部规则回退（0=不限） |
| 单工具上限 | `rag_agent_tool_call_cap` | 3 | 该工具后续调用被拦截 |
| 并行度 | `rag_agent_parallel` | 4 | 波内排队执行 |

熔断为第 7 项治理：角色连续 2 次失败 → 锁定规则回退。全部预算由 `manager.py` 从 settings 透传给方案构造器。

预算在 StateGraph 中的落点：`_use_llm`（token 预算 / 墙钟超时 / 熔断）在每个角色节点内检查；
步数 / 纠错轮数 / 预算耗尽在 `after_verify` / `after_correct` / `after_retrieve` 条件边统一判定，
超限经条件边短路到 END（不再进入下一波）——语义与原手写 while 循环完全等价，仅调度载体改为图。

## 7. 状态机与证据管线（LangGraph StateGraph）

编排器编译为一个 LangGraph StateGraph（6 异步节点 + 4 条件边，与 `agents/modes` 同款原生编排；
`while` 手写循环由循环边取代，`run()` 与 `astream()` 共用同一编译图）：

```
指代消解(deictic) → 跨轮 seed 闸门(cross_turn_seed)
    │
    ▼
START → route ──(检索无关)──→ END（寒暄直接生成）
            └─→ plan → retrieve ──(预算/超时短路)──→ END
                           └─→ grade → verify ──(可答/预算耗尽)──→ END
                                          └─→ correct ──(无可用调用)──→ END
                                                    └─→ retrieve（循环回波，每波一轮纠错）
    每波 retrieve → grade 走证据管线：
      RRF 融合（定向卷先 diversify、seed 额外一路）→ 重排 → GRADE（CRAG，无关剔除+缺口归纳）
      → 缺口且纠错预算有余 → CORRECT → 回 RETRIEVE
      VERIFY（Self-RAG 支持度矩阵）：可答 → DONE；预算耗尽仍不足 → CLARIFY（如实上报）
    │
    ▼
压缩 → 父块回填 → 包装 RetrieveResult（含 trace）
```

- 决策与执行分离：节点只做角色调度（决策在 Planner/Corrector，执行在 ToolRegistry）；
- 共享状态挂 `_GraphState.agent`（`AgentState` 就地变更），跨节点通道（`calls`/`fused`/`verdict`/`outbox`…）承载每波数据；
- 阻塞调用统一 `asyncio.to_thread`（不阻塞事件循环，SSE 逐事件刷出）；同步 `run()` 走 `ainvoke`（无运行中 loop 用 `asyncio.run`，异步上下文内调用转交独立线程执行）。

## 8. LLM 场景与配置

五角色独立场景（`llm/service.py`，全部 qwen3.5-flash、`enable_thinking=False`）：

| 场景 | temperature | max_tokens |
|---|---|---|
| `rag_agent_route` | 0.1 | 250 |
| `rag_agent_plan` | 0.2 | 500 |
| `rag_agent_grade` | 0.1 | 500 |
| `rag_agent_correct` | 0.2 | 400 |
| `rag_agent_verify` | 0.1 | 400 |

接入：`manager.py` `_SCHEME_REGISTRY["agentic"]`；seed 复用：`agents/runner.py` 对 `("modular", "agentic")` 同样传递 `seed_hits`；建库：`scripts/ingest_modular.py` `build_corpus(["modular", "agentic"])`（独立集合 `{prefix}_agentic`）。

## 9. 轨迹与事件协议

### trace（RetrieveResult.trace / retrieve 事件 trace 字段）

```json
{
  "total_events": 7,
  "tool_calls": {"hybrid": 2, "volume_search": 1},
  "total_tool_exec": 3,
  "corrections": 1,
  "role_llm_calls": {"route": 1, "plan": 1, "grade": 2, "correct": 1, "verify": 1},
  "tokens": {"prompt": 1200, "completion": 300},
  "steps": [
    {"seq": 1, "role": "router", "thought": "需要检索", "action": "route", "params": {},
     "hits": 0, "latency_ms": 120.5, "tokens": {"prompt": 100, "completion": 20}, "note": ""}
  ]
}
```

每步 TraceEvent 携带角色/思想/动作/参数/命中数/时延/token/护栏备注——支撑前端按角色渲染时间线与评测侧成本核算。

### SSE 事件

复用既有事件语义（`rewrite` / `seed_reuse` / `classify` / `retrieve` / `compress` / `answerability`），新增：

| 事件 | 时机 | 载荷要点 |
|---|---|---|
| `plan` | Planner 产出事实清单与首发计划 | `facts[]` |
| `agent_step` | 每个工具步完成 | `step: {seq, role, action, params, hits, note…}` |
| `grade` | CRAG 评审完成 | 保留数/缺口 |
| `correct` | 纠错决策产出 | 纠错波调用 |
| `verify` | Self-RAG 校验完成 | answerable/缺口 |

流式下发：各节点把事件写入 `outbox` 通道（`Annotated[list, operator.add]` 累积），
`astream()` 按 `graph.astream(stream_mode="values")` 逐 super-step 排空 yield——
协议与事件序列不变，仅调度层由手写循环改为 LangGraph；收尾的 `retrieve` / `compress` / `answerability`
仍由终态计算后下发。

## 10. 评测与诊断

- 全量评测：`scripts/eval_real_full.py --scheme agentic`（报告 `eval/reports/real_full_agentic.json`）；
  agent 轨迹聚合：`avg_events` / `avg_tool_exec` / `correction_rate`（纠错率）/
  `correction_success_rate`（纠错有效性：发生纠错的用例中 Self-RAG 闸门终判可答的比例）/
  `avg_tokens`（角色 LLM token 均值）/ `tool_calls` / `role_llm_calls`（各角色调用分布）；
  评测期五角色场景温度归零（轨迹可复现，线上默认温度不受影响）；
- 逐用例轨迹：`real_full.py` 每条记录含 `agent.total_events/tool_exec/tools/corrections/role_llm_calls/tokens`；
- 挑战题对比：`scripts/diag_agentic_compare.py`（默认 r21/r24/r25）——对照 modular vs agentic 的卷分布、evidence 覆盖与逐步轨迹（`#seq [role/action]`）；
- 单测：`tests/test_agentic_rag.py`（42 用例，FakeChatModel 脚本化角色决策 + 规则回退 + 护栏/预算/熔断断言 + astream 事件序列 + manager 透传 + LLM 场景配置，全程离线）。

## 11. 已知取舍

1. **决策温度线上 0.1~0.2 / 评测 0**：线上允许少量探索性，评测归零保证角色调用序列与轨迹可复现；
2. **继承 AdvancedRagScheme 仅取底层原语**：分块入库/重排/父块回填与 modular 共享基础设施；编排与检索执行完全独立，不受 modular 执行计划约束；
3. **token 预算超限 → 规则回退而非硬失败**：可用性优先，降级后仍产出确定性结果并在轨迹记 note；
4. **检索执行不设 LLM 角色**：决策（Planner/Corrector）与执行（ToolRegistry）分离，执行层零 LLM 成本；
5. **trace 只在 agentic 有**：modular 结果 `trace=None`，评测报告不含 agent 段——口径上不污染对比。
