# Agent Lab — AI Agent 技术实验场

基于 **LangChain + LangGraph + MCP** 的 AI Agent 技术实验场：真实调用大模型，实验室可按技术路径点选切换体验，支持多种推理模式、RAG 多方案检索、函数调用、审批与容错，所有过程经 SSE 实时流式可视化。

> 前端 Vue 3 + TypeScript + Vite + Tailwind；后端 FastAPI + LangGraph；SSE 流式输出。

## 项目目的

Agent Lab 是一个**「可讲解、可演示、可对比、可实验」**的 AI Agent 技术平台，目标是系统呈现 Agent 从理论到工程的关键技术栈：

- **讲解**：落地页以「6 大标签 × 21 张能力卡」组织 AI Agent 知识体系（提示词工程 / 上下文工程 / RAG / Agent / Harness / 协议），每张卡有图文正文与核心代码片段；
- **演示**：进入实验室，推理模式、RAG 方案、提示词策略、审批策略、工具能力等**真实运行**，并通过 SSE 事件流把思考、工具调用、检索、审批过程逐步展示出来；
- **对比**：同一任务可在不同推理模式 / 提示词策略 / RAG 方案下并排运行，直观对比差异；
- **实验**：技术方案一键点选切换、工具开关即时增删、故障注入（13 种类型）验证容错、示例一键填入，边改边看。

一句话：**把 AI Agent 的关键技术做成看得懂、跑得起来、能对比、能折腾的实验室。**

## 功能清单

按技术领域分组；每个领域先说明其技术点，再列出具体能力；完成度为**实际工程完成度**，未到 100% 的均列出明确缺口。

### Agent 策略（推理模式）

> 技术说明：Agent 的核心是「思考 - 行动」循环范式。本项目统一基于 LangChain `create_agent`（模型 ⇄ 工具循环）与 LangGraph `StateGraph`（显式构图）构建，用自定义 `AgentMiddleware` 收敛各模式差异（事件流 / HITL / 反思 / 多代理），统一接入轮数上限、审批与故障注入。

- **react**（90%）— 相关技术：`create_agent` + `AgentMiddleware`（思考-行动-观察循环）
  - 未完成：多轮上下文无限增长（无记忆压缩 / 截断）；无单轮输出长度（max_tokens）上限；超轮数终止依赖抛异常兜底而非图内优雅分支
- **plan_execute**（90%）— 相关技术：LangGraph `StateGraph`（planner → executor ⇄ tools → replanner 条件边，replan 上限 = max_iterations/2）
  - 未完成：步骤严格串行、无并行执行；计划只存内存检查点、重启即失（不落盘）；replan 次数硬编码推导、不上报、不可独立调优；计划解析仅按行去符号，无结构化 schema
- **reflection**（90%）— 相关技术：`ReflectionMiddleware`（generator ⇄ tools → critic 条件循环，PASS / FAIL 判定）
  - 未完成：评审是「流式文本 + 字符串匹配判定」，无结构化 LLM-as-a-Judge 评分器；修订是否真正改进无量化验证，可能空转到 max_iter；反思轮数与 max_iterations 耦合、无独立配置；不保留多稿对比
- **multi_agent**（雏形，未正式完成）— 相关技术：Orchestrator（create_agent）+ compute/analyze Worker（`convert_runnable_to_tool` 包装）
  - 未完成：worker 名单硬编码（compute / analyze）、无动态注册；编排者顺序调用 worker、无并行调度；结果仅靠 LLM 文本整合、无结构化聚合与冲突解决；worker 无独立状态（无 checkpointer）

### 函数调用 / 工具

> 技术说明：模型以结构化 JSON 发起工具调用。工具经 `bind_tools` 转 function schema 注入 `create_agent` 工具循环，`tools_builder` 按前端勾选组装，不可用能力置灰「不适配」。

- **calculator / time_now / web_search / run_command**（80%）— 相关技术：AST 白名单安全求值 / 本地时间 / DuckDuckGo HTML 抓取 / OpenSandbox 或本地子进程 + 危险命令拦截
  - 未完成：无并行工具调用（for 循环逐个执行）；无通用自定义工具注册接口（新增内置工具需同时改 builtin.py 与 registry.py）；web_search 强依赖 DuckDuckGo 页面结构（改版即失效）、结果数/超时硬编码；run_command 依赖用户自部署 OpenSandbox，local 兜底 `shell=True` 有注入面、黑名单为子串匹配可绕过、所有会话共享同一工作目录
- **结构化输出**（JSON Schema 约束）— 已在路由 / 规划等内部场景使用；独立能力模块 → **待实现**

### 提示词

> 技术说明：通过提示词模板控制模型行为。前端选择器切换策略，后端按策略拼装 System Prompt 并下发 `prompt_result` 事件。

- **prompt-strategy**（70%）— 相关技术：standard / few_shot / cot 三条 System Prompt 模板
  - 未完成：仅三条硬编码字符串，无自定义提示词 / 模板变量注入；策略只作用于首轮 system 前缀，后续轮次完全不受影响；few_shot 仅文本示例、cot 仅「请逐步思考」指令，无结构化解析与验证

### RAG

> 技术说明：从语料建库到在线检索的完整链路：文档解析 → 切块/分块 → 向量化入库 → 多路召回 → 融合/重排 → 压缩 → 生成。本项目落地四套方案（naive / advanced / modular / agentic）并在实验室点选对比；检索命中注入上下文，回答可引用来源。

- **naive RAG**（100%）— 相关技术：固定 500 字切块（100 重叠）+ 纯稠密向量检索
  - 定位说明：作为**对照基线**，刻意不做任何增强（无改写 / 重排 / 混合 / 压缩），用于对比展示缺陷
- **advanced RAG**（80%）— 相关技术：结构感知父子分块 / 语义分块 + 混合检索（稠密 + 本地 n-gram 稀疏，RRF 融合）+ Rerank 精排 + 父块回填
  - 未完成：稀疏检索是本地字符 n-gram 哈希向量（md5 → 2^16 桶），非真 BM25（真 BM25 仅在可选 ES 后端）；重排依赖外部 API（qwen3-rerank），无 Key 时回退「原分 + 字符二元组」的简单词法重排；无上下文压缩与充分性闸门；语义分块被结构分块旁路（真实语料走结构感知路径）；HyDE 无缓存
- **modular RAG**（90%）— 相关技术：语义路由（五维决策 D1/D3/D4/D5）→ 执行计划 → 动态编排模块（改写 / 指代消解 / 分解 / HyDE / 多跳规划-执行-验证 / 语义去重 / 压缩 / 充分性闸门），闸门前置 + 有界升级增量补缺
  - 未完成：D2 多知识库路由**明确不做**（单语料）；执行计划是确定性规则映射（if-else 枚举），非 LLM 生成 / 可插拔配置；compress 不覆盖 simple / multihop 路径；无模块级独立消融实验（现有评测为整链路 L1/L2/L3）；classifier 纯 LLM、无 Key 时 modular 不可路由
- **RAG 专项增强**（rag-variants）— HyDE（假想文档生成，融入稠密路召回）**已实现**；Self-RAG / CRAG / RAPTOR → **待实现**（Self-RAG / CRAG 已在智能体式 RAG 卡以角色编排形态实现，此卡为独立模块插件形态）
- **知识图谱 RAG**（graph-rag）— 相关技术：实体-关系建模 + 多跳推理 → **待实现**
- **智能体式 RAG**（agentic-rag，Self-RAG / CRAG / Adaptive-RAG，85%）— 相关技术：LangGraph 多角色状态机（路由 / 规划 / 评审 / 纠错 / 校验）+ 工具注册表（4 个库内检索工具 + 五类护栏）+ 预算治理（步数 / 纠错轮数 / token / 墙钟超时 / 单工具上限 / 角色熔断）+ 双闭环（CRAG 逐条证据评审 + Self-RAG 支持度校验）+ 逐事件 SSE 轨迹
  - 未完成：无多级缓存（查询 / Embedding / 检索结果三级缓存）；无规则路由 / 快速通道（纯 LLM 路由，规则仅在无 Key / 失败 / 超预算时降级兜底）；无监控告警（token / 时延 / 熔断等已记账但无看板与主动告警）；无检索级权限过滤与脱敏（库内结果按可信数据直进上下文）；多知识库路由（D2）明确不做（单语料）；工具仅限库内只读检索（无写操作 / 外部 API）
- **离线数据处理 / 建库**（offline-processing）— 语义分块 + 建库脚本 + 文本指纹幂等重建**已实现**；完整解析（OCR / PDF / docx / 表格图片 / 公式）与多语料增量挂载 → **待实现**
- **在线混合检索策略**（online-hybrid-retrieval）— RRF 融合（K=60）**已实现**；加权 RRF / 多方式融合对比、独立参数化实验 → **待实现**

### Harness 强化工程

> 技术说明：把 Agent 从「能跑」做到「可靠」：审批（人机协作）、容错（重试/熔断）、隔离（沙箱），保证真实生产环境的可控与鲁棒。

- **审批门（HITL）**（90%）— 相关技术：LangGraph `interrupt` 暂停 + `Command(resume=...)` 恢复；同 superstep 多 interrupt 合并批量审批
  - 未完成：策略仅 always / never 两档，无「仅危险操作」条件策略与 per-tool 独立策略；无审批超时自动拒绝（可无限期悬挂）；批量审批只能对所有工具统一决策，无法逐工具分别批/拒/改；无审批审计日志
- **容错·重试·熔断**（80%）— 相关技术：工具层透明重试（瞬时错误指数退避 + 抖动）+ Agent 层思考后重试；按「工具+参数签名」键的三态熔断（closed/open/half-open）+ 13 种故障注入
  - 未完成：熔断键含完整参数，换参即视为新键，本质「同参短路」而非工具整体熔断；无 QPS / 并发维度熔断；重试 / 退避参数仅全局、无 per-tool 覆盖；工具层重试会重复执行有副作用工具（无幂等）；熔断 / 重试状态仅内存、重启即失
- **沙箱**（70%）— 相关技术：OpenSandbox（Docker 服务端）或 local 子进程执行，危险命令黑名单 + 强制 HITL + 超时 / 输出截断
  - 未完成：无网络隔离（compose 注明需自补策略）；local 兜底后端在宿主 `shell=True` 执行、无文件系统 / 网络 / 资源隔离；黑名单为 20 条静态子串、无命令结构解析 / 白名单扩展；沙箱池全局持锁串行执行；`allowed_host_paths` 硬编码需人工对齐
- **可观测性与评估** — RAGAS 语义评测（L3）与语义回归（L2）**部分实现**（`backend/eval/` + `scripts/eval_*`）；Trace 全链路追踪、指标监控完整接入 → **待实现**
- **安全**（60%）— 相关技术：输入 Guardrail（规则拦截越狱/提示注入，命中短路 + 礼貌拒绝 + `guard_refused` 事件）；输出 Guardrail + 敏感数据脱敏（`StreamMasker` 流式实时脱敏手机号/身份证/银行卡/密钥 + 全文阻断提示，落库亦脱敏）；来源可信分级（web 搜索 / 命令输出 / 记忆召回 = 不可信外部来源，注入隔离指令与数据；知识库 = 受控内部语料 = 可信来源，不隔离）
  - 未完成：记忆投毒防御（跨轮长期记忆未落地）；服务端工具白名单强制；RBAC 权限管控；敏感操作审计日志

### 记忆

> 技术说明：让 Agent 具备跨会话的记忆能力：写入 → 语义召回 → 注入上下文；存储层支持多后端可替换。

- **多后端向量存储**（80%）— 相关技术：`StoreBackend` 统一接口，Qdrant（Prefetch + RRF 真混合）/ Elasticsearch（kNN + rank.rrf / 旧版 BM25）/ 内存三后端 + `MultiBackendStore` 多路融合，构造期自动选路、失败回退内存
  - 未完成：ES 混合检索需 8.8+ 且无版本门槛校验（8.0~8.7 会构造 RRF 失败）；选路 / 回退仅在启动构造期一次，运行期不探测不重连；真混合仅 Qdrant 后端、内存 / 多后端退化为多路融合；多线程共享 Embedding 客户端无并发上限
- **跨轮长期记忆**（雏形，未正式启动）— 相关技术：向量库写入 + 余弦语义召回 → **待实现**（存储后端与读写工具雏形已搭）

### 上下文工程

> 技术说明：管理 Agent 的上下文窗口：压缩、规划、缓存，控制成本与延迟。本领域当前为知识卡阶段。

- **上下文管理与压缩**（窗口规划 / 摘要压缩 / 滚动阈值）→ **待实现**
- **上下文缓存与渐进式披露**（Prompt Caching / JIT 按需加载）→ **待实现**

### 协议

> 技术说明：Agent 与外部世界交互的标准协议：MCP（工具服务化）与 A2A（智能体间通信）。

- **MCP 工具热插拔**（85%）— 相关技术：stdio / Streamable HTTP 双传输，启动自动连接 + `load_mcp_tools` 工具发现注册，连接失败标记「不适配」不注入，自带 mcp-notes 便签 server
  - 未完成：无断线重连 / 健康探测（连接仅在启动 / 首开建立一次，失败后不重试）；无 OAuth / token 刷新（仅 headers / env 直传）；工具 schema 无项目层校验；自带 notes server 单 JSON 文件、无用户隔离与条目上限、HTTP 形态无鉴权
- **A2A 智能体通信**（发现 / 委托 / 协作开放协议）→ **待实现**
- **计算机操作代理**（computer-use，看截图 / 移鼠标 / 点按钮）→ **待实现**

### 贯穿能力

> 技术说明：跨所有模式生效的横向能力：技术路径点选、流式输出、源码展示。

- **技术路径点选（能力热插拔）**（100%）— 相关技术：`POST /api/stream` 参数化动态组装 Agent 模式 / 提示词模板 / RAG 方案 / 工具集
- **SSE 流式输出**（85%）— 相关技术：`EventSourceResponse` + `asyncio.Queue` 顺序驱动 + 统一事件协议（thinking / message / tool_* / plan / retrieve / approval_request 等）
  - 未完成：无应用级心跳 / 保活事件（HITL 等待、慢检索期间连接无活性信号）；无断线重连 / 事件续传（断线需整轮重发）；事件无序号 / 无 schema 校验
- **真实源码展示**（100%）— 相关技术：`GET /api/source/{module}` 实时读取后端真实源码

## 总结

**已实现（含完成度）**：react 90%、plan_execute 90%、reflection 90%、函数调用（4 工具）80%、提示词策略 70%、RAG（naive 100% / advanced 80% / modular 90% / agentic 85% + HyDE）、HITL 90%、MCP 85%、容错·重试·熔断 80%、沙箱 70%、多后端向量存储 80%、安全防护 60%、SSE 85%、技术路径点选 100%、源码展示 100%。

**待实现**：
- 未正式启动（有雏形）：multi_agent、跨轮长期记忆
- 实现待落地：知识图谱 RAG、RAG 专项增强其余插件（RAPTOR 等）、上下文管理与压缩、上下文缓存与渐进式披露、计算机操作代理、A2A、可观测性完整接入、结构化输出独立模块
- 安全余量：记忆投毒防御、服务端工具白名单、RBAC 权限管控、敏感操作审计日志
- 增强项：离线建库完整解析（OCR / PDF / 表格 / 公式）、在线混合检索参数化实验、模块级消融评估

**明确暂不实现**（边界声明）：多知识库路由（D2）、结构化查询（Text-to-SQL）。

## 快速开始

### 1. 后端

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env                           # 配置 LLM_API_KEY（百炼，必填）等
uvicorn app.main:app --reload --port 8000
```

### 2. 前端（开发模式）

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### 3. 生产模式

```bash
cd frontend && npm run build
cd ../backend && uvicorn app.main:app --port 8000   # 直接访问 http://localhost:8000
```

## 环境变量（backend/.env）

| 变量 | 必填 | 说明 |
|---|---|---|
| `LLM_API_KEY` | 是 | 阿里云百炼（DashScope）API Key，对话模型 `qwen3.5-flash` |
| `LLM_BASE_URL` | 否 | 默认 `https://dashscope.aliyuncs.com/api/v1`（DashScope 原生 SDK） |
| `ENABLE_THINKING` | 否 | 默认 `true`：开启思考，返回 `reasoning_content`（思考）与 `content`（输出）两类结果 |
| `EMBEDDING_API_KEY` | 否 | RAG / 长期记忆需要（OpenAI 兼容接口，默认与百炼共用 Key） |
| `EMBEDDING_BASE_URL` | 否 | 默认 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL` | 否 | 默认 `text-embedding-v3` |
| `RAG_ENABLED` | 否 | 知识库检索总开关，默认 `true`（能力后端默认就绪）；`false` 整体关闭。每轮是否检索由前端「知识库检索」开关控制 |
| `RAG_MIN_SCORE` | 否 | 最小相关度阈值，默认 `0.6`；命中相似度低于该值直接丢弃（不注入上下文），全部被丢弃则本轮不注入。naive 为 cosine、advanced 为 rerank 归一分数 |
| `MCP_SERVERS` | 否 | JSON，声明 stdio 或 streamable HTTP 的 MCP Server（本项目自带 `mcp-notes` 便签 server） |
| `MCP_ENABLED` | 否 | 默认 `true`：服务启动时自动连接并发现已配置的 MCP Server（stdio 以子进程拉起 `mcp-notes`），无需手动启动 |
| `SECURITY_ENABLED` | 否 | 安全防护总开关，默认 `true` |
| `GUARD_INPUT` | 否 | 输入 Guardrail：越狱 / 提示注入特征拦截，命中短路并礼貌拒绝，默认 `true` |
| `GUARD_OUTPUT` | 否 | 输出 Guardrail：敏感数据泄露全文扫描 + 阻断提示，默认 `true` |
| `MASK_SENSITIVE_OUTPUT` | 否 | 输出敏感数据流式脱敏（手机号 / 身份证 / 银行卡 / 密钥），默认 `true` |
| `MARK_UNTRUSTED` | 否 | 不可信外部来源标记（web 搜索 / 命令输出 / 记忆召回与指令隔离，Prompt 注入防御），默认 `true` |

RAG 向量库数据在**线上前**通过建库脚本预建（在线服务启动时只加载、不现场入库）：

```bash
cd backend
python scripts/ingest_naive.py     # naive 方案（固定切块 + 纯稠密检索）
python scripts/ingest_advanced.py  # advanced 方案（语义分块 + 混合检索）
python scripts/ingest_modular.py   # modular / agentic 方案（语义分块，缺省同时建两库）
```

详见 [docs/deployment.md](docs/deployment.md)。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/capabilities` | 能力目录（内置 + MCP，含可用性/示例） |
| GET | `/api/mcp` | MCP 服务状态（enabled / servers / capabilities） |
| POST | `/api/mcp` | 页面点选开启/关闭 MCP 服务（连接并发现/移除 MCP 工具） |
| POST | `/api/stream` | SSE 流式对话（模式/能力/策略/审批策略/RAG 方案） |
| POST | `/api/approve` | HITL 审批（批准/拒绝/修改） |
| POST | `/api/stop` | 停止当前流式任务 |
| GET | `/api/rag/schemes` | RAG 方案目录（naive / advanced / modular / agentic） |
| GET | `/api/faults` | 当前注入的故障列表 |
| GET | `/api/faults/types` | 故障类型目录（瞬时 / 参数业务两类） |
| POST | `/api/fault` | 注入 / 清除工具故障（13 种类型，演示两层重试） |
| GET | `/api/source/{module}` | 后端真实源码（代码展示用） |
| GET | `/api/content` | 能力卡内容（后端解析 `backend/content/` 下的 md） |
| GET | `/api/sandbox/files` | 沙箱文件列表 |
| GET | `/api/sandbox/files/download` | 下载沙箱文件 |
| GET | `/api/health` | 健康检查 |

## 目录结构

```
my-agent/
├── backend/                     # FastAPI 后端（LangGraph + MCP + RAG + 记忆）
│   ├── app/
│   │   ├── main.py              # 应用入口：CORS / 路由 / 前端静态托管
│   │   ├── config.py            # 环境变量与配置（settings）
│   │   ├── schemas.py           # Pydantic 请求/响应模型
│   │   ├── agents/              # Agent 模式与编排
│   │   │   ├── modes/           #   react / plan_execute / reflection / multi_agent
│   │   │   ├── middleware/      #   事件流 / HITL / 多代理 中间件
│   │   │   ├── runner.py        #   SSE 驱动 + HITL 中断/恢复
│   │   │   ├── harness.py       #   护栏：审批策略 / 资源上限 / 止损 / 统计
│   │   │   ├── tools_builder.py #   按 enabled_capabilities 组装工具集
│   │   │   └── state.py
│   │   ├── capabilities/        # 能力目录：内置 + MCP 发现 + 可用性
│   │   │   ├── builtin.py
│   │   │   ├── mcp.py
│   │   │   └── registry.py
│   │   ├── core/                # 事件协议 / 错误
│   │   │   ├── events.py
│   │   │   └── errors.py
│   │   ├── security/            # 安全防护：输入/输出 Guardrail + 来源可信分级 + 脱敏
│   │   │   ├── patterns.py      #   越狱/注入拦截 + 敏感数据脱敏 + 泄露阻断 规则
│   │   │   ├── input_guard.py   #   输入 Guardrail（命中短路 + 礼貌拒绝）
│   │   │   ├── output_guard.py  #   输出 Guardrail（StreamMasker 流式脱敏 + 全文扫描）
│   │   │   └── wrap.py          #   不可信外部来源包装（提示注入防御）
│   │   ├── llm/                 # DashScope 适配（Chat + Embedding）+ Fake
│   │   │   ├── client.py
│   │   │   ├── dashscope_chat.py
│   │   │   ├── dashscope_embeddings.py
│   │   │   ├── fake_model.py
│   │   │   └── service.py
│   │   ├── tools/               # 工具：calculator / time_now / web_search / run_command / memory / retry
│   │   ├── mcp_server/          # 自带 mcp-notes 便签 server（stdio）
│   │   │   └── notes_server.py
│   │   ├── memory/              # 会话 + 长期记忆 + 多后端存储
│   │   │   ├── session_store.py
│   │   │   ├── vector_store.py
│   │   │   ├── corpus.py
│   │   │   └── stores/          #   Qdrant / Elasticsearch / 内存 可替换后端
│   │   ├── rag/                 # RAG 方案与算子
│   │   │   ├── schemes/         #   naive / advanced / modular / agentic 四套方案
│   │   │   ├── agentic/         #   Agentic RAG 编排（角色 / 工具注册表 / 状态机）
│   │   │   ├── routing/         #   语义路由 / 改写 / 分解 / 指代消解 / HyDE
│   │   │   ├── retrieval/       #   融合 / 重排 / 压缩 / 多跳 / 充分性闸门
│   │   │   ├── corpus/          #   云帆制度语料（知识库）
│   │   │   ├── docs/            #   RAG 设计文档
│   │   │   ├── base.py / manager.py / ingest.py
│   │   │   └── __init__.py
│   │   └── api/                 # chat / content / sandbox 路由
│   │       ├── chat.py
│   │       ├── content.py
│   │       └── sandbox.py
│   ├── content/                 # 21 张能力卡 Markdown（落地页 /api/content 实时解析）
│   ├── eval/                    # RAGAS 语义评测 / 离线回归 / 报告
│   │   ├── runner.py
│   │   ├── semantic.py
│   │   ├── ragas_eval.py
│   │   ├── eval_set.jsonl
│   │   └── reports/
│   ├── scripts/                 # 建库与评测脚本
│   │   ├── ingest_naive.py / ingest_advanced.py / ingest_modular.py
│   │   └── eval_modular.py / eval_ragas.py / eval_semantic.py
│   ├── tests/                   # pytest（Fake 模型 + mock MCP，不联网）
│   ├── .env.example
│   ├── pytest.ini
│   └── requirements.txt
├── frontend/                    # Vue 3 + TS + Tailwind 前端
│   ├── src/
│   │   ├── views/               #   Landing 落地页 + HomeView 实验室
│   │   ├── components/          #   能力卡 / 对话 / 控制 / 系统 组件
│   │   ├── composables/         #   useChatStream / useCapabilities / useContentData
│   │   ├── data/                #   能力卡静态元数据（capabilityData.ts）
│   │   ├── services/            #   SSE 客户端（sse.ts）
│   │   ├── router/              #   / 落地页 + /lab 实验室 路由
│   │   ├── types/ styles/       #   类型定义 + 全局样式
│   │   ├── App.vue / main.ts / vite-env.d.ts
│   │   └── vite.config.ts
│   ├── tests/                   # vitest
│   ├── index.html
│   ├── package.json
│   └── tailwind.config.js / postcss.config.js / tsconfig.json
├── docs/                        # 架构 / 部署 / 测试 / 知识体系 文档
│   ├── architecture.md
│   ├── deployment.md
│   ├── testing.md
│   ├── AI Agent 技术演变路径与实现方案.md
│   └── AI Agent 知识体系细化（6标签21卡片）.md
├── deploy/
│   └── opensandbox/             # OpenSandbox 沙箱部署编排
│       └── docker-compose.yml
├── .gitignore
└── README.md
```
