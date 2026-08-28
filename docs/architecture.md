# 架构说明（Architecture）

个人 AI Agent 技术展示平台，真实调用阿里云百炼（DashScope）Qwen 大模型，能力可热插拔，支持 MCP 集成。

## 1. 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontend（Vue 3 + TS + Vite + Tailwind）                         │
│  HomeView：能力池网格（点击开关=热插拔；示例按钮=一键填入输入框；      │
│            不可用卡片置灰标注「不适配」）+ 模式选择 + 对话区           │
│  ModuleView/CompareView：运行流式可视化 + 审批 + 代码 + 原理         │
│  useChatStream（SSE）+ StreamingText + ToolCallBadge              │
│  ApprovalDialog（HITL）+ ErrorBanner                              │
└───────────────────────────┬──────────────────────────────────────┘
        GET /api/capabilities   POST /api/stream   POST /api/approve
        GET /api/source/{m}     GET /api/health
┌───────────────────────────▼──────────────────────────────────────┐
│  Backend（FastAPI + Uvicorn）                                     │
│  capabilities/registry.py：能力目录（内置 + MCP 发现 + 可用性探测）  │
│    ├ capabilities/builtin.py：内置能力定义（含示例）                │
│    └ capabilities/mcp.py：MCP server 管理（stdio/HTTP）、工具发现、  │
│                           连接失败标记「不适配」                    │
│  agents/tools_builder.py：按 enabled_capabilities 组装工具集      │
│  agents/runner.py：SSE 驱动 + HITL 中断/恢复                        │
│  agents/harness.py：护栏层（审批策略/资源上限/止损/统计）           │
│  agents/modes/*.py：create_agent 构建四种模式                       │
│  agents/middleware/*：事件流/HITL/计划/反思/多代理 中间件           │
│  tools/（calculator/time_now/web_search/run_command/memory_tool）   │
│  memory/（session_store + vector_store + corpus）                 │
│  llm/dashscope_chat.py：DashScope 原生 SDK 适配（reason/output 分离 + 流式） │
│  llm/client.py（工厂：DashScope Chat + Embedding，可注入 Fake）           │
└──────────────────────────────────────────────────────────────────┘
```

## 2. 核心概念

### 2.1 能力池（Capabilities）与热插拔

- 每个能力是 `{id, name, desc, source, server?, availability, reason?, example, code_key}`。
- `GET /api/capabilities` 返回完整目录：内置能力均可用（不依赖外部 Key）；MCP 能力在启动后首次访问时发现。RAG/长期记忆已不再作为能力卡暴露（见下），RAG 走独立检索阶段 + 方案目录，memory 工具在 Embedding 可用时由 `tools_builder` 注入。
- **热插拔**：前端点击能力开关 → `POST /api/stream` 携带 `enabled_capabilities` → 后端 `tools_builder.build_tools()` 按 id 从注册表解析为 LangChain `StructuredTool` 注入模式 Agent；未启用或不可用的能力不注入。
- **示例一键填入**：能力卡片的「示例」按钮 → 前端启用该能力 + 把 `example` 填入输入框 → 直接发送体验。

内置能力：

| id | 说明 | 可用条件 | 示例 |
|---|---|---|---|
| `calculator` | 安全计算 | 始终 | 计算 (137×0.85−20)÷3 |
| `time_now` | 当前时间 | 始终 | 现在几点 |
| `web_search` | 网页搜索 | 始终（失败降级） | 搜索 Qwen3 发布时间 |
| `run_command` | 命令执行（沙箱） | 始终（执行前强制 HITL） | 在沙箱里查看当前目录有哪些文件 |

RAG（`rag`）与长期记忆（`memory`）**已移出内置能力目录**：RAG 重构为独立检索阶段（runner 前置检索 + 上下文注入），方案经 `GET /api/rag/schemes` 目录（naive / advanced / modular）在实验室点选；memory 工具仅在 Embedding 可用时经 `tools_builder` 注入，不作为能力卡展示。

### 2.2 MCP 集成

- 配置来源：`MCP_SERVERS` 环境变量（JSON 字符串）。
- 两种 transport：
  - **stdio**：`{"name": {"command": "npx", "args": ["-y", "pkg"], "env": {...}}}`
  - **streamable HTTP**：`{"name": {"url": "http://host/mcp", "headers": {}}}`
- `McpManager.discover()` 逐个连接并 `load_mcp_tools()` 列出工具；每个工具成为一条 MCP 能力（id 形如 `name:tool`）。连接失败 → 该 server 记一条 `unavailable` 能力（「不适配：…」），**不注入 Agent**。
- 连接保持存活（`ClientSession`），工具经 `registry.tool(cap_id)` 取回，在工具执行节点中与内置工具统一执行。

### 2.3 四种推理模式（create_agent + middleware / StateGraph）

`react` / `reflection` / `multi_agent` 统一基于 LangChain `create_agent` 构建：模型绑定、工具循环、状态管理由框架内建；模式差异收敛到自定义 `AgentMiddleware` 钩子（`abefore_model` / `awrap_model_call` / `aafter_model` / `awrap_tool_call`）。`plan_execute` 使用 LangGraph `StateGraph` 原生编排（planner → executor ⇄ tools → replanner），以体现 LangGraph 显式构图与条件边。

| 模式 | 实现 | 说明 |
|---|---|---|
| `react` | `StreamEventsMiddleware` | 模型 ⇄ 工具循环由 create_agent 内建；中间件发射 thinking/message 事件并处理工具 HITL |
| `plan_execute` | `StateGraph`（planner/executor/tools/replanner） | planner 拆解任务；executor 对当前步骤流式模型调用，产出 tool_calls 路由 tools，否则推进 `current_step`；`should_replan` 条件边：步骤工具失败触发 replanner 重规划（受 `max_replans` 限制），全部完成则 end。`tools` 节点复用 `make_tools_node`，`executor` 复用 `stream_model_call` |
| `reflection` | `ReflectionMiddleware` | 单次模型调用内（`awrap_model_call`）完成 草稿 → 批评 → 修订 → 再批评，批评为「无」或达 `max_iterations` 终止；不参与工具循环 |
| `multi_agent` | `MultiAgentMiddleware` + `StreamEventsMiddleware` | 编排者 create_agent 把 compute/analyze 两个子代理经 `convert_runnable_to_tool` 包装为工具；`MultiAgentMiddleware` 发射分派/完成事件 |

- 子代理（compute/analyze）使用 `WorkerEventsMiddleware`，不持有 checkpointer，因此不触发 HITL 中断；审批统一收敛到编排者层。
- 所有顶层图 `compile(checkpointer=MemorySaver)`，支持多轮会话与中断恢复。

### 2.4 Human-in-the-loop（HITL）

- 审批策略：`approval_policy = always | never`（经 `config['configurable']` 传入）。
- 拦截点：`StreamEventsMiddleware.awrap_tool_call` → `_execute_tool_call` 中，执行工具前 `interrupt({tool_calls})` 暂停。
- `always`：工具执行前 interrupt 暂停 → 后端产出 `approval_request` 事件（含 `approval_id`）→ 前端弹窗 → `POST /api/approve` 携带 `decision`（approve/reject/modify）与 `modified_args` → 用同一 checkpointer 重建图并 `Command(resume=...)` 恢复。
- reject 时向模型注入「用户拒绝」的 ToolMessage，模型需改用其它方式；modify 时按 `modified_args` 覆写工具参数后再执行。

### 2.5 模型层：DashScope 原生 SDK（reason 与 output 分离）

- `llm/dashscope_chat.py` 实现 `DashScopeChatModel(BaseChatModel)`，直接调用阿里云官方 `dashscope.Generation.call`，开启 `enable_thinking=True` 与 `incremental_output=True`，无需 OpenAI 兼容层。
- 响应处理用 `DashScopeTurn` 数据结构承载两类结果：
  - `reasoning` ← `reasoning_content`（思考过程，前端灰色斜体）；
  - `output` ← `content`（最终输出，前端回答区）。
- 工具调用：`bind_tools` 转 DashScope function schema；流式增量 tool_calls 经 `_to_tool_call_chunks` 转 LangChain `ToolCallChunk`，由 `AIMessageChunk.__add__` 合并回完整调用；`_to_dashscope_messages` 双向转换消息（assistant 回写 tool_calls、tool 回写 tool_call_id）。
- `events_mw.py::awrap_model_call` 用 `astream` 逐 token 生成：`reasoning_content` → `thinking` 事件、`content` → `message` 事件，实现「先展示思考 → 执行工具 → 返回输出」的实时体验。

### 2.6 SSE 事件协议

`core/events.py` 统一构造事件，`chat.py` 经 `EventSourceResponse` 下发。事件类型：

| type | 说明 | 关键字段 |
|---|---|---|
| `meta` | 会话/模式/启用能力 | `session_id`, `mode`, `capabilities[]` |
| `thinking` | 推理过程（增量） | `delta` |
| `message` | 回答（增量） | `delta` |
| `tool_start` / `tool_end` | 工具调用开始/结束 | `tool`, `args`, `result`, `success` |
| `tool_retry` | 工具层透明重试进度 | `tool`, `attempt`, `max`, `delay`(实际睡眠), `base_delay`(纯指数值), `reason` |
| `plan` | 计划更新 | `steps`, `current_step`, `status` |
| `retrieve` | RAG 检索 | `query`, `hits[]`(score) |
| `memory_write` / `memory_read` | 记忆写入/召回 | `content`, `source` |
| `approval_request` | HITL 审批请求 | `approval_id`, `tool_calls[]` |
| `reflect` / `revise` | reflection 过程 | `stage`, `critique` |
| `agent_event` | multi-agent 分派/汇总 | `worker`, `status`, `task`, `result` |
| `prompt_result` | 提示词策略结果 | `strategy`, `summary` |
| `done` | 完成 | `summary`, `stats` |
| `error` | 异常 | `message`, `detail` |

### 2.6 工具执行（StreamEventsMiddleware）

`StreamEventsMiddleware.awrap_tool_call` → `_execute_tool_call`：按 `approval_policy` 决定是否 interrupt 审批 → 推送 `tool_start`/`tool_end`（含结果与成功标记）→ 执行工具 → 返回 `ToolMessage`；工具异常兜底为失败事件与失败 ToolMessage。所有顶层模式复用；multi-agent 子代理用 `WorkerEventsMiddleware`（跳过审批）。

### 2.7 两层重试机制（tools/retry.py + harness）

工具失败按原因分层处理（`tools/retry.py`）：

- **工具层透明重试（Transparent Retry）**：对与参数无关的瞬时错误（网络超时/连接重置/服务端 5xx/限流 429，含 `RetryableToolError` 显式标记与 httpx 网络异常识别）用相同参数直接重试，最多 `tool_retry_max` 次、指数退避 + 抖动（`tool_retry_base_delay` / `tool_retry_max_delay`）；每次重试发射 `tool_retry` 事件，模型完全无感知。
- **Agent 层思考后重试（Model-mediated Retry）**：参数/业务错误（400/404/401/403/业务逻辑等）不直接重试，`format_tool_error` 生成含「错误类型/详情/建议」的结构化文本返回给模型，由模型重新思考、调整参数或换工具后再调用（ReAct 循环天然具备）；`harness.record_tool_failure` 累计同工具连续失败次数，成功即清零，达到 `agent_retry_max` 后 `tool_exhausted` 短路为「请改用其它工具」提示，避免无限烧迭代。

真实工具配合：`web_search` 网络/5xx/429、`run_command` OpenSandbox 连接层失败改为抛 `RetryableToolError`（参数/危险命令/超时/退出码等确定性结果仍走原有字符串返回）。

**可选报错注入（调试/验证两层重试）**：`harness.FAULT_TYPES` 提供 13 种注入类型，按重试分类两组，工具处理节点命中即短路执行：
- `retryable`（瞬时错误 → 工具层直接重试）：`timeout` / `conn_reset` / `dns` / `http_429` / `http_500` / `http_502` / `http_503`
- `permanent`（参数/业务错误 → 交给模型思考后重试）：`error` / `business` / `http_400` / `http_401` / `http_403` / `http_404`

注入通过 `POST /api/fault`（未知类型 400）、`GET /api/faults/types` 查看类型目录与分类；前端能力卡片故障注入下拉按两组展示。

**重试可视化（指数退避直观体现）**：`tool_retry` 事件同时下发 `delay`（实际睡眠，含 ±50% 抖动）与 `base_delay`（纯指数值 `min(cap, base*2^(attempt-1))`）。前端工具卡片在重试期间显示：
- 重试徽标 `重试 n/m · 退避 X.Xs`；
- 实时倒计时进度条（`X.Xs 后第 n+1 次重试` + 抖动后实际等待）；
- 指数退避阶梯条（`1.5s → 3s → 6s → …` 逐次翻倍、封顶，当前尝试高亮），直观展示指数退避曲线。

### 2.8 检索（RAG）与记忆

**RAG（独立检索阶段，不再作为能力卡/工具）**
- `rag/manager.py`：按 `RAG_SCHEMES` 构建 naive / advanced / modular 三方案，每方案独立向量库（`{prefix}_{scheme}`），启动只加载、不现场入库。
- `rag/schemes/`：naive（固定切块 + 纯稠密）/ advanced（语义分块 + 混合检索 + 改写/重排/父块回填）/ modular（前置语义路由 + 执行计划动态编排）。
- `rag/routing/`（路由/改写/指代消解/分解/HyDE）+ `rag/retrieval/`（RRF 融合/重排/压缩/多跳规划-执行-验证/充分性闸门）算子。
- `rag/corpus/`：云帆制度语料（`memory/corpus.py` 解析入库）。
- runner 在模型调用前按前端开关执行前置检索，命中注入上下文并下发 `retrieve` 事件；RAG 不映射为 LangChain 工具。

**记忆**
- `memory/vector_store.py`：`OpenAIEmbeddings` + 内存余弦相似度 top-k 检索。
- `memory/stores/`：Qdrant / Elasticsearch / 内存多后端可替换，`MultiBackendStore` 多路融合。
- `memory/session_store.py`：`MemorySaver` 检查点（多轮/恢复）+ 每会话长期记忆。
- 工具：`memory_tool`（写/读长期记忆，Embedding 可用时注入）。

## 3. 前端结构

- 视图：`HomeView`（能力池 + 模式 + 对话一体）、`ModuleView`（单模式深研：流式运行 + 审批 + 代码 + 原理）、`CompareView`（同任务多模式并排对比）。
- 组合式函数：`useChatStream`（SSE 状态机）、`useCapabilities`（能力加载/开关/示例）、`useDemoRunner`（对比视图演示编排）。
- 组件：能力池（`CapabilityCard`/`CapabilityGrid`/`ExampleFillHint`）、对话（`TaskInput`/`LiveStage`/`StreamingText`/`StepTimeline`/`ToolCallBadge`）、控制（`ModeSelector`/`PromptStrategyPicker`/`ControlsBar`）、系统（`ApprovalDialog`/`ErrorBanner`/`CodeSnippet`/`PrinciplePanel`/`FlowDiagram`）、布局（`AppHeader`/`FooterBar`）。
- 代码展示：`CodeSnippet` 拉取 `/api/source/{module}` 真实源码（highlight.js + 行注释），杜绝展示与运行不一致。

## 4. 配置项

| 配置 | 环境变量 | 默认 |
|---|---|---|
| 百炼（DashScope）Key | `LLM_API_KEY` | 空（必填） |
| Chat 模型 | `CHAT_MODEL` | `qwen3.5-flash` |
| Embedding Key | `EMBEDDING_API_KEY` | 空 |
| MCP Servers | `MCP_SERVERS` | `{}` |
| 最大迭代 | `MAX_ITERATIONS` | 8 |
| RAG top-k | `RAG_TOP_K` | 3 |
| 上下文阈值 | `CONTEXT_THRESHOLD` | 12 |
| 工具层重试次数 | `TOOL_RETRY_MAX` | 3 |
| 重试退避基础秒数 | `TOOL_RETRY_BASE_DELAY` | 1.5 |
| 重试最长退避秒数 | `TOOL_RETRY_MAX_DELAY` | 8.0 |
| Agent 层重试上限 | `AGENT_RETRY_MAX` | 3 |
| CORS 来源 | `CORS_ORIGINS` | `["http://localhost:5173","http://localhost:8000"]` |
