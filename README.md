# 个人 AI Agent 技术展示平台

基于 **阿里云百炼（DashScope）Qwen + LangChain + LangGraph + MCP** 的个人智能代理平台：真实调用大模型、能力可热插拔、支持四种推理模式与 MCP 集成。

> 前端 Vue 3 + TypeScript + Vite + Tailwind；后端 FastAPI + LangGraph；SSE 流式输出。

## 核心特性

- **四种推理模式**（真实运行）：`react`（ReAct 思考-行动-观察）、`plan_execute`（计划-执行-再计划）、`reflection`（生成-反思-修订）、`multi_agent`（编排者分派 Worker 汇总）。
- **能力池 / 能力热插拔**：所有能力以「能力卡片」呈现，点击开关即时注入/移除 Agent 工具集；不可用能力标记「**不适配**」置灰。
- **示例一键填入**：每个能力卡片带示例按钮，点击自动启用该能力并把示例 prompt 填入输入框，可直接发送体验。
- **MCP 集成**：能力可来自 MCP Server（stdio / Streamable HTTP），自动发现工具；未配置或连接失败的能力标记「不适配」。
- **贯穿能力**：SSE 流式输出（思考/行动/观察/计划/反思/工具/检索/记忆）、Human-in-the-loop 工具审批、提示词策略（standard / few_shot / cot）、上下文管理与长期记忆、RAG 向量检索。
- **create_agent + middleware 架构**：四种推理模式统一基于 LangChain `create_agent` 构建，模式差异全部收敛到自定义 `AgentMiddleware`（事件流 / HITL / 计划 / 反思 / 多代理）。

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
| `LLM_API_KEY` | 是 | 阿里云百炼（DashScope）API Key，对话模型 `qwen-plus` |
| `LLM_BASE_URL` | 否 | 默认 `https://dashscope.aliyuncs.com/api/v1`（DashScope 原生 SDK） |
| `ENABLE_THINKING` | 否 | 默认 `true`：开启思考，返回 `reasoning_content`（思考）与 `content`（输出）两类结果 |
| `EMBEDDING_API_KEY` | 否 | RAG / 长期记忆需要（OpenAI 兼容接口，默认与百炼共用 Key） |
| `EMBEDDING_BASE_URL` | 否 | 默认 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL` | 否 | 默认 `text-embedding-v3` |
| `MCP_SERVERS` | 否 | JSON，声明 stdio 或 streamable HTTP 的 MCP Server |

详见 [docs/deployment.md](docs/deployment.md)。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/capabilities` | 能力目录（内置 + MCP，含可用性/示例） |
| POST | `/api/stream` | SSE 流式对话（模式/能力/策略/审批策略） |
| POST | `/api/approve` | HITL 审批（批准/拒绝/修改） |
| GET | `/api/source/{module}` | 后端真实源码（代码展示用） |
| GET | `/api/health` | 健康检查 |

## 目录结构

```
my-agent/
├── backend/     FastAPI + LangGraph + MCP + 工具/记忆/能力
│   └── tests/   pytest（不联网，Fake 模型 + mock MCP）
├── frontend/    Vue 3 + TS + Tailwind（Home/Module/Compare 视图 + 组件）
│   └── tests/   vitest
└── docs/        架构 / 部署 / 测试 文档
```

## 文档

- [docs/architecture.md](docs/architecture.md) — 架构、SSE 事件协议、四种模式（create_agent + middleware）、能力与 MCP 设计
- [docs/deployment.md](docs/deployment.md) — 部署与配置指南
- [docs/testing.md](docs/testing.md) — 测试用例与测试报告
