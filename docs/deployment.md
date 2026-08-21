# 部署与环境配置（Deployment）

## 1. 环境要求

- Python 3.11+（建议 3.12/3.13）
- Node.js 18+（构建前端）
- 可访问阿里云百炼（DashScope）API；可选 Embedding（OpenAI 兼容）与 MCP Server

## 2. 后端

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2.1 配置 .env

```bash
copy .env.example .env   # Windows
# 或
cp .env.example .env     # Linux/macOS
```

关键项：

```ini
# 必填（阿里云百炼 DashScope，Key 在 https://bailian.console.aliyun.com/ 获取）
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/api/v1
CHAT_MODEL=qwen-plus
# 开启思考：返回 reasoning_content（思考过程）与 content（最终输出）两类结果
ENABLE_THINKING=true

# 可选：RAG / 长期记忆能力（默认与百炼共用同一 Key）
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3

# 可选：MCP Servers（JSON 字符串）
MCP_SERVERS={}
```

### 2.2 启动

```bash
uvicorn app.main:app --reload --port 8000
```

未配置 `LLM_API_KEY` 时，`POST /api/stream` 返回 `500 {"detail":"未配置 LLM_API_KEY（阿里云百炼 DashScope API Key），请在 backend/.env 中设置后重启服务"}`。

## 3. 前端

```bash
cd frontend
npm install
npm run dev        # 开发模式 http://localhost:5173（需后端 8000 端口运行）
```

## 4. 生产部署

```bash
cd frontend && npm run build        # 产物输出到 frontend/dist
cd ../backend && uvicorn app.main:app --port 8000
```

`app/main.py` 检测到 `frontend/dist` 存在时自动以静态资源托管前端，直接访问 http://localhost:8000 即可使用完整功能（单 uvicorn 进程）。

## 5. MCP 集成配置示例

### 5.1 stdio transport

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\tmp"]
  }
}
```

### 5.2 streamable HTTP transport

```json
{
  "my-http-server": {
    "url": "http://localhost:8001/mcp",
    "headers": { "Authorization": "Bearer xxxx" }
  }
}
```

写入 `backend/.env` 的 `MCP_SERVERS`（单行 JSON 字符串）：

```ini
MCP_SERVERS={"filesystem":{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","C:\\tmp"]}}
```

配置说明：

- 连接成功后，其工具自动以「能力卡片」出现在前端能力池（来源徽标 `MCP`），id 形如 `filesystem:read_file`。
- 连接失败或未配置 → 显示「不适配」并给出原因，不注入 Agent。
- 修改 `MCP_SERVERS` 后需重启后端。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `/api/stream` 返回「未配置 LLM_API_KEY」 | 在 `backend/.env` 配置百炼 DashScope Key 后重启 |
| RAG/记忆能力显示「不适配」 | 配置 `EMBEDDING_API_KEY/BASE_URL/MODEL`（OpenAI 兼容接口） |
| MCP 能力「不适配」 | 检查 `MCP_SERVERS` 格式与目标服务是否可达，重启后端 |
| 前端流式无响应 | 确认后端 8000 端口运行；检查浏览器控制台网络（SSE） |
| 端口占用 | 换端口：`uvicorn app.main:app --port 8001`，并同步 `CORS_ORIGINS` |
