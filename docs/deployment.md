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
CHAT_MODEL=qwen3.5-flash
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

### 2.3 离线建库（RAG，线上前运行）

RAG 向量库数据在**线上前**通过独立建库脚本预写（在线服务启动时只加载、不现场入库，保证能力加载快速）。不同 RAG 方案各自独立成脚本，按需分别运行；需配置 `EMBEDDING_API_KEY` 与 `QDRANT_URL`（或 ES）：

```bash
cd backend
python scripts/ingest_naive.py     # naive 方案：固定切块 + 纯稠密检索
python scripts/ingest_advanced.py  # advanced 方案：语义分块 + 混合检索
```

脚本幂等：语料未变则跳过，变更自动重建；`--force` 强制清空重建。

## 3. 前端

```bash
cd frontend
npm install
npm run dev        # 开发模式 http://localhost:5173（需后端 8000 端口运行）
```

## 4. 生产部署

```bash
cd backend
python scripts/ingest_naive.py && python scripts/ingest_advanced.py && python scripts/ingest_modular.py  # 线上前建库
cd ../frontend && npm run build                                      # 产物输出到 frontend/dist
cd ../backend && uvicorn app.main:app --port 8000
```

`app/main.py` 检测到 `frontend/dist` 存在时自动以静态资源托管前端，直接访问 http://localhost:8000 即可使用完整功能（单 uvicorn 进程）。服务启动时自动连接并发现 MCP（stdio 子进程拉起 `mcp-info`），无需单独启动。`MCP_ENABLED=false` 可关闭。

## 5. MCP 集成配置示例

> 本项目自带 `mcp-info` 只读 server 默认以 **stdio** 方式由在线服务启动时自动拉起（`MCP_ENABLED` 默认 `true`），无需手动启动；如需独立 HTTP 部署仍可用 `uvicorn app.mcp_server.info_server:app --port 8001`。以下为通用 MCP 配置示例。

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

## 6. 命令执行沙箱（OpenSandbox，Docker 部署）

`run_command` 工具默认在 OpenSandbox 中执行（`SANDBOX_BACKEND=opensandbox`），
也可通过 `local` 切回本机轻量沙箱兜底。
无论哪种后端，该工具调用前都**强制人工审批（HITL）**。

### 6.1 部署 OpenSandbox Server

```bash
cd deploy/opensandbox
docker compose up -d
curl http://127.0.0.1:8090/health     # → {"status": "healthy"}
```

说明：
- 服务端通过挂载的 `/var/run/docker.sock` 创建沙箱容器，需运行在装有 Docker 的机器上；
- Windows/macOS（Docker Desktop）可直接用 `host.docker.internal`；Linux 宿主机请取消
  `docker-compose.yml` 中 `extra_hosts` 注释；
- 如需鉴权，在 `docker-compose.yml` 的 config 中设置 `api_key`，并与后端 `OPENSANDBOX_API_KEY` 一致。

### 6.2 后端接入

```ini
# backend/.env
SANDBOX_BACKEND=opensandbox
SANDBOX_WORK_DIR=./data/sandbox-work   # 沙箱/宿主机共享工作目录（前端可下载其中的文件）
SANDBOX_MOUNT_TARGET=/work             # 工作目录在沙箱内的挂载点
OPENSANDBOX_DOMAIN=localhost:8090
OPENSANDBOX_PROTOCOL=http
OPENSANDBOX_API_KEY=                 # 与服务端 config 的 api_key 一致（未开启可留空）
OPENSANDBOX_IMAGE=ubuntu:22.04       # 沙箱容器镜像
```

### 6.3 沙箱文件持久化与下载

- `run_command` 会把 `SANDBOX_WORK_DIR` 以 Volume 挂载进沙箱（挂载到 `SANDBOX_MOUNT_TARGET`）；
  沙箱内写入该目录的文件会**持久化到宿主机**，沙箱销毁后仍存在。
- **必须先放行宿主机路径**：在 `deploy/opensandbox/docker-compose.yml` 的 `[storage].allowed_host_paths`
  中加入 `SANDBOX_WORK_DIR` 的绝对路径（例如 `["D:/workspace/my-agent-lab/backend/data/sandbox-work"]`），
  否则创建沙箱会因路径不在白名单内而失败。
- 前端聊天页右上「沙箱文件」按钮打开面板：列出工作目录中的文件，点击「下载」即从
  `GET /api/sandbox/files/download?path=...` 下载；`run_command` 执行结束后列表自动刷新。
- `local` 兜底后端同样以 `SANDBOX_WORK_DIR` 作为工作目录，产物同样可下载。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `/api/stream` 返回「未配置 LLM_API_KEY」 | 在 `backend/.env` 配置百炼 DashScope Key 后重启 |
| RAG/记忆能力显示「不适配」 | 配置 `EMBEDDING_API_KEY/BASE_URL/MODEL`（OpenAI 兼容接口） |
| MCP 能力「不适配」 | 检查 `MCP_SERVERS` 格式与目标服务是否可达，重启后端 |
| `run_command` 报「创建 OpenSandbox 沙箱失败 …not in allowed host paths / 权限」 | 在 `deploy/opensandbox/docker-compose.yml` 的 `[storage].allowed_host_paths` 加入 `SANDBOX_WORK_DIR` 绝对路径，`docker compose up -d` 重启 |
| 前端流式无响应 | 确认后端 8000 端口运行；检查浏览器控制台网络（SSE） |
| 端口占用 | 换端口：`uvicorn app.main:app --port 8001`，并同步 `CORS_ORIGINS` |
