---
id: mcp
name: MCP 工具热插拔
shortDesc: 基于 Model Context Protocol，工具像插件一样按需加载，无需重启服务。
icon: plug
difficulty: adv
completeLevel: 90
tags: [MCP, Tooling, Protocol]
techFilters: [MCP]
accent: '#22d3a8'
enabledTools: [save_note, list_notes, get_note, delete_note]
prompts:
  - 帮我把「周三下午三点和客户张总开会」记成一条笔记，然后列出我所有的笔记标题。
  - 查看我笔记里关于「RAG」的笔记内容。
  - 把「记住：老板不喜欢红色方案」存为一条笔记。
---
## 概述

MCP（Model Context Protocol）是 AI 工具调用的「USB 标准」——一套**标准化的 Server–Client 接口协议**：MCP Server 把工具 / 资源 / 提示词能力声明为服务，任何支持 MCP 的 Agent（Client）都能动态发现并调用，无需为每个 Agent 单独适配。

一句话：**工具不再写死在 Agent 里，而是像插件一样挂在外面，Agent 运行时按需连接、发现、调用、断开。**

本项目把 MCP 落成**一条完整可运行的链路**，并且默认开启、服务启动即就绪，页面侧边栏可随时关闭，直观对比「有 MCP / 无 MCP」的能力差异：

- **服务端（MCP Server）**：自建 `mcp-notes` 便签服务（FastMCP + **stdio 传输**，由在线服务启动时以子进程自动拉起），提供 `save_note / list_notes / get_note / delete_note` 四个工具，JSON 文件持久化；
- **客户端（MCP Client）**：后端 `McpManager` 默认开启（`MCP_ENABLED=true`），**服务启动时自动连接 + 发现工具** → 工具以 `mcp-notes:xxx` 能力出现在页面，可逐个启用/示例/对话调用；页面「MCP 服务」开关**只控制这些能力是否进入目录**（服务连接在启动时已建立、与开关无关），关闭则能力从目录消失。

## 为什么需要

工具能力进化的三个痛点，MCP 分别给出答案：

1. **标准不统一**：每个 Agent 都要为 OpenAI / 百度 / 自研 各自实现一套函数调用协议。MCP 是「打印机驱动标准」——协议标准化后，**一个 MCP Server，所有支持 MCP 的 Agent 都能用**；
2. **能力与主程序强耦合**：传统工具写死在 Agent 进程内，新增/下线工具要改代码、重启服务。MCP 把工具放到**独立进程的独立服务**里，Agent 运行时动态发现，**零停机扩展能力**；
3. **工具开发成本高**：为每个业务写一套工具接入代码重复劳动。MCP 社区已有大量现成 Server（文件系统、数据库、GitHub、浏览器…），直接复用生态。

对本项目的直接价值：能力是**可热插拔的**——MCP 工具与内置工具走同一套「能力 → 工具 → 对话调用」链路，开关即插即拔，不重启服务。

## 通用知识：MCP 协议核心

### Server 与 Client 的角色

```
┌─────────────────┐  传输(Transport)  ┌─────────────────┐
│   MCP Client    │ ◄───────────────► │   MCP Server    │
│   (Agent 宿主)  │  stdio / HTTP     │  (工具服务进程)  │
└─────────────────┘                   └─────────────────┘
        │  initialize（握手）              │
        │  tools/list   （发现工具列表）    │
        │  tools/call   （调用工具）        │
```

### 两种主流传输方式

| 传输 | 适用 | 特点 |
|------|------|------|
| **stdio** | Server 与 Client 同机 | Client 以子进程方式拉起 Server，通过标准输入输出通信；进程生命周期由 Client 管理 |
| **Streamable HTTP** | Server 可独立部署（推荐生产） | Server 以 HTTP 服务运行（独立端口），Client 通过 HTTP 长连接通信；可跨机、可托管、可用现有鉴权 |

### 一次典型调用过程

```
Client                     Server
  │  1. initialize 握手      │
  │ ───────────────────────► │  协商协议版本与能力
  │  2. tools/list           │
  │ ───────────────────────► │
  │ ◄─────────────────────── │  返回工具列表（名称/描述/JSON Schema 参数）
  │  3. tools/call {name,args}│
  │ ───────────────────────► │  执行工具逻辑
  │ ◄─────────────────────── │  返回结构化结果（isError/content）
```

核心思想：**工具以「服务」而非「代码」存在**——Client 不知道工具怎么实现，只按协议声明动态发现、按 Schema 校验参数、按响应取回结果。

## 本项目的做法

### 一、MCP Server 端（`backend/app/mcp_server/notes_server.py`）

用 `mcp` SDK 的 `FastMCP` 定义一个便签服务，数据以 JSON 文件持久化（`NotesStore`：线程锁 + 读盘→变更→临时文件 `os.replace` 原子写回）。

伪代码：

```python
# backend/app/mcp_server/notes_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-notes")                    # 声明服务名
store = NotesStore(os.getenv("MCP_NOTES_FILE", "./data/mcp-notes.json"))

@mcp.tool()
def save_note(title: str, content: str) -> str:
    """保存一条便签；同标题覆盖。"""
    return store.save(title, content)

@mcp.tool()
def list_notes() -> str:
    """列出全部便签标题与摘要（按更新时间倒序）。"""
    return store.list()

# get_note / delete_note 同理，中文 docstring 即工具描述

app = mcp.streamable_http_app()   # 可选：独立 HTTP 部署时用 uvicorn app.mcp_server.notes_server:app --port 8001
                                  # 默认以 stdio 由在线服务子进程拉起（python -m app.mcp_server.notes_server）
```

要点：

- `FastMCP` + `@mcp.tool()` 自动从函数签名生成 JSON Schema（参数、必填、描述），Client 侧 `tools/list` 拿到的就是它；
- 默认 **stdio 传输**：`mcp.run()` 走标准输入输出，在线服务以子进程自动拉起，无需手动启动；`streamable_http_app()` 可选地把它变成可独立部署的 HTTP 服务；
- 每个工具是**纯函数 + 原子持久化**，无状态、可并发、可挂到任何宿主上。

### 二、MCP Client 端（`backend/app/capabilities/mcp.py`）

后端 `McpManager` 是 MCP 客户端：读取 `.env` 里注册的 server 配置，**服务启动时默认连接**（`MCP_ENABLED=true`，stdio 以子进程自动拉起 server）；**连接在启动时建立并保持**，页面开关只决定 MCP 能力是否进入能力目录（转成 LangChain 工具注入 Agent 的开关）。

伪代码：

```python
class McpManager:
    def __init__(self, servers_json="{}", enabled=True):
        self.servers = parse(servers_json)   # {"mcp-notes": {"command": "python", "args": ["-m", "app.mcp_server.notes_server"]}}
        self.enabled = enabled               # 页面开关：是否在能力目录中使用 MCP（默认开启）
        self.capabilities: list[dict] = []   # 已发现的能力（是否暴露由 registry.list 按 enabled 过滤）
        self.tools_by_id: dict = {}          # cap_id -> LangChain 工具
        self._contexts: dict = {}            # 持有传输上下文，防 GC 关流

    async def enable(self):                  # 页面点选开启：仅把能力纳入目录，不重新连接
        self.enabled = True
        if not self._discovered:
            await self.discover()            # 仅当启动时未连接（MCP_ENABLED=false）才补连

    def disable(self):                       # 页面点选关闭：仅把能力移出目录，连接保持
        self.enabled = False

    async def discover(self):                # 服务启动时连接（幂等，与开关无关）
        if self._discovered:
            return
        for name, conf in self.servers.items():
            tools = await self._load_tools(name, conf)   # stdio / streamable http
            for t in tools:
                cap_id = f"{name}:{t.name}"
                self.capabilities.append({... source="mcp", server=name, availability="available"})
                self.tools_by_id[cap_id] = t

    async def _load_http(self, name, conf):
        http_client = httpx.AsyncClient(headers=conf.get("headers", {})) if conf.get("headers") else None
        ctx = streamable_http_client(conf["url"], http_client=http_client)  # 新版 API：headers 改由 http_client 携带
        read, write = await ctx.__aenter__()              # 进入传输上下文
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()                        # ① 握手
        tools = await load_mcp_tools(session)             # ② 内部 tools/list + schema 转换
        self._contexts[name] = ctx                        # ③ 持有上下文，连接持续复用
        return tools
```

关键工程点：

- **开关语义**：`enabled=False` 时 `discover()` 被门卫短路，能力目录不含任何 MCP 工具——这就是「默认不开」；
- **连接保活**：传输上下文存进 `self._contexts`，避免被 GC 回收导致流中断（anyio.WouldBlock / CancelledError）；
- **能力注册**：`mcp-notes:save_note` 这种「`server:tool`」id 与内置能力统一进能力目录，前端同一套卡片渲染；
- **工具注入**：`registry.tool_for(cap_id)` 对 MCP 能力走 `mcp.tool()` 返回 LangChain 工具，与 `calculator` 等内置工具同链路进入 `create_agent`。

### 三、开关 API（`backend/app/api/chat.py`）

```python
@router.get("/mcp")                  # 查询状态
async def mcp_status():
    mcp = get_registry().mcp
    return {"enabled": mcp.enabled, "servers": list(mcp.servers.keys()),
            "capabilities": mcp.capabilities}

@router.post("/mcp")                 # 页面点选开关
async def mcp_toggle(req: McpToggleRequest):
    mcp = get_registry().mcp
    if req.enabled and not mcp.enabled:
        await mcp.enable()           # 连接 + 发现
    elif not req.enabled and mcp.enabled:
        mcp.disable()                # 清空能力与工具映射
    return {"enabled": mcp.enabled, "capabilities": mcp.capabilities}
```

### 四、前端「MCP 服务」开关与分组（`useCapabilities.ts` + `CapabilitySidebar.vue`）

- 状态：`mcpEnabled` 初始 `false`，`loadMcp()` 从后端 `/api/mcp` 读取（后端默认 `MCP_ENABLED=true`）；`builtinCaps` / `mcpCaps` 按 `source` 分组；
- 交互：`setMcpEnabled(v)` → `POST /api/mcp` 成功后重新拉能力列表；
- 展示：侧边栏「MCP 服务」分组有开关——**开启态**（默认）显示「已连接 mcp-notes · 发现 N 个工具」+ fuchsia「MCP」徽标卡片（可逐个开关/示例/故障注入）；**关闭态**显示「MCP 未启用 — 仅使用内置能力」；内置能力单列一组，直观对比有无 MCP。

### 五、端到端流程：有无 MCP 的对比

| 阶段 | 有 MCP（默认开启） | 无 MCP（页面点选关闭） |
|------|-------------------|----------------------|
| 能力目录 | 内置能力 + 4 个 `mcp-notes:*`（fuchsia MCP 徽标，独立分组） | 只有内置能力（计算器/时间/搜索/沙箱…） |
| 侧边栏 | 「已连接 mcp-notes · 发现 4 个工具」 | 「MCP 服务」开关关闭，虚线框提示未启用 |
| 对话能力 | 发「记一条便签：明天 10 点开会」→ 模型调用 `save_note` → 落盘 `data/mcp-notes.json` | 记不了便签 |
| 开关动作 | 关闭开关 → MCP 分组消失，回到仅内置能力（不重启） | 再开启 → 重新连接并发现（不重启） |
| server 无法拉起 | 能力标「不适配（连接失败）」置灰，其余对话不受影响 | — |

伪代码（前端交互）：

```ts
// useCapabilities.ts
const mcpEnabled = ref(false)
const builtinCaps = computed(() => caps.value.filter(c => c.source !== 'mcp'))
const mcpCaps    = computed(() => caps.value.filter(c => c.source === 'mcp'))

async function loadMcp() {          // 初始化读取后端状态
  const res = await fetch('/api/mcp')
  mcpEnabled.value = Boolean((await res.json()).enabled)
}
async function setMcpEnabled(v: boolean) {   // 页面开关
  await fetch('/api/mcp', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ enabled: v }) })
  mcpEnabled.value = v
  await load()   // 重新拉能力列表：MCP 工具出现 / 消失
}
```

与通用设计的对照：

| 通用概念 | 本项目实现 |
|---------|-----------|
| MCP Server（工具服务进程） | `mcp-notes`（FastMCP + stdio，服务启动时以子进程自动拉起，JSON 持久化） |
| MCP Client（Agent 宿主） | `McpManager`：连接 + `tools/list` 发现 → 能力注册 → 工具注入 |
| 传输：stdio / Streamable HTTP | 两者均支持；本项目默认 stdio |
| 握手 / 发现 / 调用 | `initialize` → `load_mcp_tools`（内部 `tools/list`）→ `tools/call` |
| 工具 Schema | `@mcp.tool()` 函数签名自动生成（描述 + 参数 JSON Schema） |
| 能力目录 / 工具注入 | `registry.list()` 按 `mcp.enabled` 合并 MCP 能力；`registry.tool_for()` 解析 |
| 开关 / 热插拔 | 默认开；连接在服务启动时建立，开关仅控制能力是否入目录，不重启服务 |
| 连接失败降级 | 标记「不适配（连接失败）」置灰，不影响其它能力 |

## 收益与边界

**收益**

- 工具以独立服务存在，与 Agent 解耦，新增/下线能力不改主程序代码；
- 标准化协议：同一 MCP Server 可被任意支持 MCP 的 Agent 复用；
- 默认开启（服务启动即就绪）+ 页面可关闭 + 分组展示，能力热插拔直观可见，降级友好（连接失败不影响对话）。

**边界**

- 连接管理是进程级：`disable()` 只清空能力与工具映射、允许再次 discover，不显式关闭底层 session（连接由进程生命周期兜底）；
- stdio server 由在线服务以子进程自动拉起，无需手动启动；仅当 server 无法拉起（如模块/依赖缺失）时能力标「不适配」；
- 本项目是 MCP **Client + Server** 的自证闭环（一个自建 Server 验证全链路），更复杂场景（鉴权、资源/提示词、多 server 管理）可按同一协议扩展。

## 参考链接

- [Model Context Protocol 规范](https://modelcontextprotocol.io/) — 协议规范、传输、Schema 定义
- [MCP Python SDK（mcp）](https://github.com/modelcontextprotocol/python-sdk) — `FastMCP`、`ClientSession`、stdio / streamable http client
- [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) — `load_mcp_tools` 把 MCP 工具转为 LangChain 工具
- [Specification v1 中文社区解读](https://github.com/modelcontextprotocol) — 协议演进与生态示例
