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
---
## 为什么需要它

MCP（Model Context Protocol）是 AI 工具调用的"USB 标准"——标准化 Server-Client 接口，一次实现 MCP Server，所有支持 MCP 的 Agent 都能用。工具以独立服务形式存在，Agent 运行时按需连接和断开，是函数调用的标准化演进。

## 怎么解决

难点在于工具发现协议的实现、动态 schema 生成，以及不同 MCP Server 的连接管理。我实现了 MCP Client SDK 的核心子集，支持 SSE 和 stdio 两种传输方式，工具列表动态注入到 prompt 中。

## 核心实现

```python
# MCP 工具管理器
class MCPToolManager:
    async def connect_server(self, config):
        transport = SSETransport(config.url)
        server = MCPServer(transport)
        await server.initialize()

        tools = await server.list_tools()
        for tool in tools:
            self.tool_registry[tool.name] = {
                "schema": tool.inputSchema,
                "server": server,
                "description": tool.description,
            }
        return tools

    def get_enabled_tools(self, enabled_names):
        return [
            self.tool_registry[name]
            for name in enabled_names
            if name in self.tool_registry
        ]
```

## 收益与边界

- 工具以 MCP Server 形式独立部署，解耦彻底
- 运行时动态连接 / 断开，零停机扩缩能力
- 标准化协议，社区工具生态直接复用
