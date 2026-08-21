"""MCP 集成：管理配置的 MCP Server，发现工具并暴露为能力；连接失败标记不适配。"""
from __future__ import annotations

import json


class McpManager:
    def __init__(self, servers_json: str = "{}"):
        try:
            raw = json.loads(servers_json) if servers_json else {}
        except Exception:
            raw = {}
        self.servers = raw if isinstance(raw, dict) else {}
        self.capabilities: list[dict] = []
        self.tools_by_id: dict[str, object] = {}
        self._sessions: dict[str, object] = {}
        self._streams: dict[str, object] = {}
        self._discovered = False

    async def discover(self) -> None:
        """连接全部配置的 MCP Server 并收集工具；单个失败不影响其它。"""
        if self._discovered:
            return
        self._discovered = True
        for name, conf in self.servers.items():
            try:
                tools = await self._load_tools(name, conf)
                for t in tools:
                    cap_id = f"{name}:{t.name}"
                    self.capabilities.append(
                        {
                            "id": cap_id,
                            "name": f"{name} · {t.name}",
                            "source": "mcp",
                            "server": name,
                            "requires": None,
                            "availability": "available",
                            "unavailable_reason": None,
                            "desc": getattr(t, "description", "") or f"MCP Server {name} 提供的工具",
                            "example": f"使用 {name} 的 {t.name} 工具完成任务",
                            "code_key": "mcp",
                        }
                    )
                    self.tools_by_id[cap_id] = t
            except Exception as exc:
                self.capabilities.append(
                    {
                        "id": f"{name}:*",
                        "name": f"{name}（连接失败）",
                        "source": "mcp",
                        "server": name,
                        "requires": None,
                        "availability": "unavailable",
                        "desc": f"MCP Server {name} 无法连接，能力不适配",
                        "example": "",
                        "code_key": "mcp",
                        "unavailable_reason": f"不适配：{exc}",
                    }
                )

    async def _load_tools(self, name: str, conf: dict):
        """按配置类型连接 MCP Server 并加载其工具，连接保持存活。"""
        if "command" in conf:
            return await self._load_stdio(name, conf)
        if "url" in conf:
            return await self._load_http(name, conf)
        raise ValueError("MCP 配置需包含 command（stdio）或 url（HTTP）")

    async def _load_stdio(self, name: str, conf: dict):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        from langchain_mcp_adapters.tools import load_mcp_tools

        params = StdioServerParameters(
            command=conf["command"], args=conf.get("args", []), env=conf.get("env")
        )
        entered = await stdio_client(params).__aenter__()
        read, write = entered[0], entered[1]
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()
        tools = await load_mcp_tools(session)
        self._sessions[name] = session
        self._streams[name] = (read, write)
        return tools

    async def _load_http(self, name: str, conf: dict):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        from langchain_mcp_adapters.tools import load_mcp_tools

        ctx = streamablehttp_client(conf["url"], headers=conf.get("headers", {}))
        entered = await ctx.__aenter__()
        read, write = entered[0], entered[1]
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()
        tools = await load_mcp_tools(session)
        self._sessions[name] = session
        self._streams[name] = (read, write)
        return tools

    def tool(self, cap_id: str):
        """返回某能力对应的 LangChain 工具（未连接则 None）。"""
        return self.tools_by_id.get(cap_id)
