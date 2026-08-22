"""临时验证：MCP client → mcp-notes server 的 tools/call 链路（验证后删除）。"""
import asyncio

from app.capabilities.mcp import McpManager


async def main():
    mcp = McpManager('{"mcp-notes": {"url": "http://127.0.0.1:8001/mcp"}}', enabled=True)
    await mcp.enable()
    print("caps:", [c["id"] for c in mcp.capabilities])
    print("save:", await mcp.tool("mcp-notes:save_note").ainvoke({"title": "验证", "content": "MCP 链路打通"}))
    print("list:", await mcp.tool("mcp-notes:list_notes").ainvoke({}))
    print("get:", await mcp.tool("mcp-notes:get_note").ainvoke({"title": "验证"}))
    print("delete:", await mcp.tool("mcp-notes:delete_note").ainvoke({"title": "验证"}))


asyncio.run(main())
