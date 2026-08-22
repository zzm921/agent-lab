"""复现 save_note 失败：走完整 McpManager → tool.ainvoke 链路。"""
import asyncio

from app.capabilities.mcp import McpManager

SERVERS = '{"mcp-notes": {"url": "http://127.0.0.1:8001/mcp"}}'


async def main():
    mcp = McpManager(SERVERS, enabled=True)
    await mcp.enable()
    tool = mcp.tool("mcp-notes:save_note")
    print("tool =", tool)
    args = {"title": "张三的笔记", "content": "我叫张三"}
    print("args =", args)
    try:
        res = await tool.ainvoke(args)
        print("RESULT =", res)
    except Exception as exc:
        import traceback
        print("EXC TYPE =", type(exc).__name__)
        print("EXC =", repr(exc))
        traceback.print_exc()


asyncio.run(main())
