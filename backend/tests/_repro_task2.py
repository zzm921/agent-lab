"""精确复现后端真实场景：
1) enable() 在任务 A 中 __aenter__ 连接（模拟 POST /api/mcp 请求）
2) 随后在独立任务 B 中调用工具（模拟 POST /api/stream → AgentRunner._run_graph 的后台任务）
观察是否出现「空 detail 的工具失败」。"""
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from langchain_mcp_adapters.tools import load_mcp_tools

URL = "http://127.0.0.1:8001/mcp"


async def main():
    print("== 任务 A：建立连接 ==")
    ctx = streamable_http_client(URL)
    entered = await ctx.__aenter__()
    read, write = entered[0], entered[1]
    session = await ClientSession(read, write).__aenter__()
    await session.initialize()
    tools = await load_mcp_tools(session)
    print("  tools:", [t.name for t in tools])
    holder = {"ctx": ctx, "tools": tools}
    await asyncio.sleep(0.2)  # 模拟请求 A 已结束

    print("== 任务 B：独立任务中调用 save_note ==")

    async def call():
        try:
            res = await holder["tools"][0].ainvoke({"title": "跨任务测试", "content": "hello"})
            print("  任务 B 调用结果:", res)
            return "OK"
        except Exception as e:
            print("  任务 B 调用异常:", type(e).__name__, "str=", repr(str(e)))
            return f"FAIL: {type(e).__name__}"

    result = await asyncio.create_task(call())
    print("== 结果:", result, "==")


asyncio.run(main())
