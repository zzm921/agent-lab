"""验证跨任务问题：在一个 task 中 __aenter__ 连接，在另一个 task 中调用工具。
模拟后端：POST /api/mcp（请求任务 A 中 enable）→ 之后 Agent 后台任务 B 中调用工具。"""
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from langchain_mcp_adapters.tools import load_mcp_tools

URL = "http://127.0.0.1:8001/mcp"


async def scenario_a():
    """同任务进入 + 同任务调用（应成功）"""
    ctx = streamable_http_client(URL)
    read, write = await ctx.__aenter__()
    session = await ClientSession(read, write).__aenter__()
    await session.initialize()
    tools = await load_mcp_tools(session)
    res = await tools[0].ainvoke({"title": "测试A", "content": "同任务"})
    print("A 同任务调用:", "OK" if res else "FAIL", res)
    return ctx


async def scenario_b(ready, do_call):
    """跨任务：主协程进入连接后结束（模拟请求完成），另一任务中调用工具。"""
    ctx = streamable_http_client(URL)
    read, write = await ctx.__aenter__()
    session = await ClientSession(read, write).__aenter__()
    await session.initialize()
    tools = await load_mcp_tools(session)
    print("B 连接已建立（模拟 enable 请求结束）")
    ready.set()

    async def call():
        await do_call.wait()
        try:
            res = await tools[0].ainvoke({"title": "测试B", "content": "跨任务"})
            print("B 跨任务调用结果:", res)
        except Exception as e:
            print("B 跨任务调用异常:", type(e).__name__, "str=", repr(str(e)))
    return ctx, call


async def main():
    # 场景 A：同任务
    ctx_a = await scenario_a()

    # 场景 B：enable 的请求任务结束后，另起任务调用
    ready = asyncio.Event()
    do_call = asyncio.Event()
    ctx_b, call_b = await scenario_b(ready, do_call)
    t = asyncio.create_task(call_b())
    await asyncio.sleep(0.5)
    do_call.set()
    await t
    print("done")


asyncio.run(main())
