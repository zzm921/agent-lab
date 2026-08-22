"""通过真实 Agent 链路复现 save_note 失败：Fake 模型发 save_note 工具调用，观察结果。"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from app.capabilities.mcp import McpManager
from app.agents.tools_builder import build_tools
from app.agents.harness import AgentHarness
from app.capabilities.registry import CapabilityRegistry
from app.config import settings
from app.memory.session_store import SessionStore

SERVERS = '{"mcp-notes": {"url": "http://127.0.0.1:8001/mcp"}}'


class FakeLLM:
    async def ainvoke(self, messages, **kw):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_note",
                    "args": {"title": "张三的笔记", "content": "我叫张三"},
                    "id": "call_123",
                    "type": "tool_call",
                }
            ],
        )


async def main():
    mcp = McpManager(SERVERS, enabled=True)
    await mcp.enable()
    store = SessionStore()
    reg = CapabilityRegistry(settings, store, mcp, None, None)
    harness = AgentHarness(settings)
    emit = lambda d: print("EVENT:", d.get("type"), d.get("tool"), d.get("success"))

    tools = build_tools(reg, ["mcp-notes:save_note"], "s1", emit)
    print("tools =", [t.name for t in tools])
    t = tools[0]

    # 模拟 tools/runner.py 的调用方式
    from app.tools.retry import invoke_with_retry, format_tool_error

    async def _run():
        return await t.ainvoke({"title": "张三的笔记", "content": "我叫张三"})

    output, success, error, retries = await invoke_with_retry(_run, "save_note", settings, emit)
    print("success =", success, "retries =", retries)
    print("output =", repr(output))
    if error:
        print("error type =", type(error).__name__)
        print("error =", repr(error))
        print("error str =", str(error))
        print("formatted =", format_tool_error("save_note", error, retried=retries))
    mcp.disable()


asyncio.run(main())
