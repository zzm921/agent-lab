"""通过真实 AgentRunner.stream 复现：react 模式 + mcp-notes save_note 工具。"""
import asyncio
import json

from langchain_core.messages import AIMessage

from app.agents.runner import AgentRunner
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.config import Settings
from app.llm.fake_model import FakeChatModel
from app.memory.session_store import SessionStore

SERVERS = '{"mcp-notes": {"url": "http://127.0.0.1:8001/mcp"}}'


class FakeLLM(FakeChatModel):
    calls: int = 0

    def _next(self):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "save_note", "args": {"title": "张三的笔记", "content": "我叫张三"}, "id": "call_1", "type": "tool_call"}
                ],
            )
        return AIMessage(content="已保存便签", tool_calls=[])


async def main():
    settings = Settings(mcp_servers=SERVERS, mcp_enabled=True)
    mcp = McpManager(SERVERS, enabled=True)
    await mcp.enable()
    store = SessionStore()
    reg = CapabilityRegistry(settings, store, mcp, None, None)
    llm = FakeChatModel()  # 默认返回空；用 FakeLLM 替代
    runner = AgentRunner(settings, FakeLLM(), reg, store)

    print("capabilities:", [c["id"] for c in reg.list() if c["source"] == "mcp"])
    events = []
    async for ev in runner.stream(
        session_id="s-mcp",
        message="帮我记一条便签：我叫张三",
        mode="react",
        enabled=["mcp-notes:save_note"],
        prompt_strategy="standard",
        approval_policy="never",
    ):
        events.append(ev)
        if ev.get("type") in ("tool_start", "tool_end", "message", "done", "error"):
            print("EVT:", json.dumps(ev, ensure_ascii=False)[:300])

    mcp.disable()


asyncio.run(main())
