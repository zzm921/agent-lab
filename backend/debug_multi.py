"""临时调试：multi_agent worker 失败路径（跑完删除）。"""
import asyncio

from langchain_core.messages import AIMessage

from app.agents.runner import AgentRunner
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.llm.fake_model import FakeChatModel, FakeEmbeddings
from app.memory.session_store import SessionStore
from app.memory.vector_store import VectorStore
from app.rag.manager import RagManager
from tests.conftest import ai_with_tool, collect_stream, make_settings


async def main():
    settings = make_settings()
    embeddings = FakeEmbeddings()
    sessions = SessionStore()
    rag = RagManager(settings, embeddings, top_k=settings.rag_top_k)
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), rag, embeddings)

    llm = FakeChatModel()
    llm.script = [
        ai_with_tool(
            "派发计算",
            name="worker",
            args={"tasks": [{"id": "t1", "role": "compute", "task": "计算 2+2"}]},
        ),
        *[ai_with_tool(f"计算{i}", name="calculator", args={"expression": "1+1"}) for i in range(6)],
        AIMessage(content="最终汇总答案"),
    ]
    runner = AgentRunner(settings, llm, registry, sessions)
    events = await collect_stream(runner, mode="multi_agent", enabled=["calculator"])
    for e in events:
        print(e.get("type"), e.get("worker"), e.get("status"), e.get("task_id"), str(e.get("result"))[:80])


if __name__ == "__main__":
    asyncio.run(main())
