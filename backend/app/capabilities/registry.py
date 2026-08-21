"""能力注册表：合并内置能力与 MCP 能力，负责可用性判断与工具解析。"""
from __future__ import annotations

from app.capabilities.builtin import BUILTIN_CAPABILITIES
from app.capabilities.mcp import McpManager
from app.config import Settings
from app.memory.session_store import SessionStore
from app.memory.vector_store import VectorStore
from app.tools.calculator import calculator
from app.tools.memory_tool import make_memory_tools
from app.tools.rag_tool import make_rag_tool
from app.tools.time_now import time_now
from app.tools.web_search import web_search


class CapabilityRegistry:
    def __init__(
        self,
        settings: Settings,
        session_store: SessionStore,
        mcp_manager: McpManager,
        corpus_store: VectorStore | None,
        embeddings,
    ):
        self.settings = settings
        self.sessions = session_store
        self.mcp = mcp_manager
        self.corpus_store = corpus_store
        self.embeddings = embeddings
        self._index: dict[str, dict] = {}

    async def refresh(self) -> None:
        """触发 MCP 发现（幂等）。"""
        await self.mcp.discover()

    def list(self) -> list[dict]:
        """返回完整能力目录（内置 + MCP），含可用性与不适配原因。"""
        caps = []
        for c in BUILTIN_CAPABILITIES:
            ok = self.embeddings is not None or c["requires"] is None
            caps.append(
                {
                    "id": c["id"],
                    "name": c["name"],
                    "source": c["source"],
                    "desc": c["desc"],
                    "example": c["example"],
                    "code_key": c["code_key"],
                    "availability": "available" if ok else "unavailable",
                    "unavailable_reason": None if ok else "未配置 Embedding API Key（EMBEDDING_API_KEY）",
                }
            )
        caps.extend(self.mcp.capabilities)
        self._index = {c["id"]: c for c in caps}
        return caps

    def get(self, cap_id: str) -> dict | None:
        """按 id 获取能力（索引未构建时先构建）。"""
        if not self._index:
            self.list()
        return self._index.get(cap_id)

    def tool_for(self, cap_id: str, session_id: str, emit=None):
        """把能力 id 解析为 LangChain 工具；不可用返回 None。"""
        cap = self.get(cap_id)
        if cap is None or cap.get("availability") != "available":
            return None
        if cap_id == "calculator":
            return calculator
        if cap_id == "time_now":
            return time_now
        if cap_id == "web_search":
            return web_search
        if cap_id == "rag":
            if self.corpus_store is None:
                return None
            return make_rag_tool(self.corpus_store, self.settings.rag_top_k, emit)
        if cap_id == "memory":
            if self.embeddings is None:
                return None
            store = self.sessions.long_memory(session_id, self.embeddings)
            return make_memory_tools(store, self.settings.rag_top_k, emit)
        return self.mcp.tool(cap_id)
