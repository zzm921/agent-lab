"""聊天 API：能力目录、SSE 流式对话、HITL 审批、源码展示、健康检查。"""
import json
from pathlib import Path

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.agents.runner import AgentRunner
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.config import settings
from app.core.errors import ConfigError
from app.llm.client import create_chat_model, create_embeddings
from app.memory.corpus import KNOWLEDGE_CORPUS
from app.memory.session_store import SessionStore
from app.memory.vector_store import VectorStore
from app.schemas import ApproveRequest, StopRequest, StreamRequest

router = APIRouter(prefix="/api", tags=["chat"])


def _sse(events):
    """把事件生成器转换为 SSE 响应（EventSourceResponse 负责 data: 格式化）。"""

    async def gen():
        async for ev in events:
            yield {"data": json.dumps(ev, ensure_ascii=False)}

    return EventSourceResponse(gen())

# 运行时单例：registry/runner 分开构建，能力目录不依赖大模型 Key
_RUNTIME: dict = {"sessions": None, "registry": None, "runner": None}


def _build_embeddings_and_corpus():
    """构建 Embedding 与知识库向量库；未配 Embedding Key 时二者均为 None。"""
    try:
        embeddings = create_embeddings(fake=False)
    except ConfigError:
        return None, None
    corpus = VectorStore(embeddings, name="knowledge")
    for text in KNOWLEDGE_CORPUS:
        corpus.add(text, {"source": "builtin"})
    return embeddings, corpus


def _build_registry(sessions):
    embeddings, corpus = _build_embeddings_and_corpus()
    mcp = McpManager(settings.mcp_servers)
    return CapabilityRegistry(settings, sessions, mcp, corpus, embeddings)


def get_sessions() -> SessionStore:
    if _RUNTIME["sessions"] is None:
        _RUNTIME["sessions"] = SessionStore()
    return _RUNTIME["sessions"]


def get_registry() -> CapabilityRegistry:
    if _RUNTIME["registry"] is None:
        _RUNTIME["registry"] = _build_registry(get_sessions())
    return _RUNTIME["registry"]


def get_runner() -> AgentRunner:
    if _RUNTIME["runner"] is None:
        llm = create_chat_model(fake=False)  # 未配百炼 API Key 时抛 ConfigError
        _RUNTIME["runner"] = AgentRunner(settings, llm, get_registry(), get_sessions())
    return _RUNTIME["runner"]


def set_runtime(sessions=None, registry=None, runner=None) -> None:
    """测试注入 fake 运行时。"""
    if sessions is not None:
        _RUNTIME["sessions"] = sessions
    if registry is not None:
        _RUNTIME["registry"] = registry
    if runner is not None:
        _RUNTIME["runner"] = runner


@router.get("/capabilities")
async def list_capabilities():
    """能力目录：内置 + MCP 发现，含可用性与不适配原因。"""
    registry = get_registry()
    await registry.refresh()
    return {"capabilities": registry.list()}


@router.post("/stream")
async def chat_stream(req: StreamRequest):
    """SSE 流式对话：思考/行动/观察/计划/反思/工具/审批/完成事件。"""
    runner = get_runner()
    session_id = req.session_id or get_sessions().create()
    events = runner.stream(
        session_id,
        req.message,
        req.mode,
        req.enabled_capabilities,
        req.prompt_strategy,
        req.approval_policy,
    )
    return _sse(events)


@router.post("/approve")
async def approve(req: ApproveRequest):
    """HITL 审批：批准/拒绝/修改工具参数后恢复执行。"""
    runner = get_runner()
    events = runner.resume(req.approval_id, req.decision, req.modified_args)
    return _sse(events)


@router.post("/stop")
async def stop_run(req: StopRequest):
    """停止指定会话的后端执行：立即取消后台图任务，避免继续消耗 token。"""
    get_runner().stop(req.session_id)
    return {"ok": True}


_SOURCE_FILES: dict[str, str] = {
    "calculator": "tools/calculator.py",
    "time_now": "tools/time_now.py",
    "web_search": "tools/web_search.py",
    "run_command": "tools/run_command.py",
    "rag": "tools/rag_tool.py",
    "memory": "tools/memory_tool.py",
    "mcp": "capabilities/mcp.py",
    "registry": "capabilities/registry.py",
    "react": "agents/modes/react.py",
    "plan_execute": "agents/modes/plan_execute.py",
    "reflection": "agents/modes/reflection.py",
    "multi_agent": "agents/modes/multi_agent.py",
    "runner": "agents/runner.py",
    "harness": "agents/harness.py",
}


@router.get("/source/{module}")
async def source(module: str):
    """返回后端真实源码供前端代码展示，杜绝展示与运行不一致。"""
    rel = _SOURCE_FILES.get(module)
    if rel is None:
        return {"module": module, "content": ""}
    path = Path(__file__).resolve().parents[1] / rel
    return {"module": module, "content": path.read_text(encoding="utf-8") if path.exists() else ""}


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model": settings.chat_model,
        "mcp_configured": settings.mcp_servers.strip() not in ("", "{}"),
        "embedding_configured": bool(settings.embedding_api_key),
    }
