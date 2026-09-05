"""聊天 API：能力目录、SSE 流式对话、HITL 审批、源码展示、健康检查。"""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.agents.runner import AgentRunner
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.config import settings
from app.core.errors import ConfigError
from app.core.rate_limit import DailyQuota
from app.llm.client import create_embeddings, llm_service
from app.memory.session_store import SessionStore
from app.rag.manager import RagManager
from app.schemas import ApproveRequest, FaultRequest, McpToggleRequest, StopRequest, StreamRequest

router = APIRouter(prefix="/api", tags=["chat"])


def _sse(events):
    """把事件生成器转换为 SSE 响应（EventSourceResponse 负责 data: 格式化）。"""

    async def gen():
        async for ev in events:
            yield {"data": json.dumps(ev, ensure_ascii=False)}

    return EventSourceResponse(gen())

# 运行时单例：registry/runner 分开构建，能力目录不依赖大模型 Key
_RUNTIME: dict = {"sessions": None, "registry": None, "runner": None}
_QUOTA: DailyQuota | None = None


def _build_embeddings_and_rag():
    """构建 Embedding 与多 RAG 方案管理器；未配 Embedding Key 时二者均为 None。

    向量库数据在线上前由 scripts/ingest_naive.py / scripts/ingest_advanced.py 预建，
    本函数只构建各方案的 store（加载已建好的集合），不做现场入库，保证能力加载快速。
    rag_enabled（默认开启）控制是否构建 RagManager：关闭时不构建，知识库检索整体关闭，
    仅保留 Embedding（长期记忆能力仍可用）；开启时构建 RAG。
    方案内部需要 LLM 的阶段（Query 重写/路由/分解/多跳规划验证）按命名场景从全局
    LLMService 懒取模型；未配聊天 Key 时自动回退确定性规则实现（仅 Embedding 也能跑通）。
    """
    try:
        embeddings = create_embeddings(fake=False)
    except ConfigError:
        return None, None
    if not settings.rag_enabled:
        return embeddings, None
    rag = RagManager(settings, embeddings, top_k=settings.rag_top_k)
    return embeddings, rag


def _build_registry(sessions):
    embeddings, rag = _build_embeddings_and_rag()
    mcp = McpManager(settings.mcp_servers, enabled=settings.mcp_enabled)
    return CapabilityRegistry(settings, sessions, mcp, rag, embeddings)


def get_sessions() -> SessionStore:
    if _RUNTIME["sessions"] is None:
        _RUNTIME["sessions"] = SessionStore(
            memory_dir=settings.memory_dir,
            top_k=settings.memory_top_k,
            threshold=settings.memory_threshold,
            dedup_threshold=settings.memory_dedup_threshold,
            max_per_namespace=settings.memory_max_per_namespace,
            ttl_days=settings.memory_ttl_days,
        )
    return _RUNTIME["sessions"]


def get_registry() -> CapabilityRegistry:
    if _RUNTIME["registry"] is None:
        _RUNTIME["registry"] = _build_registry(get_sessions())
    return _RUNTIME["registry"]


def get_runner() -> AgentRunner:
    if _RUNTIME["runner"] is None:
        # 传入全局 LLMService：runner 按模式场景（chat/planner/critic）取不同模型与参数
        _RUNTIME["runner"] = AgentRunner(settings, llm_service, get_registry(), get_sessions())
    return _RUNTIME["runner"]


def set_runtime(sessions=None, registry=None, runner=None) -> None:
    """测试注入 fake 运行时。"""
    if sessions is not None:
        _RUNTIME["sessions"] = sessions
    if registry is not None:
        _RUNTIME["registry"] = registry
    if runner is not None:
        _RUNTIME["runner"] = runner


def set_quota(quota: DailyQuota | None) -> None:
    """测试注入/重置每日配额实例（None 表示按配置懒加载）。"""
    global _QUOTA
    _QUOTA = quota


def get_quota() -> DailyQuota:
    """获取（必要时按配置懒创建）每日配额实例。"""
    global _QUOTA
    if _QUOTA is None:
        _QUOTA = DailyQuota(
            limit=settings.quota_daily_limit,
            path=settings.quota_store_path or None,
        )
    return _QUOTA


def _client_ip(request: Request) -> str:
    """获取客户端真实 IP：部署在反向代理后时优先信任转发头。"""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real = request.headers.get("x-real-ip", "")
    if real:
        return real.strip()
    return request.client.host if request.client else "unknown"


def _client_key(request: Request) -> str:
    """客户端标识：优先设备指纹（X-Client-Id，精确区分「一台电脑」），否则退回 IP。"""
    client_id = request.headers.get("x-client-id", "").strip()
    if client_id:
        return f"cid:{client_id}"
    return f"ip:{_client_ip(request)}"


@router.get("/capabilities")
async def list_capabilities():
    """能力目录：内置 + MCP 发现，含可用性与不适配原因。"""
    registry = get_registry()
    await registry.refresh()
    return {"capabilities": registry.list()}


@router.get("/mcp")
async def mcp_status():
    """MCP 开关状态：enabled（是否在能力目录中使用）+ 已注册 server + 当前暴露的能力。"""
    mcp = get_registry().mcp
    return {
        "enabled": mcp.enabled,
        "servers": list(mcp.servers.keys()),
        "capabilities": mcp.capabilities if mcp.enabled else [],
    }


@router.post("/mcp")
async def mcp_toggle(req: McpToggleRequest):
    """页面点选开启/关闭 MCP 能力：仅控制能力是否进入目录（服务连接在启动时已建立）。"""
    mcp = get_registry().mcp
    if req.enabled and not mcp.enabled:
        await mcp.enable()
    elif not req.enabled and mcp.enabled:
        mcp.disable()
    return {"enabled": mcp.enabled, "capabilities": mcp.capabilities if mcp.enabled else []}


@router.post("/stream")
async def chat_stream(req: StreamRequest, request: Request):
    """SSE 流式对话：思考/行动/观察/计划/反思/工具/审批/完成事件。

    每次调用消耗一次「每日对话配额」（按设备指纹或 IP 计数），
    超过每日上限（默认 20 次）返回 429，防止单台设备/IP 滥用。
    """
    if settings.quota_enabled:
        quota = get_quota()
        key = _client_key(request)
        allowed, _ = quota.try_consume(key)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"今日对话次数已达上限（{quota.limit} 次），请明天再试",
            )
    runner = get_runner()
    session_id = req.session_id or get_sessions().create()
    events = runner.stream(
        session_id,
        req.message,
        req.mode,
        req.enabled_capabilities,
        req.prompt_strategy,
        req.approval_policy,
        rag_scheme=req.rag_scheme,
        rag_enabled=req.rag_enabled,
        memory_enabled=req.memory_enabled,
        context_keep_rounds=req.context_keep_rounds,
        # 常驻记忆按客户端隔离：设备指纹优先、IP 兜底（同一台电脑/同一 IP 各一份记忆）
        client_key=_client_key(request),
    )
    return _sse(events)


@router.get("/rag/schemes")
async def rag_schemes():
    """可选的 RAG 方案目录：id/名称/描述/集合名/语料条数，供前端渲染方案选择器。"""
    return {"schemes": get_registry().rag_schemes()}


@router.get("/quota")
async def quota_info(request: Request):
    """当前客户端今日对话配额使用情况（前端可据此提示剩余次数）。"""
    quota = get_quota()
    key = _client_key(request)
    return {
        "enabled": settings.quota_enabled,
        "limit": quota.limit,
        "remaining": quota.remaining(key) if settings.quota_enabled else quota.limit,
    }


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


@router.get("/faults")
async def list_faults():
    """当前故障注入配置（验证两层重试/熔断机制用）。"""
    return {"faults": get_runner().harness.faults()}


@router.get("/faults/types")
async def list_fault_types():
    """可用故障注入类型及重试分类：retryable=瞬时错误（工具层直接重试），
    permanent=参数/业务错误（返回给模型思考后重试）。"""
    return {"types": get_runner().harness.available_fault_modes()}


@router.post("/fault")
async def set_fault(req: FaultRequest):
    """设置工具故障注入。类型分类决定重试走向：
    - 瞬时错误（timeout/conn_reset/dns/http_429/http_5xx）→ 工具层直接重试（透明重试）
    - 参数/业务错误（error/business/http_400/http_401/http_403/http_404）→ 返回给模型思考后重试
    off/none 恢复正常；未知类型返回 400。"""
    try:
        get_runner().harness.set_fault(req.tool, req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "faults": get_runner().harness.faults()}


_SOURCE_FILES: dict[str, str] = {
    "calculator": "tools/calculator.py",
    "time_now": "tools/time_now.py",
    "web_search": "tools/web_search.py",
    "run_command": "tools/run_command.py",
    "big_output": "tools/big_output.py",
    "rag": "rag/manager.py",
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
