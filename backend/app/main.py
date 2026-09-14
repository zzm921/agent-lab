"""FastAPI 应用入口：CORS、路由、异常处理、前端静态托管。"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import get_registry, get_sessions, router as chat_router
from app.api.content import router as content_router
from app.api.memory import router as memory_router
from app.api.sandbox import router as sandbox_router
from app.api.sessions import router as sessions_router
from app.api.telemetry import router as telemetry_router
from app.config import settings
from app.core.errors import ConfigError

# 统一日志配置：root logger 输出到控制台（uvicorn 不配置 root，应用模块的
# logger.info 默认会被丢弃）。级别由环境变量 LOG_LEVEL / .env 控制，默认 INFO。
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

_log = logging.getLogger("app.main")

# 后台会话清理任务（每日 TTL 扫描）句柄，lifespan 退出时取消
_GC_TASK: asyncio.Task | None = None


async def _session_gc_loop() -> None:
    """每日执行一次会话 TTL 清理；异常吞掉避免任务退出。"""
    while True:
        try:
            await asyncio.sleep(86400)  # 先睡一天，启动时的首次清理已由 lifespan 完成
            await get_sessions().gc_ttl()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            _log.warning("会话 TTL 清理任务异常（下轮重试）: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _GC_TASK
    # 服务启动即就绪：构建能力注册表（RAG 预建数据直接可用）。
    # MCP 默认开启：启动时即连接并发现（stdio 拉起 mcp-info），页面开关只控制能力是否入目录。
    # 失败不阻断服务（如未配 Embedding Key / MCP server 不可达），能力目录仍可正常访问。
    try:
        registry = get_registry()
        if registry.mcp.enabled:
            await registry.refresh()
    except Exception:  # noqa: BLE001
        pass
    # 会话持久化：把内存 checkpointer 切换为文件型 SQLite + 启动时清理过期会话（>TTL 天未活跃）
    try:
        sessions = get_sessions()
        await sessions.init_checkpointer()
        await sessions.gc_ttl()
        if settings.checkpoint_ttl_days > 0:
            _GC_TASK = asyncio.create_task(_session_gc_loop())
    except Exception as exc:  # noqa: BLE001 — 会话持久化失败不阻断服务
        _log.warning("会话持久化初始化失败（回退内存）: %s", exc)
    yield
    if _GC_TASK is not None:
        _GC_TASK.cancel()
        try:
            await _GC_TASK
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(title="AI Agent 平台", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat_router)
    app.include_router(sandbox_router)
    app.include_router(content_router)
    app.include_router(memory_router)
    app.include_router(telemetry_router)
    app.include_router(sessions_router)

    @app.exception_handler(ConfigError)
    async def config_error_handler(_request: Request, exc: ConfigError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.exists():
        # 静态资源（Vite 构建产物带 hash，按文件路径直接提供）
        app.mount(
            "/assets",
            StaticFiles(directory=str(frontend_dist / "assets")),
            name="assets",
        )

        # SPA 兜底：非 /api 路径直接刷新（如 /lab）时返回 index.html，
        # 由前端路由接管；dist 下真实存在的文件（如 favicon）仍按原文件返回。
        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            candidate = frontend_dist / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

    return app


app = create_app()
