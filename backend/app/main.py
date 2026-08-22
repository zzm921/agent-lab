"""FastAPI 应用入口：CORS、路由、异常处理、前端静态托管。"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.content import router as content_router
from app.api.sandbox import router as sandbox_router
from app.config import settings
from app.core.errors import ConfigError


def create_app() -> FastAPI:
    app = FastAPI(title="AI Agent 平台", version="1.0.0")
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

    @app.exception_handler(ConfigError)
    async def config_error_handler(_request: Request, exc: ConfigError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    return app


app = create_app()
