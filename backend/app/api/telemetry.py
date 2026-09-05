"""运行记录（telemetry）API：当前客户端的 run 列表 + 详情，供前端「运行记录」面板查看与回放。

每条 run = 一次 stream（或 stream+resume 合并）的完整 SSE 事件流 + LLM 调用明细 + 聚合统计；
按 client_key 隔离（设备指纹优先、IP 兜底，与记忆隔离同源），只能看到本人的 run。
telemetry_enabled 关闭时列表返回空、详情返回 404。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.chat import _client_key
from app.telemetry.store import get_run_store

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


@router.get("/runs")
async def list_runs(request: Request, session_id: str = "", limit: int = 50):
    """当前客户端的运行记录列表（最新在前）；可按会话过滤。"""
    store = get_run_store()
    if store is None:
        return {"runs": [], "enabled": False}
    runs = store.list(
        _client_key(request),
        session_id or None,
        limit=max(1, min(limit, 200)),
    )
    return {"runs": runs, "enabled": True}


@router.get("/runs/{run_id}")
async def run_detail(run_id: str, request: Request):
    """一次 run 的完整记录 {meta, events}；不存在或非本人返回 404。"""
    store = get_run_store()
    if store is None:
        raise HTTPException(status_code=404, detail="运行记录功能未开启")
    doc = store.get(run_id, _client_key(request))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"运行记录不存在：{run_id}")
    return doc
