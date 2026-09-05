"""长期记忆管理 API：查看（会话/全局）、手动写入、删除（用户掌控权）。

常驻（全局）记忆按客户端隔离：scope=global 时由服务端从请求判定 client_key
（设备指纹 X-Client-Id 优先、IP 兜底），每个试用者只读写自己的常驻库，无法越权访问他人。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.chat import _client_key, get_registry, get_sessions
from app.schemas import MemoryWriteRequest

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _store(scope: str, session_id: str, client_key: str = "default"):
    """按 scope 取会话库或当前客户端的常驻库；无 embedding 时记忆能力不可用。"""
    registry = get_registry()
    if registry.embeddings is None:
        raise HTTPException(status_code=400, detail="未配置 Embedding API Key，记忆能力不可用")
    if scope == "global":
        return registry.sessions.constant_memory(registry.embeddings, client_key)
    return registry.sessions.long_memory(session_id or "", registry.embeddings)


@router.get("")
async def list_memories(request: Request, session_id: str = "", kind: str | None = None, scope: str = "session"):
    """记忆列表：scope=session 返回会话记忆，scope=global 返回当前客户端的常驻记忆；可按 kind 过滤。"""
    store = _store(scope, session_id, _client_key(request))
    items = []
    for it in store.list(kind=kind or None):
        items.append(
            {
                "id": it.get("id"),
                "kind": it.get("kind"),
                "text": it.get("text"),
                "importance": it.get("importance"),
                "created_at": it.get("created_at"),
                "last_access_at": it.get("last_access_at"),
                "access_count": it.get("access_count"),
                "scope": scope,
            }
        )
    return {"items": items, "scope": scope}


@router.post("")
async def write_memory(req: MemoryWriteRequest, request: Request):
    """手动写入一条记忆（面板演示用）：scope 决定写会话库还是当前客户端的常驻库。"""
    store = _store(req.scope, req.session_id, _client_key(request))
    result = store.add(req.text, kind=req.kind, importance=req.importance)
    return {"ok": True, "id": result["id"], "action": result["action"]}


@router.get("/audit")
async def memory_audit(scope: str = "", limit: int = 50):
    """记忆操作审计流水（新增/更新/删除，按时间倒序）；scope 可按 session/global 过滤。"""
    store = _store("session", "", "default")
    items = store.list_audit(limit=max(1, min(limit, 200)), scope=scope or None)
    return {"items": items, "scope": scope or "all"}


@router.delete("/{mem_id}")
async def delete_memory(mem_id: str, request: Request, session_id: str = "", scope: str = "session"):
    """删除一条记忆（用户掌控权）；不存在返回 404。"""
    store = _store(scope, session_id, _client_key(request))
    if not store.delete(mem_id):
        raise HTTPException(status_code=404, detail=f"记忆不存在：{mem_id}")
    return {"ok": True, "id": mem_id}
