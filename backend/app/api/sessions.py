"""会话（多会话）管理 API：新建 / 列表 / 重命名 / 删除。

按客户端隔离：会话归属 client_key（设备指纹 X-Client-Id 优先、IP 兜底），
每个试用者只看到/管理自己的会话；删除会话联动清理其 checkpoint 线程与会话记忆库。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.chat import _client_key, get_sessions

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions(request: Request):
    """当前客户端的会话列表：按最后活跃倒序，含标题/消息数/创建与活跃时间。"""
    sessions = get_sessions()
    return {"sessions": sessions.list(client_key=_client_key(request))}


@router.post("")
async def create_session(request: Request):
    """新建会话：后端分配 session_id 并登记归属（客户端隔离）。"""
    sessions = get_sessions()
    session_id = sessions.create(client_key=_client_key(request))
    return {"session_id": session_id, "session": sessions.get(session_id)}


@router.patch("/{session_id}")
async def rename_session(session_id: str, request: Request):
    """重命名会话（标题为空则清空，等待下一条消息重新生成）。"""
    body = await request.json()
    title = str(body.get("title") or "").strip()
    sessions = get_sessions()
    rec = sessions.get(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if rec.get("client_key") != _client_key(request):
        raise HTTPException(status_code=403, detail="无权操作他人会话")
    return {"session": sessions.rename(session_id, title)}


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request):
    """删除会话：移除元数据 + checkpoint 线程 + 会话记忆文件。"""
    sessions = get_sessions()
    rec = sessions.get(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if rec.get("client_key") != _client_key(request):
        raise HTTPException(status_code=403, detail="无权操作他人会话")
    await sessions.delete(session_id)
    return {"ok": True}


@router.get("/{session_id}/events")
async def session_events(session_id: str, request: Request):
    """会话历史事件流：切换会话时前端按该事件流重放，UI 与直播时完全一致。"""
    sessions = get_sessions()
    rec = sessions.get(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if rec.get("client_key") != _client_key(request):
        raise HTTPException(status_code=403, detail="无权查看他人会话")
    return {"session_id": session_id, "events": sessions.load_events(session_id)}
