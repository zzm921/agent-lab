"""长期记忆管理 API：查看（会话/全局）、手动写入、删除（用户掌控权）、记忆梦游。

常驻（全局）记忆按客户端隔离：scope=global 时由服务端从请求判定 client_key
（设备指纹 X-Client-Id 优先、IP 兜底），每个试用者只读写自己的常驻库，无法越权访问他人。
「记忆梦游」是手动触发的显式巩固：取本会话全部对话 → 提取 → 匹配 → 裁决 → 落库，
返回结构化中间结果报告（每步结果可见）。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.chat import _client_key, get_registry, get_runner, get_sessions
from app.memory.consolidate import dream_consolidate, dream_tidy
from app.schemas import MemoryWriteRequest
from app.telemetry.sink import ACTIVE_SINK

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


@router.post("/dream")
async def memory_dream(request: Request, session_id: str = "", mode: str = "extract"):
    """手动触发「记忆梦游」：

    - mode=extract（默认）：对本会话完整对话执行一次提取→匹配→裁决→落库，
      返回中间结果报告（对话 → 记忆）；
    - mode=tidy：读取现有记忆库（会话 + 常驻）全部条目，LLM 给出 merge/replace/pattern
      整理建议并直接落库，返回整理报告（对齐 Claude Dreaming：现有记忆整理，不改原库输入快照，
      落库后给出前后对比）。

    消息来自 LangGraph 检查点（sessions.checkpointer），无 embedding 或会话无对话时返回 400。
    """
    registry = get_registry()
    if registry.embeddings is None:
        raise HTTPException(status_code=400, detail="未配置 Embedding API Key，记忆能力不可用")
    runner = get_runner()
    sessions = get_sessions()
    store = registry.sessions.long_memory(session_id, registry.embeddings)
    constant = registry.sessions.constant_memory(registry.embeddings, _client_key(request))

    # 整理模式不依赖会话对话，直接读库整理（无需 checkpointer）
    if mode == "tidy":
        # 手动梦游属显式副作用：摘除 ACTIVE_SINK，避免其 LLM 调用污染运行记录
        token = ACTIVE_SINK.set(None)
        try:
            report = await dream_tidy(store, constant, runner._scenario_llm("memory_consolidate"), runner.settings, session_id)
        finally:
            ACTIVE_SINK.reset(token)
        return {"session_id": session_id, "mode": "tidy", **report}

    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    try:
        snap = await sessions.checkpointer.aget_tuple({"configurable": {"thread_id": session_id}})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法读取会话对话记录：{exc}") from exc
    msgs: list = []
    if snap is not None:
        channel_values = (snap.checkpoint or {}).get("channel_values") or {}
        msgs = channel_values.get("messages") or []
    if not msgs:
        raise HTTPException(status_code=400, detail="会话暂无对话记录，先聊几句再梦游")
    # 手动梦游属显式副作用：摘除 ACTIVE_SINK，避免其 LLM 调用污染运行记录
    token = ACTIVE_SINK.set(None)
    try:
        report = await dream_consolidate(store, constant, msgs, runner._scenario_llm("memory_consolidate"), runner.settings, session_id)
    finally:
        ACTIVE_SINK.reset(token)
    return {"session_id": session_id, "mode": "extract", **report}


@router.delete("/{mem_id}")
async def delete_memory(mem_id: str, request: Request, session_id: str = "", scope: str = "session"):
    """删除一条记忆（用户掌控权）；不存在返回 404。"""
    store = _store(scope, session_id, _client_key(request))
    if not store.delete(mem_id):
        raise HTTPException(status_code=404, detail=f"记忆不存在：{mem_id}")
    return {"ok": True, "id": mem_id}
