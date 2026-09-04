"""长期记忆工具：写入（分类/重要性/去重）/ 召回（规范化注入块 + 老化提示）。

注入块格式对齐 3.5 Prompt 规范：`[i] (kind·重要度x) text —— 记录于 date`
+ 兜底句「若与本次说明矛盾，以本次说明为准」，命中 > 老化天数附「可能过时」提示。
"""
from __future__ import annotations

import time

from langchain_core.tools import tool

from app.memory.long_memory import MEMORY_KINDS


def _fmt_date(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "?"


def _injection_block(hits: list[dict]) -> str:
    """把命中记录组装为规范化注入块（memory_recall 返回给模型的文本）。"""
    lines = [
        f"[{i}] ({meta.get('kind', 'fact')}·重要度{meta.get('importance', 0):.1f}) "
        f"{h['text']} —— 记录于 {_fmt_date(meta.get('created_at'))}"
        for i, (h, meta) in enumerate(
            ((h, h.get("metadata") or {}) for h in hits),
            start=1,
        )
    ]
    return (
        "【长期记忆检索结果】\n"
        + "\n".join(lines)
        + "\n请优先参考以上记忆回答；若与本次说明矛盾，以本次说明为准。"
    )


def _stale_hint(hits: list[dict], old_days: int) -> str:
    """命中超过 old_days 天未更新的记忆时，追加老化验证提示（不删旧记忆，强制先验证）。"""
    now = time.time()
    if any(
        (meta.get("created_at") or 0) and (now - meta["created_at"]) > old_days * 86400
        for h in hits
        for meta in [h.get("metadata") or {}]
    ):
        return (
            f"\n注意：以下记忆来自 {old_days} 天以上，可能已过时。"
            "使用前请与当前实际情况核对，确认仍适用再采用。"
        )
    return ""


def make_memory_tools(store, constant_store, settings, emit=None):
    """构建长期记忆读写工具；提供 emit 时可推送 memory_write/memory_read 事件。"""
    top_k = settings.memory_top_k
    threshold = settings.memory_threshold
    old_days_hint = settings.memory_old_days_hint

    @tool
    def memory_write(fact: str, kind: str = "fact", importance: float = 0.5, scope: str = "session") -> str:
        """把一条重要事实写入长期记忆，供后续对话回忆。

        kind: fact(事实) | preference(偏好) | episodic(过往事件) | procedural(操作经验)
        scope: session(写入当前会话记忆) | global(写入全局常驻记忆，会话启动即注入 system)
        """
        if kind not in MEMORY_KINDS:
            kind = "fact"
        if scope not in ("session", "global"):
            scope = "session"
        target = constant_store if scope == "global" else store
        target.add(fact, kind=kind, importance=importance)
        if emit is not None:
            emit(
                {
                    "type": "memory_write",
                    "content": fact,
                    "kind": kind,
                    "importance": importance,
                    "scope": scope,
                }
            )
        return f"已记住（{kind}·重要度{importance:.1f}）：{fact}"

    @tool
    def memory_recall(query: str, kind: str | None = None) -> str:
        """从长期记忆中检索与查询相关的事实（支持按 kind 过滤）。

        命中超过 memory_old_days_hint 天的记忆会附带「可能已过时，使用前核对」提示。
        """
        hits = store.search(query, top_k=top_k, threshold=threshold, kind=kind)
        if emit is not None:
            emit(
                {
                    "type": "memory_read",
                    "query": query,
                    "hits": [{"text": h["text"], "score": h["score"]} for h in hits],
                }
            )
        if not hits:
            return "长期记忆中没有相关记录。"
        return _injection_block(hits) + _stale_hint(hits, old_days_hint)

    return [memory_write, memory_recall]
