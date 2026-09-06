"""在线评测采样（线上评估闭环 P0）：把真实对话的轻量样本落盘 JSONL。

与 telemetry 运行记录（完整事件流、TTL 7 天、供回放排障）互补：样本是面向
「回流评测集」的轻量结构化记录（query / 命中 / 答案 / 耗时 / 用户反馈），
按日分文件长期保留，供 scripts/eval_online.py 筛选回流为评测用例。

- 采集点：runner.stream() 收口落盘后异步追加（asyncio.to_thread，不阻塞 SSE）；
- 反馈回填：POST /api/feedback 按 session_id + query 匹配样本行回填 vote；
- 仅当本轮存在检索命中（retrieved_ids 非空）才落样本（寒暄 / 未配 Key 跳过）；
- 任何采集/回填失败静默吞掉，绝不影响主对话链路。
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.version import get_app_version


def _sample_dir() -> Path:
    p = Path(settings.eval_sample_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sample_path() -> Path:
    return _sample_dir() / f"online_{time.strftime('%Y%m%d')}.jsonl"


def _extract_answer(events: list[dict[str, Any]]) -> str:
    """从事件流提取最终回答：最后一条合并后的 message 文本（审批/异常时可空）。"""
    text = ""
    for ev in events:
        if ev.get("type") == "message":
            text = str(ev.get("text") or ev.get("delta") or "")
    return text


def _tool_sequence(events: list[dict[str, Any]]) -> list[str]:
    """从事件流提取实际工具轨迹：成功的 tool_end 工具名，保序去重（同工具反复调用只计一次）。

    供 agent 域回流：线上真实轨迹还原为评测用例的 must_call（行为回归依据）。
    """
    seen: set[str] = set()
    seq: list[str] = []
    for ev in events:
        if ev.get("type") != "tool_end" or not ev.get("success"):
            continue
        t = str(ev.get("tool") or "").strip()
        if t and t not in seen:
            seen.add(t)
            seq.append(t)
    return seq


def build_sample_row(
    session_id: str,
    client_key: str,
    meta: dict[str, Any],
    events: list[dict[str, Any]],
    rag_signals: dict[str, Any] | None,
) -> dict[str, Any]:
    """从一轮 run 收口后的 meta/events 构造轻量样本行（供 JSONL 追加）。

    capability：rag（有检索命中）/ agent（无命中，纯工具或寒暄）——供回流脚本分流；
    tool_sequence：实际工具轨迹（agent 域回流还原 must_call 依据）。
    """
    signals = rag_signals or {}
    retrieved_ids = signals.get("retrieved_ids") or []
    tool_seq = _tool_sequence(events)
    return {
        "sample_id": uuid.uuid4().hex,
        "session_id": session_id,
        "client_key": client_key,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": get_app_version(),
        "query": meta.get("message") or "",
        "effective_query": signals.get("effective_message") or "",
        "capability": "rag" if retrieved_ids else "agent",
        "tool_sequence": tool_seq,
        "mode": meta.get("mode") or "",
        "rag_scheme": meta.get("rag_scheme") or "",
        "rag_enabled": bool(meta.get("rag_enabled")),
        "complexity": signals.get("complexity"),
        "retrieval_mode": signals.get("retrieval_mode"),
        "generation_mode": signals.get("generation_mode"),
        "insufficient": bool(signals.get("insufficient")),
        "retrieved_ids": retrieved_ids,
        "answer": _extract_answer(events),
        "elapsed_ms": meta.get("duration_ms"),
        "status": meta.get("status") or "",
        "stats": meta.get("stats") or {},
        "vote": None,
        "reason": None,
    }


def append_online_sample(
    session_id: str,
    client_key: str,
    meta: dict[str, Any],
    events: list[dict[str, Any]],
    rag_signals: dict[str, Any] | None,
) -> None:
    """落盘一行样本。有 RAG 命中或有工具调用才落（寒暄/纯问答跳过）；失败静默。"""
    try:
        row = build_sample_row(session_id, client_key, meta, events, rag_signals)
        if not row["retrieved_ids"] and not row["tool_sequence"]:
            return  # 无检索命中且无工具调用（寒暄 / 未配 Key / rag 未启用且未走工具）不落样本
        with _sample_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — 采样失败绝不影响主流程
        pass


def _load_today_rows() -> list[dict[str, Any]]:
    path = _sample_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def apply_feedback(session_id: str, query: str, vote: str, reason: str) -> tuple[bool, bool]:
    """按 session_id + query 匹配当日样本行，回填 vote / reason。

    返回 (matched, updated)：matched=是否找到样本行；updated=是否实际发生写入。
    幂等：对已回填行仅更新 reason（vote 保持首次选择），避免同一轮被反复覆盖。
    """
    rows = _load_today_rows()
    if not rows:
        return False, False
    matched = False
    updated = False
    norm_reason = reason or None  # 空 reason 统一为 None，保证重复提交幂等（None 与 "" 不误判为变更）
    for row in rows:
        if row.get("session_id") == session_id and row.get("query") == query:
            matched = True
            if row.get("vote") != vote or row.get("reason") != norm_reason:
                row["vote"] = vote
                row["reason"] = norm_reason
                updated = True
    if updated:
        with _sample_path().open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return matched, updated


def append_orphan_feedback(session_id: str, query: str, vote: str, reason: str) -> None:
    """未匹配到样本行的反馈落盘为 orphan（供排查：样本 TTL 清理 / 未采集 / 篡改）。"""
    try:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": session_id,
            "query": query,
            "vote": vote,
            "reason": reason or None,
        }
        with (_sample_dir() / "orphan_feedback.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass
