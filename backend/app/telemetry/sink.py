"""TelemetrySink：一次运行（run）的事件收集器。

在 runner.stream / runner.resume 的 yield 边界挂载（不侵入业务）：每个下发前端的
SSE 事件先经 observe() 落进本次 run 记录，close() 时聚合统计并交给 RunStore 持久化。

职责：
- 增量事件合并：thinking / message / revise / critique 是 6 字符 delta 切块，
  合并为完整文本再入库，回放数据才可读；
- LLM 调用明细：LoggedChatModel 通过本模块的 ACTIVE_SINK 上下文变量定位当前 run，
  调用 record_llm() 记录 scenario / 模型 / 时延 / token / 成败；
- 聚合统计：close() 时计算工具调用、LLM token 与成本、RAG/记忆/护栏计数等元信息。
"""
from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Any

# 当前请求（run）的活动 sink：LoggedChatModel 据此上报每次 LLM 调用。
# contextvars 在 asyncio.to_thread / create_task 中自动传播，跨线程/子任务可读。
ACTIVE_SINK: ContextVar["TelemetrySink | None"] = ContextVar("active_telemetry_sink", default=None)

# 增量事件类型：SSE 中以 delta 分片下发，入库前须合并为完整文本
_DELTA_TYPES = ("thinking", "message", "revise", "critique")


class TelemetrySink:
    """一次运行的收集器：事件序列 + 聚合元信息，close() 时持久化。"""

    def __init__(
        self,
        store,
        session_id: str,
        client_key: str,
        meta: dict[str, Any],
        prices: dict[str, float] | None = None,
    ):
        self.store = store
        self.run_id = uuid.uuid4().hex
        self.session_id = session_id
        self.client_key = client_key
        self._meta_base = dict(meta)
        self._prices = prices or {"input": 0.0, "output": 0.0}
        self._events: list[dict[str, Any]] = []
        self._delta: dict[str, str] = {}
        self._seq = 0
        self._start = time.time()
        self._last_type = ""
        self._closed = False
        self._meta: dict[str, Any] = {}

    # ---- 生命周期 ----
    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def paused(self) -> bool:
        """是否停在审批请求（approval_request 为最后事件）：保留记录供 resume 续写。"""
        return self._last_type == "approval_request"

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    # ---- 事件收集 ----
    def _append(self, ev: dict[str, Any]) -> None:
        self._seq += 1
        self._events.append({"seq": self._seq, "ts": int(time.time() * 1000), **ev})

    def _flush_delta(self) -> None:
        """把积压的增量文本合并为完整事件写入序列（保持相对顺序）。"""
        for t, text in self._delta.items():
            if text:
                self._append({"type": t, "text": text})
        self._delta = {}

    def observe(self, data: dict[str, Any]) -> None:
        """记录一个下发给前端的 SSE 事件（含 meta/thinking/tool/rag/done 等全部类型）。"""
        if self._closed:
            return
        t = data.get("type") or ""
        if t in _DELTA_TYPES:
            self._delta[t] = self._delta.get(t, "") + str(data.get("delta", ""))
            self._last_type = t
            return
        self._flush_delta()
        self._append({k: v for k, v in data.items()})
        self._last_type = t

    def record_llm(self, call: dict[str, Any]) -> None:
        """记录一次 LLM 调用明细（LoggedChatModel 在 ACTIVE_SINK 存在时调用）。"""
        if self._closed:
            return
        self._flush_delta()
        self._append({"type": "llm_call", **call})

    # ---- 续写（HITL 审批 resume）----
    @classmethod
    def resume_from(cls, store, doc: dict[str, Any], prices: dict[str, float] | None = None) -> "TelemetrySink":
        """从既有 run 文档恢复一个续写 sink：沿用 run_id 与事件序列，close() 覆盖写回同一文件。

        供审批（HITL）场景：stream 遇 approval_request 已落盘 status=pending，
        resume 续写时保持同一 run_id，保证一轮对话只产生一条完整记录。
        """
        meta = doc.get("meta") or {}
        events = doc.get("events") or []
        reserved = {
            "run_id", "session_id", "client_key", "start_ts", "end_ts",
            "duration_ms", "status", "summary", "error", "stats",
        }
        base = {k: v for k, v in meta.items() if k not in reserved}
        sink = cls(
            store,
            str(meta.get("session_id", "")),
            str(meta.get("client_key", "")),
            base,
            prices,
        )
        sink.run_id = str(meta.get("run_id") or sink.run_id)
        sink._events = [dict(ev) for ev in events]
        sink._seq = len(sink._events)
        sink._last_type = str(sink._events[-1].get("type", "")) if sink._events else ""
        try:
            sink._start = time.mktime(
                time.strptime(str(meta.get("start_ts", "")), "%Y-%m-%d %H:%M:%S")
            )
        except (ValueError, TypeError, OverflowError):
            sink._start = time.time()
        return sink

    # ---- 收口 ----
    def close(self, status: str = "") -> dict[str, Any]:
        """收口本次运行：合并残余增量、聚合统计、持久化。幂等。"""
        if self._closed:
            return self._meta
        self._flush_delta()
        self._closed = True
        self._meta = self._build_meta(status)
        try:
            self.store.save(self)
        except Exception:  # noqa: BLE001 — 可观测性落盘失败不应影响主流程
            pass
        return self._meta

    def _build_meta(self, status: str) -> dict[str, Any]:
        duration_ms = int((time.time() - self._start) * 1000)
        tool_calls: dict[str, int] = {}
        tool_failures: dict[str, int] = {}
        retries = 0
        approvals = 0
        guards = 0
        offloads = 0
        rag_retrieves = 0
        rag_hits = 0
        memory_reads = 0
        memory_writes = 0
        llm_calls = 0
        tokens = {"input": 0, "output": 0, "total": 0}
        summary = ""
        error = ""
        for ev in self._events:
            t = ev.get("type")
            if t == "tool_start":
                tool = ev.get("tool", "")
                tool_calls[tool] = tool_calls.get(tool, 0) + 1
            elif t == "tool_end":
                tool = ev.get("tool", "")
                if ev.get("success") is False:
                    tool_failures[tool] = tool_failures.get(tool, 0) + 1
            elif t == "tool_retry":
                retries += 1
            elif t == "llm_call":
                llm_calls += 1
                tk = ev.get("tokens") or {}
                tokens["input"] += int(tk.get("input", 0) or 0)
                tokens["output"] += int(tk.get("output", 0) or 0)
                tokens["total"] += int(tk.get("total", 0) or 0)
            elif t == "retrieve":
                rag_retrieves += 1
                rag_hits += len(ev.get("hits") or [])
            elif t == "memory_read":
                memory_reads += 1
            elif t == "memory_write":
                memory_writes += 1
            elif t == "guard_refused":
                guards += 1
            elif t == "context" and ev.get("kind") == "offload":
                offloads += 1
            elif t == "approval_request":
                approvals += 1
            elif t == "done":
                summary = ev.get("summary") or ""
            elif t == "error":
                error = ev.get("message") or ev.get("detail") or ""
        if status:
            final_status = status
        elif self._last_type == "error":
            final_status = "error"
        elif self._last_type == "done":
            final_status = "done"
        else:
            final_status = "interrupted"
        cost = round(
            tokens["input"] / 1e6 * self._prices.get("input", 0.0)
            + tokens["output"] / 1e6 * self._prices.get("output", 0.0),
            4,
        )
        return {
            **self._meta_base,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "client_key": self.client_key,
            "start_ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._start)),
            "end_ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
            "duration_ms": duration_ms,
            "status": final_status,
            "summary": summary,
            "error": error,
            "stats": {
                "tool_calls": dict(sorted(tool_calls.items())),
                "tool_failures": dict(sorted(tool_failures.items())),
                "retries": retries,
                "approvals": approvals,
                "llm_calls": llm_calls,
                "tokens": tokens,
                "cost_yuan": cost,
                "rag_retrieves": rag_retrieves,
                "rag_hits": rag_hits,
                "memory_reads": memory_reads,
                "memory_writes": memory_writes,
                "guards": guards,
                "offloads": offloads,
            },
        }

    # ---- 序列化（供调试/日志） ----
    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "events": self._events}

    def __repr__(self) -> str:
        return f"<TelemetrySink run_id={self.run_id} status={self.meta.get('status')} events={len(self._events)}>"
