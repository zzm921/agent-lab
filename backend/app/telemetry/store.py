"""RunStore：运行记录（telemetry）持久化与治理。

- 落盘：每 run 一个 JSON 文档 {meta, events}，原子写入（tmp + rename）；
- 治理：TTL 过期清理 + 全库最大数量 LRU（删最旧），防止存储膨胀；
- 隔离：按 client_key 过滤读写（设备指纹优先、IP 兜底，与记忆隔离同源），
  前端「运行记录」面板只能看到自己客户端的 run。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.telemetry.sink import TelemetrySink

_RUN_STORE: "RunStore | None" = None


class RunStore:
    def __init__(
        self,
        dir_path: str,
        ttl_days: int = 7,
        max_runs: int = 500,
        price_input_per_1m: float = 0.0,
        price_output_per_1m: float = 0.0,
    ):
        self._dir = Path(dir_path) / "runs"
        self._ttl_days = max(1, ttl_days)
        self._max_runs = max(1, max_runs)
        self._prices = {"input": price_input_per_1m, "output": price_output_per_1m}

    def new_run(self, session_id: str, client_key: str, meta: dict[str, Any]) -> TelemetrySink:
        """创建一次运行的收集器（未落盘，close() 时持久化）。"""
        return TelemetrySink(self, session_id, client_key, meta, self._prices)

    @property
    def prices(self) -> dict[str, float]:
        return self._prices

    def find_pending(self, session_id: str, client_key: str) -> dict[str, Any] | None:
        """返回该会话最近一条 status=pending 且属于该客户端的 run 文档（审批待续写）。"""
        for path in sorted(self._dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            meta = doc.get("meta") or {}
            if meta.get("client_key") != client_key:
                continue
            if meta.get("session_id") != session_id:
                continue
            if meta.get("status") == "pending":
                return doc
        return None

    def resume_run(self, session_id: str, client_key: str) -> TelemetrySink | None:
        """为该会话最近一条 pending 的 run 返回续写 sink（事件续接、同一 run_id）；无则 None。"""
        doc = self.find_pending(session_id, client_key)
        if doc is None:
            return None
        return TelemetrySink.resume_from(self, doc, self._prices)


    def save(self, sink: TelemetrySink) -> None:
        """把一次 run 写盘（原子替换），随后执行存储治理。"""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{sink.run_id}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        doc = {"meta": sink.meta, "events": sink.events}
        tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        self._prune()

    def list(self, client_key: str, session_id: str | None = None, limit: int = 50) -> list[dict]:
        """按客户端返回运行记录元信息（最新在前）；仅返回本人 client_key 的 run。"""
        self._prune()
        out: list[dict] = []
        for path in sorted(self._dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                meta = self._read_meta(path)
            except (OSError, ValueError):
                continue
            if meta.get("client_key") != client_key:
                continue
            if session_id and meta.get("session_id") != session_id:
                continue
            out.append(meta)
            if len(out) >= max(1, limit):
                break
        return out

    def get(self, run_id: str, client_key: str) -> dict[str, Any] | None:
        """返回一次 run 的完整记录 {meta, events}；不存在或非本人返回 None。"""
        path = self._dir / f"{run_id}.jsonl"
        if not path.exists():
            return None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (doc.get("meta") or {}).get("client_key") != client_key:
            return None
        return doc

    def count(self) -> int:
        return len(list(self._dir.glob("*.jsonl"))) if self._dir.exists() else 0

    def _read_meta(self, path: Path) -> dict[str, Any]:
        doc = json.loads(path.read_text(encoding="utf-8"))
        meta = doc.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError("bad meta")
        return meta

    def _prune(self) -> None:
        """TTL 过期清理 + 全库数量上限（LRU 删最旧）。写盘/列表时触发。"""
        if not self._dir.exists():
            return
        now = time.time()
        for path in self._dir.glob("*.jsonl"):
            try:
                if now - path.stat().st_mtime > self._ttl_days * 86400:
                    path.unlink(missing_ok=True)
            except OSError:
                pass
        paths = sorted(self._dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        while len(paths) > self._max_runs:
            try:
                paths[0].unlink(missing_ok=True)
            except OSError:
                pass
            paths = paths[1:]


def get_run_store() -> RunStore | None:
    """全局运行记录库（惰性构建）；telemetry 总开关关闭时返回 None。"""
    global _RUN_STORE
    if not getattr(settings, "telemetry_enabled", True):
        return None
    if _RUN_STORE is None:
        _RUN_STORE = RunStore(
            dir_path=settings.telemetry_dir,
            ttl_days=settings.telemetry_ttl_days,
            max_runs=settings.telemetry_max_runs,
            price_input_per_1m=settings.llm_price_input_per_1m,
            price_output_per_1m=settings.llm_price_output_per_1m,
        )
    return _RUN_STORE


def set_run_store(store: RunStore | None) -> None:
    """测试注入/重置运行记录库（None 恢复懒加载）。"""
    global _RUN_STORE
    _RUN_STORE = store
