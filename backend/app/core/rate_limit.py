"""每日配额限流：按「自然日 + 客户端」维度统计当天对话次数，防止单台设备/IP 滥用。

统计口径：每发起一次 POST /api/stream 视为一次「对话」。
客户端标识：
- 优先使用前端下发的 X-Client-Id（浏览器 localStorage 生成的设备指纹，用于精确区分「一台电脑」，
  规避办公网 NAT 下多人共享同一公网 IP 被合并计数的问题）；
- 未携带时退回请求方 IP（区分「一个 IP」）。
按自然日（服务器本地日期）计数，跨天自动清零；支持持久化到 JSON 文件，服务重启后计数不丢。
"""
from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path


class DailyQuota:
    """固定窗口的每日计数器，可文件持久化。

    - limit 为每日允许次数上限；
    - path 为 None 时仅内存计数（测试用），否则每次变更后落盘，重启不丢；
    - 所有方法线程安全（内部加锁），可安全用于 FastAPI 端点。
    """

    def __init__(self, limit: int = 20, path: str | None = None):
        self.limit = max(0, int(limit))
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._date = date.today().isoformat()
        self._counts: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._date = data.get("date", date.today().isoformat())
            self._counts = data.get("counts", {})
        except (OSError, ValueError):
            # 文件损坏/不可读时按空计数处理，避免因限流文件异常导致服务不可用
            self._date = date.today().isoformat()
            self._counts = {}

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"date": self._date, "counts": self._counts}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            # 目录只读等场景：仅内存计数，不阻断请求
            pass

    def _rollover(self) -> None:
        today = date.today().isoformat()
        if self._date != today:
            self._date = today
            self._counts = {}

    def remaining(self, client_key: str) -> int:
        """查询某客户端今日剩余可用次数。"""
        with self._lock:
            self._rollover()
            return max(0, self.limit - self._counts.get(client_key, 0))

    def try_consume(self, client_key: str) -> tuple[bool, int]:
        """尝试消耗一次额度。

        返回 (是否允许, 剩余次数)；超过每日上限时返回 (False, 0)。
        """
        with self._lock:
            self._rollover()
            used = self._counts.get(client_key, 0)
            if used >= self.limit:
                return False, 0
            self._counts[client_key] = used + 1
            self._save()
            return True, self.limit - used - 1
