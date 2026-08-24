"""MCP Server：mcp-notes 便签服务（默认 stdio 传输，由在线服务启动时自动拉起）。

提供 save_note / list_notes / get_note / delete_note 四个工具，
数据以 JSON 文件持久化（默认 backend/data/mcp-notes.json，可用 MCP_NOTES_FILE 覆盖）。

默认启动方式（stdio，在线服务以子进程自动拉起，无需手动启动）：
    MCP_SERVERS={"mcp-notes": {"command": "python", "args": ["-m", "app.mcp_server.notes_server"]}}

需要独立 HTTP 部署时仍可用：uvicorn app.mcp_server.notes_server:app --port 8001
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DEFAULT_FILE = "./data/mcp-notes.json"


class NotesStore:
    """线程安全的 JSON 文件便签存储：每次操作读盘→变更→原子写回。"""

    def __init__(self, path: str | Path = DEFAULT_FILE):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def save(self, title: str, content: str) -> str:
        with self._lock:
            data = self._load()
            data[title] = {
                "content": content,
                "updated_at": datetime.now().isoformat(timespec="microseconds"),
            }
            self._write(data)
        return f"已保存便签：{title}"

    @staticmethod
    def _display_ts(value: str) -> str:
        """展示用：去掉微秒后缀，保留到秒。"""
        return value[:19] if value and len(value) > 19 else value

    def list(self) -> str:
        with self._lock:
            data = self._load()
        items = sorted(data.items(), key=lambda kv: kv[1].get("updated_at", ""), reverse=True)
        if not items:
            return "暂无便签"
        lines = []
        for title, rec in items:
            preview = rec.get("content", "")
            if len(preview) > 30:
                preview = preview[:30] + "…"
            lines.append(f"- {title}：{preview}（{self._display_ts(rec.get('updated_at', ''))}）")
        return "\n".join(lines)

    def get(self, title: str) -> str:
        with self._lock:
            data = self._load()
        rec = data.get(title)
        if rec is None:
            return f"未找到便签：{title}"
        return f"{title}（{self._display_ts(rec.get('updated_at', ''))}）\n{rec.get('content', '')}"

    def delete(self, title: str) -> str:
        with self._lock:
            data = self._load()
            if title not in data:
                return f"未找到便签：{title}"
            del data[title]
            self._write(data)
        return f"已删除便签：{title}"


store = NotesStore(os.getenv("MCP_NOTES_FILE", DEFAULT_FILE))

mcp = FastMCP("mcp-notes")


@mcp.tool()
def save_note(title: str, content: str) -> str:
    """保存一条便签；同标题会覆盖旧内容。"""
    return store.save(title, content)


@mcp.tool()
def list_notes() -> str:
    """列出全部便签标题与摘要（按更新时间倒序）。"""
    return store.list()


@mcp.tool()
def get_note(title: str) -> str:
    """按标题读取便签全文。"""
    return store.get(title)


@mcp.tool()
def delete_note(title: str) -> str:
    """按标题删除便签。"""
    return store.delete(title)


app = mcp.streamable_http_app()


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输：在线服务以子进程 `python -m app.mcp_server.notes_server` 拉起
