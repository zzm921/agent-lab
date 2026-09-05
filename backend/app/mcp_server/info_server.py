"""MCP Server：mcp-info 只读信息服务（默认 stdio 传输，由在线服务启动时自动拉起）。

提供 now / system_info / env_get 三个**只读**工具：
只返回当前时间、系统与白名单环境变量等信息，不产生任何写入 / 修改副作用。

默认启动方式（stdio，在线服务以子进程自动拉起，无需手动启动）：
    MCP_SERVERS={"mcp-info": {"command": "python", "args": ["-m", "app.mcp_server.info_server"]}}

需要独立 HTTP 部署时仍可用：uvicorn app.mcp_server.info_server:app --port 8001
"""
from __future__ import annotations

import datetime
import os
import platform
import socket

from mcp.server.fastmcp import FastMCP

# env_get 仅允许读取以下非敏感环境变量（避免暴露密钥等敏感信息）
_ENV_WHITELIST = frozenset(
    {"HOME", "USERPROFILE", "USER", "USERNAME", "TZ", "LANG", "LC_ALL", "PYTHONPATH"}
)

mcp = FastMCP("mcp-info")


@mcp.tool()
def now() -> str:
    """返回当前日期、时间与系统时区（只读）。"""
    t = datetime.datetime.now().astimezone()
    return f"当前时间：{t:%Y-%m-%d %H:%M:%S}\n时区：{t.tzname()}（UTC{t:%z}）"


@mcp.tool()
def system_info() -> str:
    """返回运行环境信息：操作系统、Python 版本、主机名、CPU 逻辑核数（只读）。"""
    return (
        f"操作系统：{platform.system()} {platform.release()}（{platform.machine()}）\n"
        f"Python：{platform.python_version()}（{platform.python_implementation()}）\n"
        f"主机名：{socket.gethostname()}\n"
        f"CPU 逻辑核数：{os.cpu_count() or '未知'}"
    )


@mcp.tool()
def env_get(name: str) -> str:
    """读取白名单内的环境变量（只读）；白名单外返回拒绝提示，避免泄露敏感信息。"""
    if name not in _ENV_WHITELIST:
        allowed = " / ".join(sorted(_ENV_WHITELIST))
        return f"拒绝读取环境变量：{name}（仅允许 {allowed}）"
    value = os.environ.get(name)
    return f"{name}={value}" if value is not None else f"{name} 未设置"


app = mcp.streamable_http_app()


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输：在线服务以子进程 `python -m app.mcp_server.info_server` 拉起
