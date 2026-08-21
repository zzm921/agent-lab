"""当前时间工具。"""
from datetime import datetime

from langchain_core.tools import tool


@tool
def time_now() -> str:
    """返回当前本地日期与时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
