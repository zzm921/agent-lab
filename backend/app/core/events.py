"""SSE 事件：所有后端→前端的流式事件统一从这里构造。"""

from __future__ import annotations

import json
from typing import Any


def event(type: str, **kwargs: Any) -> dict[str, Any]:
    """构造一个标准事件字典。"""
    return {"type": type, **kwargs}


def encode(data: dict[str, Any]) -> str:
    """将事件字典编码为 SSE 的 data 行。"""
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


def emit_text(emit, type: str, text: str) -> None:
    """把一段文本切分为增量块逐个发射，形成流式效果。"""
    text = text or ""
    chunk_size = 6
    for i in range(0, len(text), chunk_size):
        emit(event(type, delta=text[i : i + chunk_size]))
