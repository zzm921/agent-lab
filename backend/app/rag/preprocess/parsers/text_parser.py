"""文本类解析：TXT / Markdown / HTML → 结构化元素。

- 编码检测：charset-normalizer 兜底 GBK/UTF-8 误判（乱码进入清洗层还会二次拦截）；
- Markdown：`#` 标题映射 title(level)，`|` 表格行聚合为 table 元素，其余按空行分段；
- HTML：轻量去标签（不引入 bs4），按块分段为 text 元素。
"""
from __future__ import annotations

import html as html_mod
import re

from charset_normalizer import from_bytes

from app.rag.preprocess.models import ParsedDocument, ParsedElement, RawFile

_MD_TITLE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def decode(data: bytes) -> str:
    """字节 → 文本：charset-normalizer 检测编码，失败回退 UTF-8 替换模式。"""
    best = from_bytes(data).best()
    if best is not None:
        return str(best)
    return data.decode("utf-8", errors="replace")


def parse_text(raw: RawFile) -> ParsedDocument:
    text = decode(raw.data)
    if raw.mime == "text/markdown":
        elements = _parse_markdown(text)
    elif raw.mime == "text/html":
        elements = _parse_html(text)
    else:
        elements = _parse_plain(text)
    return ParsedDocument(elements=elements)


def _parse_markdown(text: str) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    table_buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            table_buf.append(stripped)
            continue
        if table_buf:
            elements.append(ParsedElement(type="table", text="\n".join(table_buf)))
            table_buf = []
        m = _MD_TITLE.match(stripped)
        if m:
            elements.append(ParsedElement(type="title", text=m.group(2), level=len(m.group(1))))
        elif stripped:
            elements.append(ParsedElement(type="text", text=stripped))
    if table_buf:
        elements.append(ParsedElement(type="table", text="\n".join(table_buf)))
    return elements


def _parse_plain(text: str) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block:
            elements.append(ParsedElement(type="text", text=block))
    return elements


_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _parse_html(text: str) -> list[ParsedElement]:
    text = _BLOCK.sub("", text)
    text = _TAG.sub("\n", text)
    text = html_mod.unescape(text)
    return _parse_plain(text)
