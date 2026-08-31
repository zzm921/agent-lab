"""阶段二：页眉页脚与页码移除（跨页重复度检测）。

不是写死「底部的就是页脚」，而是统计跨页重复度：分页文档（≥3 页）取每页
首/尾各 2 行，页码数字归一化后统计出现页数占比，> 60% 判为页眉页脚移除。
这样「每页都有的噪声」被删，而只出现一次的签名/审批行不会误伤。
无页结构的文本类文档仅移除纯页码模式行（第X页/共X页、- X -、Page X of Y）。
"""
from __future__ import annotations

import re

from app.rag.preprocess.models import ParsedDocument, ParsedElement

# 页眉页脚占比阈值
_REPEAT_RATIO = 0.6
# 每页取首尾行数
_EDGE_LINES = 2
# 纯页码模式（无论是否重复都移除）
_PAGE_NO_PATTERNS = [
    re.compile(r"^第\s*\d+\s*页\s*[,，/]\s*共\s*\d+\s*页$"),
    re.compile(r"^第\s*\d+\s*页$"),
    re.compile(r"^共\s*\d+\s*页$"),
    re.compile(r"^-\s*\d+\s*-$"),
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE),
    re.compile(r"^\d+\s*/\s*\d+$"),
]


def _is_page_no(line: str) -> bool:
    return any(p.match(line) for p in _PAGE_NO_PATTERNS)


def _norm(line: str) -> str:
    """页眉页脚频率比较口径：仅去首尾空白（不做数字归一化——正文条款行
    「第X条…」若被归一成同型，会把不同条款误判为重复噪声）。"""
    return line.strip()


def remove_boilerplate(parsed: ParsedDocument) -> tuple[list[ParsedElement], dict]:
    """返回 (过滤后元素列表, 统计)。"""
    stats = {"removed_boilerplate_lines": 0, "removed_page_no_lines": 0}
    elements = parsed.elements

    if parsed.page_count >= 3:
        # 跨页重复度统计：每页首/尾行（跳过页码行本身）
        edge: dict[int, list[str]] = {}
        for el in elements:
            if el.type != "text" or el.page is None:
                continue
            lines = [ln.strip() for ln in el.text.split("\n") if ln.strip()]
            lines = [ln for ln in lines if not _is_page_no(ln)]
            edge.setdefault(el.page, []).extend(lines[:_EDGE_LINES] + lines[-_EDGE_LINES:])
        total_pages = len(edge)
        if total_pages >= 3:
            freq: dict[str, set[int]] = {}
            for page_no, lines in edge.items():
                for line in lines:
                    freq.setdefault(_norm(line), set()).add(page_no)
            boilerplate = {
                pattern for pattern, pages in freq.items() if len(pages) / total_pages > _REPEAT_RATIO
            }
        else:
            boilerplate = set()
    else:
        boilerplate = set()

    filtered: list[ParsedElement] = []
    for el in elements:
        if el.type == "page_marker":
            filtered.append(el)
            continue
        lines = el.text.split("\n")
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if _is_page_no(stripped):
                stats["removed_page_no_lines"] += 1
                continue
            if boilerplate and stripped and _norm(stripped) in boilerplate:
                stats["removed_boilerplate_lines"] += 1
                continue
            kept.append(line)
        text = "\n".join(kept).strip()
        if text:
            filtered.append(ParsedElement(type=el.type, text=text, level=el.level, page=el.page))
    return filtered, stats
