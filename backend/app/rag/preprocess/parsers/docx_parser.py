"""DOCX 解析：Heading 样式 → 标题层级，表格 → 「列名: 值」行表示。

按 body 顺序遍历段落与表格（保持原文档阅读顺序），表格转 Markdown
并逐行扁平化（列名1: 值1 | 列名2: 值2），提高关键词召回。
"""
from __future__ import annotations

from docx.document import Document as _DocumentBody
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.rag.preprocess.models import ParsedDocument, ParsedElement, RawFile


def parse_docx(raw: RawFile) -> ParsedDocument:
    import io

    from docx import Document

    doc = Document(io.BytesIO(raw.data))
    elements: list[ParsedElement] = []
    for block in _iter_blocks(doc):
        if isinstance(block, Table):
            elements.append(ParsedElement(type="table", text=_table_to_lines(block)))
            continue
        para = block
        text = para.text.strip()
        if not text:
            continue
        level = _heading_level(para)
        if level:
            elements.append(ParsedElement(type="title", text=text, level=level))
        else:
            elements.append(ParsedElement(type="text", text=text))
    return ParsedDocument(elements=elements)


def _iter_blocks(doc: _DocumentBody):
    """按文档顺序产出段落与表格（python-docx 官方推荐遍历法）。"""
    from docx.oxml.ns import qn

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _heading_level(para: Paragraph) -> int:
    """Heading N / 标题 N 样式 → 层级 N；正文返回 0。"""
    style = (para.style.name or "").lower()
    for token in style.split():
        if token.isdigit():
            return min(int(token), 6)
    return 0


def _table_to_lines(table: Table) -> str:
    """表格 → 表头行 + 每行「列名: 值 | 列名: 值」扁平化文本。"""
    rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    headers = rows[0]
    lines = ["| " + " | ".join(headers) + " |"]
    for row in rows[1:]:
        pairs = [f"{h}: {v}" for h, v in zip(headers, row)]
        lines.append(" | ".join(pairs))
    return "\n".join(lines)
