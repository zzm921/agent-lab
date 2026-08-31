"""文本型 PDF 解析：PyMuPDF 逐页提取，块级坐标排序重建阅读顺序。

PDF 没有「段落」概念，只有字符和坐标；get_text("blocks") 返回文本块及
bbox，按 (y0, x0) 排序后输出，避免左右栏交错。每页插入 page_marker
元素，供页眉页脚阶段做跨页重复度统计。逐页流式产出，不整档驻留。
"""
from __future__ import annotations

import fitz

from app.rag.preprocess.models import ParsedDocument, ParsedElement, RawFile


def parse_pdf(raw: RawFile) -> ParsedDocument:
    doc = fitz.open(stream=raw.data, filetype="pdf")
    elements: list[ParsedElement] = []
    try:
        for page_no, page in enumerate(doc, start=1):
            elements.append(ParsedElement(type="page_marker", text="", page=page_no))
            blocks = page.get_text("blocks") or []
            # 块结构：(x0, y0, x1, y1, text, block_no, block_type)；文本块 type=0
            text_blocks = sorted(
                (b for b in blocks if b[6] == 0 and b[4].strip()),
                key=lambda b: (round(b[1], 1), round(b[0], 1)),
            )
            for block in text_blocks:
                elements.append(ParsedElement(type="text", text=block[4].strip(), page=page_no))
        return ParsedDocument(elements=elements, page_count=doc.page_count)
    finally:
        doc.close()
