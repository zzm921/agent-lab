"""解析路由：按 MIME + 复杂度路由分发到对应解析器。"""
from __future__ import annotations

from app.rag.preprocess.models import ParsedDocument, RawFile
from app.rag.preprocess.parsers.docx_parser import parse_docx
from app.rag.preprocess.parsers.ocr_parser import parse_ocr
from app.rag.preprocess.parsers.pdf_parser import parse_pdf
from app.rag.preprocess.parsers.text_parser import parse_text
from app.rag.preprocess.sniffer import MIME_DOCX, MIME_PDF, is_text


def route_parse(raw: RawFile, route: str) -> ParsedDocument:
    """MIME（真实格式）+ 路由（light/ocr）→ 对应解析器。"""
    if route == "ocr":
        return parse_ocr(raw)
    if is_text(raw.mime):
        return parse_text(raw)
    if raw.mime == MIME_DOCX:
        return parse_docx(raw)
    if raw.mime == MIME_PDF:
        return parse_pdf(raw)
    raise ValueError(f"不支持的 MIME 类型：{raw.mime}")
