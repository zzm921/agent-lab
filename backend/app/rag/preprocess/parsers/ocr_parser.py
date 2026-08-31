"""扫描件/图片 OCR 解析：逐页渲染 → qwen3.5-flash 多模态识别。

- 图片：原图直接送 OCR；
- 扫描 PDF：每页渲染为 200 DPI PNG（PyMuPDF matrix 缩放）逐页识别，
  页间插入 page_marker（页眉页脚阶段可对 OCR 文本做跨页统计）；
- 单页 OCR 失败不放弃整档：记录 warning 后继续（该页缺失可在报告中发现）。
"""
from __future__ import annotations

import fitz

from app.llm.multimodal import OcrError, ocr_image
from app.rag.preprocess.models import ParsedDocument, ParsedElement, RawFile
from app.rag.preprocess.sniffer import MIME_PDF, MIME_PNG

# 200 DPI ≈ 72 * 2.78；OCR 对 200-300 DPI 的识别率最佳
_RENDER_ZOOM = 200 / 72


def parse_ocr(raw: RawFile) -> ParsedDocument:
    if raw.mime == MIME_PDF:
        return _parse_ocr_pdf(raw)
    text = ocr_image(raw.data)  # 图片（JPEG/PNG）
    elements = [ParsedElement(type="text", text=text.strip())]
    return ParsedDocument(elements=[e for e in elements if e.text], route="ocr", warnings=[])


def _parse_ocr_pdf(raw: RawFile) -> ParsedDocument:
    doc = fitz.open(stream=raw.data, filetype="pdf")
    elements: list[ParsedElement] = []
    warnings: list[str] = []
    try:
        for page_no, page in enumerate(doc, start=1):
            elements.append(ParsedElement(type="page_marker", text="", page=page_no))
            pix = page.get_pixmap(matrix=fitz.Matrix(_RENDER_ZOOM, _RENDER_ZOOM))
            png_bytes = pix.tobytes("png")
            try:
                text = ocr_image(png_bytes).strip()
            except OcrError as exc:
                warnings.append(f"第 {page_no} 页 OCR 失败，该页跳过：{exc}")
                continue
            if text:
                elements.append(ParsedElement(type="text", text=text, page=page_no))
        return ParsedDocument(elements=elements, route="ocr", page_count=doc.page_count, warnings=warnings)
    finally:
        doc.close()
