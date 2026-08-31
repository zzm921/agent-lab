"""复杂度判定：文本型 PDF 按页字符密度二分路由（light / OCR）。

简化自技术方案文档 §3.2 的八维评分：当前文档规模只需回答一个问题——
「有没有文本层」。字符密度 < 50 字/页 判为扫描页，扫描页占比 > 0.5 或
全文几乎无可提取字符时走 OCR 重路径（qwen3.5-flash 多模态识别），
否则走快路径。图片文件天然无文本层，直接路由 OCR。
"""
from __future__ import annotations

from app.rag.preprocess.models import ROUTE_LIGHT, ROUTE_OCR
from app.rag.preprocess.sniffer import MIME_PDF, is_image

# 判定阈值（与《复杂情况应对手册》保持同步）
SCANNED_PAGE_CHARS = 50  # 单页可提取字符低于该值视为扫描页
SCANNED_RATIO = 0.5  # 扫描页占比超过该值 → 整档走 OCR


def route_document(mime: str, data: bytes) -> tuple[str, dict]:
    """返回 (route, stats)。stats 附带判定依据，写入处理报告供排查。"""
    if is_image(mime):
        return ROUTE_OCR, {"reason": "图片文件，无文本层"}
    if mime == MIME_PDF:
        return _route_pdf(data)
    return ROUTE_LIGHT, {"reason": "文本类格式"}


def _route_pdf(data: bytes) -> tuple[str, dict]:
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        total_pages = doc.page_count
        scanned_pages = 0
        total_chars = 0
        for page in doc:
            chars = len(page.get_text("text").strip())
            total_chars += chars
            if chars < SCANNED_PAGE_CHARS:
                scanned_pages += 1
        if total_pages == 0:
            return ROUTE_LIGHT, {"reason": "空 PDF（0 页）", "total_pages": 0}
        ratio = scanned_pages / total_pages
        route = ROUTE_OCR if (ratio > SCANNED_RATIO or total_chars < SCANNED_PAGE_CHARS) else ROUTE_LIGHT
        stats = {
            "total_pages": total_pages,
            "scanned_pages": scanned_pages,
            "scanned_ratio": round(ratio, 2),
            "total_chars": total_chars,
        }
        return route, stats
    finally:
        doc.close()
