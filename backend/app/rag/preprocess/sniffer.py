"""格式识别：magic bytes 优先、扩展名兜底，并前置拦截加密/损坏文件。

为什么不能只看扩展名：`.txt` 可能实际是 PDF（重命名伪装），按扩展名会走
纯文本解析读出二进制乱码。这里对字节头部做真实格式嗅探后再路由。
不引入 python-magic（Windows 需 libmagic DLL）：常见格式手写 magic 判定足够。
"""
from __future__ import annotations

import re
from pathlib import Path

from app.rag.preprocess.models import DocumentRejected, RawFile

# magic bytes → MIME
_MAGIC_PDF = b"%PDF-"
_MAGIC_ZIP = b"PK\x03\x04"
_MAGIC_JPEG = b"\xff\xd8\xff"
_MAGIC_PNG = b"\x89PNG\r\n\x1a\n"

MIME_PDF = "application/pdf"
MIME_DOCX = "docx"  # Office OpenXML（本期仅 DOCX 消费 zip 容器）
MIME_ZIP = "application/zip"  # zip 压缩包容器（区别于 OOXML 的 docx）
MIME_EML = "message/rfc822"  # 邮件文件（正文 + 附件容器）
MIME_JPEG = "image/jpeg"
MIME_PNG = "image/png"
MIME_TXT = "text/plain"
MIME_MD = "text/markdown"
MIME_HTML = "text/html"

_TEXT_MIMES = {MIME_TXT, MIME_MD, MIME_HTML}
_IMAGE_MIMES = {MIME_JPEG, MIME_PNG}

# EML 头部首行特征（无 magic bytes，靠首行邮件头 + MIME-Version 判定）
_EML_FIRST_LINE = re.compile(
    r"^(from|to|subject|date|received|message-id|return-path|mime-version):",
    re.IGNORECASE,
)


def sniff(data: bytes, path: Path | str) -> RawFile:
    """按字节头识别真实格式，构造 RawFile；无法识别的二进制直接拒绝。"""
    p = Path(path)
    ext = p.suffix.lower()
    if data.startswith(_MAGIC_PDF):
        mime = MIME_PDF
    elif data.startswith(_MAGIC_ZIP):
        mime = _zip_mime(data)  # PK 头：docx（OOXML）或 zip 压缩包，需看内层
    elif data.startswith(_MAGIC_JPEG):
        mime = MIME_JPEG
    elif data.startswith(_MAGIC_PNG):
        mime = MIME_PNG
    else:
        # 文本类：按扩展名细分；含大量 NUL 字节的未知二进制拒绝（防误读乱码）
        if b"\x00" in data[:1024]:
            raise DocumentRejected(f"无法识别的二进制格式（扩展名 {ext or '无'}），已拒绝处理")
        if ext == ".eml" or _looks_like_eml(data):
            mime = MIME_EML
        else:
            mime = _text_mime(ext)
    return RawFile(path=p, data=data, mime=mime, ext=ext)


def _zip_mime(data: bytes) -> str:
    """PK 头细分：docx（含 word/document.xml）→ docx；其他 OOXML 拒绝；其余为 zip 压缩包。

    顺带前置拦截损坏/加密 zip（namelist 或试读抛异常），不进解析管道。
    """
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            for name in names:  # 试读首条目：加密 zip 此处抛 RuntimeError
                if not name.endswith("/"):
                    zf.read(name)
                    break
    except RuntimeError:
        raise DocumentRejected("ZIP 已加密（需要密码），请先解密后再入库")
    except Exception as exc:
        raise DocumentRejected(f"ZIP 容器损坏，无法打开：{exc}") from exc
    if "word/document.xml" in names:
        return MIME_DOCX
    if "[Content_Types].xml" in names:
        raise DocumentRejected("Office OpenXML 格式（xlsx/pptx 等）暂不支持，请先另存为 docx 或 PDF")
    return MIME_ZIP


def _looks_like_eml(data: bytes) -> bool:
    """内容启发式：首行是 RFC822 邮件头且出现 MIME-Version/Subject 特征。"""
    head = data[:2048]
    if b"mime-version:" not in head.lower():
        return False
    first_line = head.split(b"\n", 1)[0].strip()
    return bool(_EML_FIRST_LINE.match(first_line.decode("utf-8", errors="ignore")))


def _text_mime(ext: str) -> str:
    if ext in (".md", ".markdown"):
        return MIME_MD
    if ext in (".html", ".htm"):
        return MIME_HTML
    return MIME_TXT


def is_text(mime: str) -> bool:
    return mime in _TEXT_MIMES


def is_image(mime: str) -> bool:
    return mime in _IMAGE_MIMES


def is_container(mime: str) -> bool:
    """容器类型（zip 压缩包 / EML 邮件）：需展开为子文件队列递归处理。"""
    return mime in (MIME_ZIP, MIME_EML)


def check_pdf_openable(data: bytes) -> None:
    """PDF 前置检查：损坏/加密直接拒绝（进 DLQ），不进解析管道。"""
    import fitz  # pymupdf

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # 损坏文件 fitz 抛 MupdfError 等
        raise DocumentRejected(f"PDF 文件损坏，无法打开：{exc}") from exc
    try:
        if doc.needs_pass:
            raise DocumentRejected("PDF 已加密（需要密码），请先解密后再入库")
    finally:
        doc.close()
