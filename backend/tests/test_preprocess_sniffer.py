"""格式识别与复杂度路由测试：magic bytes 优先、加密/损坏拦截、二分路由。"""
import fitz
import pytest

from app.rag.preprocess.complexity import route_document
from app.rag.preprocess.models import DocumentRejected
from app.rag.preprocess.sniffer import (
    MIME_DOCX,
    MIME_JPEG,
    MIME_PDF,
    MIME_PNG,
    MIME_TXT,
    check_pdf_openable,
    sniff,
)


def _make_pdf(tmp_path, name="t.pdf", text="员工差旅制度：住宿一线城市每晚不超过四百元。", pages=1, **save_kw):
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            # china-s 为 MuPDF 内置简体中文字体（默认 helv 不支持 CJK）；
            # 单行超页宽会被裁剪，按 ~20 字/行拆成多行落在页面内
            for i in range(0, len(text), 20):
                page.insert_text((72, 72 + (i // 20) * 16), text[i : i + 20], fontname="china-s", fontsize=12)
    path = tmp_path / name
    doc.save(path, **save_kw)
    doc.close()
    return path.read_bytes()


class TestSniff:
    def test_pdf_magic(self, tmp_path):
        raw = sniff(_make_pdf(tmp_path), tmp_path / "t.txt")
        assert raw.mime == MIME_PDF  # 扩展名 .txt 但真实为 PDF

    def test_zip_magic_docx(self):
        raw = sniff(b"PK\x03\x04xxxx", "a.docx")
        assert raw.mime == MIME_DOCX
        assert raw.ext == ".docx"

    def test_image_magics(self):
        assert sniff(b"\x89PNG\r\n\x1a\n....", "a.png").mime == MIME_PNG
        assert sniff(b"\xff\xd8\xff\xe0....", "a.jpg").mime == MIME_JPEG

    def test_text_by_ext(self):
        assert sniff("正文内容".encode(), "a.md").mime == "text/markdown"
        assert sniff("正文内容".encode(), "a.html").mime == "text/html"
        assert sniff("正文内容".encode(), "a.txt").mime == MIME_TXT

    def test_unknown_binary_rejected(self):
        with pytest.raises(DocumentRejected):
            sniff(b"\x00\x01\x02binary\x00junk", "a.bin")


class TestPdfGate:
    def test_corrupted_pdf_rejected(self):
        with pytest.raises(DocumentRejected, match="损坏"):
            check_pdf_openable(b"%PDF-1.4 this is not a real pdf body")

    def test_encrypted_pdf_rejected(self, tmp_path):
        data = _make_pdf(
            tmp_path, "enc.pdf",
            encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="user", owner_pw="owner",
        )
        with pytest.raises(DocumentRejected, match="加密"):
            check_pdf_openable(data)

    def test_normal_pdf_passes(self, tmp_path):
        check_pdf_openable(_make_pdf(tmp_path))


class TestComplexityRouting:
    def test_text_pdf_light(self, tmp_path):
        data = _make_pdf(tmp_path, text="正文内容。" * 20, pages=2)
        route, stats = route_document(MIME_PDF, data)
        assert route == "light"
        assert stats["scanned_ratio"] == 0

    def test_blank_pdf_ocr(self, tmp_path):
        data = _make_pdf(tmp_path, text="", pages=2)  # 无文本层 → 扫描件
        route, stats = route_document(MIME_PDF, data)
        assert route == "ocr"
        assert stats["scanned_ratio"] == 1.0

    def test_image_always_ocr(self):
        route, _ = route_document(MIME_PNG, b"\x89PNG....")
        assert route == "ocr"

    def test_text_file_light(self):
        route, _ = route_document(MIME_TXT, "正文".encode())
        assert route == "light"
