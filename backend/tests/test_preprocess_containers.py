"""容器递归测试：ZIP/EML 识别与展开、防炸弹护栏、嵌套层级、e2e 层级报告。

所有二进制容器在测试内现场生成，不向仓库提交二进制夹具；OCR mock 不联网。
"""
import io
import zipfile
from pathlib import Path

import pytest
from email.message import EmailMessage

from app.rag.preprocess import run_pipeline
from app.rag.preprocess.models import DocumentRejected
from app.rag.preprocess.parsers.container import (
    ZIP_MAX_ENTRIES,
    expand_eml,
    expand_zip,
)
from app.rag.preprocess.sniffer import (
    MIME_DOCX,
    MIME_EML,
    MIME_ZIP,
    sniff,
)

GOOD_TEXT = (
    "项目管理制度总则规定，全员应当遵守研发与文档各项规范安排。"
    "研发文档须在评审通过后归档至知识库，注明作者、版本与生效日期。"
    "版本变更需在变更记录中写明修改原因与影响范围，经负责人审批后生效。"
    "知识库文档每季度复审一次，过期文档标记为历史版本并从主索引移除。"
    "本制度自发布之日起施行，由项目管理办公室负责解释与修订完善。"
)

MEETING_TEXT = (
    "会议室使用管理规定要求，各部门使用会议室须提前一天预约登记。"
    "单场会议时长原则上不超过两小时，超时须现场确认后续时段是否空闲。"
    "会后使用人应当清理白板、归位桌椅并关闭投影与空调等设备电源。"
    "外部来访人员使用会议室由接待部门统一申请，并负责全程陪同管理。"
    "行政部每周汇总会议室使用情况，对多次违约使用的部门进行通报。"
)

TRAVEL_TEXT = (
    "差旅报销管理规定明确，员工因公出差须事先提交申请并获部门负责人批准。"
    "交通工具选择以经济实用为原则，同城高铁两小时以内原则上不安排飞机出行。"
    "住宿费用按职级分档报销，票据须为本人实名且日期与行程一致方可受理。"
    "报销材料应当在返回后十个工作日内提交，逾期未提交的视为自动放弃处理。"
    "财务部按月抽查报销凭证，发现虚报冒领的移交人力资源部按制度处理。"
)


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _eml_bytes(subject: str, body: str, attachments: list[tuple[str, str, bytes]]) -> bytes:
    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    msg["To"] = "receiver@example.com"
    msg["Subject"] = subject
    msg.set_content(body)
    for name, ctype, data in attachments:
        maintype, subtype = ctype.split("/")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    return msg.as_bytes()


class TestSniffContainer:
    def test_plain_zip_vs_docx(self):
        zip_data = _zip_bytes({"a.md": GOOD_TEXT.encode()})
        assert sniff(zip_data, "a.zip").mime == MIME_ZIP

        from docx import Document

        doc = Document()
        doc.add_heading("标题", level=1)
        buf = io.BytesIO()
        doc.save(buf)
        assert sniff(buf.getvalue(), "a.docx").mime == MIME_DOCX

    def test_ooxml_non_docx_rejected(self):
        # xlsx 形态：有 [Content_Types].xml 但无 word/document.xml → 明确拒绝
        xlsx_like = _zip_bytes({"[Content_Types].xml": b"<Types/>", "xl/workbook.xml": b"<workbook/>"})
        with pytest.raises(DocumentRejected, match="暂不支持"):
            sniff(xlsx_like, "a.xlsx")

    def test_corrupt_zip_rejected(self):
        corrupt = b"PK\x03\x04" + b"\x00garbage" * 16
        with pytest.raises(DocumentRejected, match="损坏"):
            sniff(corrupt, "broken.zip")

    def test_eml_by_content_and_ext(self):
        data = _eml_bytes("周报", "正文内容。", [])
        assert sniff(data, "mail.eml").mime == MIME_EML
        assert sniff(data, "无扩展名文件").mime == MIME_EML  # 内容启发式兜底
        assert sniff("普通文本内容而已。".encode(), "note.txt").mime == "text/plain"


class TestExpand:
    def test_zip_skips_junk(self):
        raw = sniff(
            _zip_bytes(
                {
                    "docs/one.md": GOOD_TEXT.encode(),
                    "__MACOSX/docs/._one.md": b"junk",
                    ".DS_Store": b"junk",
                    "sub/.hidden.txt": "隐藏文件内容。".encode(),
                }
            ),
            "a.zip",
        )
        names = [item.name for item in expand_zip(raw)]
        assert names == ["docs/one.md", "sub/.hidden.txt"]

    def test_eml_body_and_attachments(self):
        raw = sniff(
            _eml_bytes(
                "项目周报",
                "本周完成容器解析模块开发。",
                [("week.md", "text/markdown", GOOD_TEXT.encode())],
            ),
            "mail.eml",
        )
        items = expand_eml(raw)
        assert [i.name for i in items] == ["项目周报.txt", "week.md"]
        assert "容器解析" in items[0].data.decode()

    def test_eml_attachment_without_filename(self):
        msg = EmailMessage()
        msg["From"] = "a@b.c"
        msg["Subject"] = "无附件名"
        msg.set_content("正文。")
        msg.add_attachment(b"binary-ish", maintype="application", subtype="octet-stream")
        items = expand_eml(sniff(msg.as_bytes(), "m.eml"))
        assert items[1].name == "attachment_application_octet-stream.bin"

    def test_zip_entry_bomb_guard(self):
        entries = {f"f{i:03d}.md": "内容。".encode() for i in range(ZIP_MAX_ENTRIES + 1)}
        with pytest.raises(ValueError, match="条目数"):
            expand_zip(sniff(_zip_bytes(entries), "bomb.zip"))


class TestPipelineContainers:
    def test_zip_and_eml_e2e_hierarchy(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.zip").write_bytes(
            _zip_bytes(
                {
                    "one.md": GOOD_TEXT.encode(),
                    "two.md": MEETING_TEXT.encode(),
                    "bad.md": ("������ 乱码片段" * 20).encode(),
                }
            )
        )
        (docs / "b.eml").write_bytes(
            _eml_bytes(
                "周报",
                "邮件正文：本周推进建库事项，容器解析模块已上线并接入前置管线。"
                "增量入库台账已完成设计，按内容哈希判断文档是否需要重建索引。"
                "下周计划开展真实语料试跑，观察隔离率与 DLQ 分布并校准阈值。"
                "另请各位在周五前提交本月知识库文档复审意见，逾期视为无修改。",
                [("week.md", "text/markdown", TRAVEL_TEXT.encode())],
            )
        )
        clean_docs, report = run_pipeline(docs, tmp_path / "out")

        by_path = {d.path: d for d in report.docs}
        assert by_path[str(docs / "a.zip")].status == "container"
        assert by_path[str(docs / "b.eml")].status == "container"
        # 子文档层级路径 + 独立状态（坏子文档 dlq 不拖垮容器与其余子文档）
        assert by_path[f"{docs / 'a.zip'}/one.md"].status == "ok"
        assert by_path[f"{docs / 'a.zip'}/two.md"].status == "ok"
        assert by_path[f"{docs / 'a.zip'}/bad.md"].status == "dlq"
        assert by_path[f"{docs / 'b.eml'}/周报.txt"].status == "ok"
        assert by_path[f"{docs / 'b.eml'}/week.md"].status == "ok"

        # 干净文档：source 为虚拟层级路径，metadata 记录父容器
        sources = {cd.metadata["source"]: cd for cd in clean_docs}
        assert f"{docs / 'a.zip'}/bad.md" not in sources
        assert sources[f"{docs / 'a.zip'}/one.md"].metadata["container"] == str(docs / "a.zip")
        assert "差旅报销管理规定" in sources[f"{docs / 'b.eml'}/week.md"].text

    def test_nested_zip_depth(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        inner = _zip_bytes({"inner.md": GOOD_TEXT.encode()})
        (docs / "outer.zip").write_bytes(_zip_bytes({"inner.zip": inner}))
        clean_docs, report = run_pipeline(docs, tmp_path / "out")

        by_path = {d.path: d for d in report.docs}
        assert by_path[f"{docs / 'outer.zip'}/inner.zip"].status == "container"
        assert by_path[f"{docs / 'outer.zip'}/inner.zip/inner.md"].status == "ok"
        assert any(cd.metadata["source"].endswith("inner.zip/inner.md") for cd in clean_docs)

    def test_triple_nested_zip_rejected(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        inner = _zip_bytes({"f.md": "内容。".encode()})
        mid = _zip_bytes({"inner.zip": inner})
        (docs / "outer.zip").write_bytes(_zip_bytes({"mid.zip": mid}))
        clean_docs, report = run_pipeline(docs, tmp_path / "out")

        by_path = {d.path: d for d in report.docs}
        # 第 3 层容器（depth=2 达上限）拒绝，但已有内层文档不受影响
        assert by_path[f"{docs / 'outer.zip'}/mid.zip/inner.zip"].status == "dlq"
        assert "容器嵌套超过" in by_path[f"{docs / 'outer.zip'}/mid.zip/inner.zip"].error

    def test_container_guard_dlq(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        entries = {f"f{i:03d}.md": "内容。".encode() for i in range(ZIP_MAX_ENTRIES + 1)}
        (docs / "bomb.zip").write_bytes(_zip_bytes(entries))
        _, report = run_pipeline(docs, tmp_path / "out")
        assert report.docs[0].status == "dlq"
        assert "条目数" in report.docs[0].error
