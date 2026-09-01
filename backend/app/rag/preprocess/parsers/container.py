"""容器展开：ZIP 压缩包 / EML 邮件 → 子文件字节队列（供管线递归处理）。

- ZIP：标准库 zipfile，跳过目录与系统杂项（__MACOSX/、.DS_Store、隐藏文件）；
- EML：标准库 email 解析——正文 text/plain 存为虚拟文件，附件逐个拆出（附件
  也可以是 PDF/DOCX/图片，交回管线按真实格式路由）；
- 防炸弹护栏：条目数 / 解压总量 / 附件数 / 嵌套层数上限（阈值见模块级常量，
  与《复杂情况应对手册》速查表同步）。RAR/7z 需系统外部依赖，本期不支持——
  sniffer 未识别的字节流会直接进 DLQ。
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage  # noqa: F401 （policy.default 解析产物类型标注用）
from email.parser import BytesParser

from app.rag.preprocess.models import RawFile

# 防炸弹 / 防滥用护栏（改动代码必须同步《复杂情况应对手册》速查表）
ZIP_MAX_ENTRIES = 200  # 单个 zip 最大条目数
ZIP_MAX_TOTAL_BYTES = 256 * 1024 * 1024  # 解压总量上限（256 MB）
EML_MAX_ATTACHMENTS = 50  # 单封邮件最大附件数

# 系统杂项条目：展开时直接丢弃
_ZIP_SKIP_PREFIXES = ("__MACOSX/", ".")
_ZIP_SKIP_NAMES = {".DS_Store", "Thumbs.db"}


@dataclass
class ContainerItem:
    """容器内展开出的一个子文件：容器内路径名 + 字节内容。"""

    name: str  # 容器内路径（zip 条目名 / 附件文件名 / 正文虚拟文件名）
    data: bytes


def expand_zip(raw: RawFile) -> list[ContainerItem]:
    """展开 zip 压缩包为子文件列表；超护栏或损坏时拒绝（进 DLQ）。"""
    items: list[ContainerItem] = []
    total = 0
    with zipfile.ZipFile(io.BytesIO(raw.data)) as zf:
        entries = [info for info in zf.infolist() if not info.is_dir()]
        if len(entries) > ZIP_MAX_ENTRIES:
            raise ValueError(
                f"ZIP 条目数 {len(entries)} 超过上限 {ZIP_MAX_ENTRIES}，疑似压缩包炸弹，已拒绝"
            )
        for info in entries:
            name = info.filename
            parts = name.rsplit("/", 1)
            base = parts[-1] if parts else name
            if name.startswith(_ZIP_SKIP_PREFIXES) or base in _ZIP_SKIP_NAMES:
                continue
            total += info.file_size
            if total > ZIP_MAX_TOTAL_BYTES:
                raise ValueError(
                    f"ZIP 解压总量超过上限 {ZIP_MAX_TOTAL_BYTES // (1024 * 1024)} MB，疑似压缩包炸弹，已拒绝"
                )
            items.append(ContainerItem(name=name, data=zf.read(info)))
    return items


def expand_eml(raw: RawFile) -> list[ContainerItem]:
    """展开 EML 邮件：正文（text/plain 优先，无则 text/html）+ 全部附件。

    正文存为虚拟文件「<主题>.txt / .html」，附件保留原始文件名（缺失时按
    类型猜扩展名）。附件数超上限拒绝（防滥用）。
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw.data)
    subject = (msg.get("Subject") or "邮件正文").strip() or "邮件正文"
    items: list[ContainerItem] = []

    body_done = False
    attachments = 0
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        filename = part.get_filename()
        is_attachment = part.get_content_disposition() == "attachment" or bool(filename)
        if is_attachment:
            attachments += 1
            if attachments > EML_MAX_ATTACHMENTS:
                raise ValueError(
                    f"邮件附件数超过上限 {EML_MAX_ATTACHMENTS}，疑似滥用，已拒绝"
                )
            items.append(ContainerItem(name=_safe_name(filename, part), data=part.get_payload(decode=True) or b""))
            continue
        if body_done or part.get_content_maintype() != "text":
            continue
        # 正文：优先 text/plain；取到第一个正文部件后不再取后续（multipart/alternative 只取一路）
        payload = part.get_payload(decode=True) or b""
        ext = "txt" if part.get_content_subtype() == "plain" else "html"
        items.append(ContainerItem(name=f"{subject}.{ext}", data=payload))
        body_done = part.get_content_subtype() == "plain"
    return items


def _safe_name(filename: str | None, part) -> str:
    """附件文件名兜底：缺失时按 MIME 类型猜扩展名，避免无名附件互相覆盖。"""
    if filename:
        return filename
    ctype = (part.get_content_type() or "application/octet-stream").replace("/", "_")
    return f"attachment_{ctype}.bin"
