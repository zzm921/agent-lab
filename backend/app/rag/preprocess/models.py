"""前置处理数据结构：从原始文件到干净文本的全链路载体。

- RawFile：格式识别后的原始文件（字节 + MIME + 扩展名）；
- ParsedElement / ParsedDocument：解析层产物——保留结构（标题层级/表格/页边界），
  支撑页眉页脚跨页检测、按章节溯源与后续分块；
- CleanDocument：清洗后的可入库文本（附质量分与处理统计）；
- DocReport / PipelineReport：逐文档处理报告与批次汇总（status 四态）。

status 四态 + 容器标记含义：
- ok          正常清洗完成，进入入库候选；
- superseded  近似/精确重复，保留最新一份，旧版仅标记不删除（合规可追溯）；
- quarantined 质量分 50-69 的低质量文档：不入主索引，原文件保留在输入目录；
- dlq         死信队列：解析/OCR/乱码/极低质量等原因彻底失败，原文件复制到 DLQ 目录；
- container   容器文档（ZIP/EML）：本身不入库，已展开为子文件递归处理（子文件有独立报告）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 文档状态
STATUS_OK = "ok"
STATUS_SUPERSEDED = "superseded"
STATUS_QUARANTINED = "quarantined"
STATUS_DLQ = "dlq"
STATUS_CONTAINER = "container"  # 容器文档：已展开为子文件，本身不入库

# 解析路由
ROUTE_LIGHT = "light"  # 快路径：文本/MD/DOCX/文本型 PDF 直接解析
ROUTE_OCR = "ocr"      # 重路径：扫描 PDF / 图片 → qwen3.5-flash 多模态 OCR


@dataclass
class RawFile:
    """格式识别后的原始文件。"""

    path: Path
    data: bytes
    mime: str  # sniff 出的真实 MIME（text/plain | text/markdown | text/html | docx | application/pdf | image/jpeg | image/png）
    ext: str  # 原始扩展名（小写，含点），仅作兜底参考


@dataclass
class ParsedElement:
    """解析后的结构元素：标题/正文/表格/页边界标记。"""

    type: str  # title | text | table | page_marker
    text: str
    level: int = 0  # 标题层级（title 有效，1-6）
    page: int | None = None  # 来源页码（PDF/OCR 有值，文本类为 None）


@dataclass
class ParsedDocument:
    """单个文档的解析结果。"""

    elements: list[ParsedElement]
    route: str = ROUTE_LIGHT
    page_count: int = 0  # 页数（分页文档有值；供页眉页脚跨页统计）
    warnings: list[str] = field(default_factory=list)  # 解析期警告（如部分页 OCR 低置信）


@dataclass
class CleanDocument:
    """清洗完成的可入库文档。"""

    text: str
    metadata: dict = field(default_factory=dict)  # source/quality_score/route/stats


@dataclass
class DocReport:
    """单个文档的处理报告。"""

    path: str
    mime: str = ""
    route: str = ""
    status: str = ""
    quality_score: int | None = None
    stage_stats: dict = field(default_factory=dict)  # 各清洗阶段统计
    error: str = ""  # status=dlq 时的中文失败原因


@dataclass
class PipelineReport:
    """整批处理报告：逐文档明细 + 状态汇总。"""

    docs: list[DocReport] = field(default_factory=list)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for d in self.docs:
            counts[d.status] = counts.get(d.status, 0) + 1
        return {
            "total": len(self.docs),
            "ok": counts.get(STATUS_OK, 0),
            "superseded": counts.get(STATUS_SUPERSEDED, 0),
            "quarantined": counts.get(STATUS_QUARANTINED, 0),
            "dlq": counts.get(STATUS_DLQ, 0),
            "container": counts.get(STATUS_CONTAINER, 0),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "docs": [
                {
                    "path": d.path,
                    "mime": d.mime,
                    "route": d.route,
                    "status": d.status,
                    "quality_score": d.quality_score,
                    "stage_stats": d.stage_stats,
                    "error": d.error,
                }
                for d in self.docs
            ],
        }


class DocumentRejected(Exception):
    """文档前置拦截（加密/损坏/空文件等）：直接进 DLQ，不进解析。"""


class GarbledDocument(Exception):
    """乱码/空文本拦截：清洗阶段判定不可入库，进 DLQ。"""
