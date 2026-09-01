"""RAG 建库前置处理：全链路编排。

    扫描输入目录 → 格式识别（加密/损坏 → DLQ）→ 复杂度路由（light/OCR）
      → 解析 → 五阶段清洗 → 质量分流 → 跨文档去重 → 报告落盘 / DLQ 归档

单文档任何异常都被捕获为 DocReport(status=dlq)，不中断整批处理。
产出：CleanDocument 列表（供 scripts/ingest_documents.py 交给 RagManager 入库）、
data/ingest/report.json（处理报告）、data/ingest/dlq/（死信归档）。

离线同步实现；若日后接入在线 API（事件循环内），调用方需 asyncio.to_thread() 包裹。
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from app.llm.multimodal import OcrError
from app.rag.preprocess.cleaning.dedup import find_duplicates
from app.rag.preprocess.cleaning.pipeline import clean_document
from app.rag.preprocess.cleaning.quality import SCORE_PASS, SCORE_QUARANTINE
from app.rag.preprocess.complexity import route_document
from app.rag.preprocess.models import (
    STATUS_CONTAINER,
    STATUS_DLQ,
    STATUS_OK,
    STATUS_QUARANTINED,
    STATUS_SUPERSEDED,
    CleanDocument,
    DocReport,
    DocumentRejected,
    GarbledDocument,
    PipelineReport,
    RawFile,
)
from app.rag.preprocess.parsers import route_parse
from app.rag.preprocess.parsers.container import expand_eml, expand_zip
from app.rag.preprocess.sniffer import (
    MIME_PDF,
    MIME_ZIP,
    check_pdf_openable,
    is_container,
    sniff,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("data/docs")
DEFAULT_OUTPUT_DIR = Path("data/ingest")
MAX_CONTAINER_DEPTH = 2  # 容器嵌套层数上限（zip 套 zip/eml 只允许再展开一层）


def run_pipeline(
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> tuple[list[CleanDocument], PipelineReport]:
    """执行全链路前置处理，返回 (可入库的干净文档列表, 批次报告)。"""
    in_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    files = sorted(
        (p for p in in_dir.rglob("*") if p.is_file()),
        key=lambda p: (p.stat().st_mtime, p.name),
    )  # 旧→新：去重时保留 mtime 最新者

    report = PipelineReport()
    clean_docs: list[CleanDocument] = []

    for path in files:
        doc_report = DocReport(path=str(path))
        report.docs.append(doc_report)
        try:
            _process_one(path, doc_report, clean_docs, report)
        except (DocumentRejected, GarbledDocument) as exc:
            doc_report.status = STATUS_DLQ
            doc_report.error = str(exc)
        except OcrError as exc:
            doc_report.status = STATUS_DLQ
            doc_report.error = str(exc)
        except Exception as exc:  # 单文档未知异常：记录进 DLQ，批次继续
            logger.exception("前置处理未知异常：%s", path)
            doc_report.status = STATUS_DLQ
            doc_report.error = f"处理异常：{exc}"

    # 跨文档去重（阶段 4）：对已通过清洗的文档执行
    _dedup_batch(clean_docs, report)

    _write_outputs(out_dir, files, report, clean_docs)
    return clean_docs, report


def _process_one(path: Path, doc_report: DocReport, clean_docs: list[CleanDocument], report: PipelineReport) -> None:
    """顶层入口：真实文件 → 递归处理（容器类型会展开出子文档报告）。"""
    _process_bytes(
        path.read_bytes(), str(path), doc_report, clean_docs, report, depth=0, container=None
    )


def _process_bytes(
    data: bytes,
    display_path: str,
    doc_report: DocReport,
    clean_docs: list[CleanDocument],
    report: PipelineReport,
    depth: int,
    container: str | None,
) -> None:
    """处理一个字节载荷（真实文件或容器子文件），写入 doc_report；容器递归展开。

    display_path：展示/溯源路径——容器子文件为「父容器路径/条目名」（虚拟路径，不落盘）；
    container：父容器路径（顶层文件为 None），随 CleanDocument.metadata 供台账清理子文档。
    """
    raw = sniff(data, Path(display_path))
    doc_report.mime = raw.mime

    if is_container(raw.mime):
        _expand_container(raw, display_path, doc_report, clean_docs, report, depth, container)
        return

    if raw.mime == MIME_PDF:
        check_pdf_openable(data)  # 加密/损坏 → DocumentRejected → DLQ

    route, route_stats = route_document(raw.mime, data)
    doc_report.route = route
    doc_report.stage_stats["route"] = route_stats

    parsed = route_parse(raw, route)
    text, clean_stats = clean_document(parsed)
    doc_report.stage_stats.update(clean_stats)
    doc_report.quality_score = clean_stats.get("quality_score")

    if doc_report.quality_score < SCORE_QUARANTINE:
        doc_report.status = STATUS_DLQ
        doc_report.error = f"质量分 {doc_report.quality_score} 低于 {SCORE_QUARANTINE}，近乎无有效信息"
        return
    if doc_report.quality_score < SCORE_PASS:
        doc_report.status = STATUS_QUARANTINED
        doc_report.error = f"质量分 {doc_report.quality_score} 处于隔离区间 [{SCORE_QUARANTINE}, {SCORE_PASS})，不入主索引"
        return

    doc_report.status = STATUS_OK
    metadata = {
        "source": display_path,
        "quality_score": doc_report.quality_score,
        "route": route,
        "stats": clean_stats,
    }
    if container is not None:
        metadata["container"] = container
    clean_docs.append(CleanDocument(text=text, metadata=metadata))


def _expand_container(
    raw: RawFile,
    display_path: str,
    doc_report: DocReport,
    clean_docs: list[CleanDocument],
    report: PipelineReport,
    depth: int,
    container: str | None,
) -> None:
    """容器（ZIP/EML）展开：子文件逐个递归走完整管线，子报告独立、子失败不拖垮父。"""
    if depth >= MAX_CONTAINER_DEPTH:
        raise DocumentRejected(f"容器嵌套超过 {MAX_CONTAINER_DEPTH} 层，已拒绝：{display_path}")
    expand = expand_zip if raw.mime == MIME_ZIP else expand_eml
    items = expand(raw)
    doc_report.status = STATUS_CONTAINER
    doc_report.stage_stats["container"] = {"mime": raw.mime, "children": len(items)}
    for item in items:
        child_path = f"{display_path}/{item.name}"
        child_report = DocReport(path=child_path)
        report.docs.append(child_report)
        try:
            _process_bytes(
                item.data,
                child_path,
                child_report,
                clean_docs,
                report,
                depth=depth + 1,
                container=display_path,
            )
        except (DocumentRejected, GarbledDocument, OcrError) as exc:
            child_report.status = STATUS_DLQ
            child_report.error = str(exc)
        except Exception as exc:  # 子文档未知异常：只标记该子文档，容器与其余子文档继续
            logger.exception("容器子文档处理异常：%s", child_path)
            child_report.status = STATUS_DLQ
            child_report.error = f"处理异常：{exc}"


def _dedup_batch(clean_docs: list[CleanDocument], report: PipelineReport) -> None:
    """近似/精确去重：旧版本从入库列表移除并标 superseded，保留 mtime 最新版。"""
    if len(clean_docs) < 2:
        return
    # find_duplicates 返回 {旧版下标: 保留新版下标}
    duplicates = find_duplicates([cd.text for cd in clean_docs])
    superseded_idx = set(duplicates)
    kept_docs = [cd for i, cd in enumerate(clean_docs) if i not in superseded_idx]
    for old_idx, kept_idx in duplicates.items():
        old_source = clean_docs[old_idx].metadata["source"]
        kept_source = clean_docs[kept_idx].metadata["source"]
        for doc_report in report.docs:
            if doc_report.path == old_source:
                doc_report.status = STATUS_SUPERSEDED
                doc_report.error = f"与 {kept_source} 内容重复（保留最新版本）"
        logger.info("去重：%s 被 %s 覆盖", old_source, kept_source)
    clean_docs[:] = kept_docs


def _write_outputs(
    out_dir: Path,
    input_files: list[Path],
    report: PipelineReport,
    clean_docs: list[CleanDocument],
) -> None:
    """报告落盘 + DLQ 归档（原文件复制 + 同名 .error.txt 说明）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()
    report_dict["ingested_texts"] = [cd.metadata["source"] for cd in clean_docs]
    (out_dir / "report.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dlq_dir = out_dir / "dlq"
    for doc_report in report.docs:
        if doc_report.status != STATUS_DLQ:
            continue
        src = Path(doc_report.path)
        dlq_dir.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dlq_dir / src.name)
        (dlq_dir / (src.name + ".error.txt")).write_text(
            doc_report.error or "未知原因", encoding="utf-8"
        )
