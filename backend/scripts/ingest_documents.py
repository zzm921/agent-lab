"""建库脚本：真实文档前置处理（解析 + 清洗）→ 台账判重 → 各 RAG 方案增量入库。

前置链路：扫描 data/docs/ → 格式识别（容器 ZIP/EML 递归展开）→ 复杂度路由
（扫描件走 qwen3.5-flash OCR）→ 五阶段清洗（归一化/页眉页脚/乱码/质量/去重）
→ 内容 hash 台账判重 → 仅变更文档重新分块向量化。
处理报告落盘 data/ingest/report.json，失败文档归档 data/ingest/dlq/，
台账落盘 data/ingest/ledger.json（doc_path → 内容 hash，支持增量更新）。

在 backend/ 目录下执行：
  python scripts/ingest_documents.py [--input data/docs] [--force]
    [--schemes naive,advanced,modular]
缺省按台账增量：hash 未变跳过；变更/新增按 source 删旧块插新块；
文件删除 / 容器内移除 / 本轮被淘汰（去重·隔离·DLQ）的文档清理其在库旧块。
--force：清空集合与台账全量重建。
与内嵌语料通道（scripts/ingest_*.py → build_corpus）互不影响。
"""
import argparse
from pathlib import Path

from app.config import settings
from app.llm.client import create_embeddings
from app.rag.manager import RagManager
from app.rag.preprocess import run_pipeline
from app.rag.preprocess.ledger import LedgerStore
from app.rag.preprocess.models import (
    STATUS_DLQ,
    STATUS_QUARANTINED,
    STATUS_SUPERSEDED,
    CleanDocument,
    PipelineReport,
)
from app.rag.preprocess.pipeline import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR


def run_incremental_ingest(
    clean_docs: list[CleanDocument],
    report: PipelineReport,
    rag: RagManager,
    ledger: LedgerStore,
    in_dir: Path,
) -> dict:
    """台账增量入库：过期清理 → hash 判重跳过 → 变更文档按 source 删旧插新。

    返回 {"removed": [来源...], "rebuilt": [来源...], "skipped": 份数} 供摘要打印。
    """
    current_sources = {cd.metadata["source"] for cd in clean_docs}
    children_by_container: dict[str, set[str]] = {}
    for cd in clean_docs:
        container = cd.metadata.get("container")
        if container:
            children_by_container.setdefault(container, set()).add(cd.metadata["source"])

    # 1) 过期清理：文件删除 / 容器子文档移除 / 本轮被淘汰（去重·隔离·DLQ）的文档
    removals = set(ledger.stale_sources(current_sources, children_by_container, in_dir=in_dir))
    eliminated = {
        d.path
        for d in report.docs
        if d.status in (STATUS_SUPERSEDED, STATUS_QUARANTINED, STATUS_DLQ)
    }
    removals |= {s for s in eliminated if ledger.get(s) is not None}
    for source in sorted(removals):
        for scheme in rag.schemes.values():
            scheme.store.delete_source(source)
        ledger.remove(source)

    # 2) 增量判重：内容 hash 未变的文档跳过（不重复向量化）
    changed: list[CleanDocument] = []
    for cd in clean_docs:
        if ledger.get(cd.metadata["source"]) == ledger.hash_of(cd.text):
            continue
        changed.append(cd)

    # 3) 变更/新增文档：先删旧块再插新块，逐方案执行，台账记录新 hash
    for cd in changed:
        source = cd.metadata["source"]
        for scheme in rag.schemes.values():
            scheme.ingest_document(cd.text, source)
        ledger.update(source, ledger.hash_of(cd.text), container=cd.metadata.get("container"))

    return {
        "removed": sorted(removals),
        "rebuilt": [cd.metadata["source"] for cd in changed],
        "skipped": len(clean_docs) - len(changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="真实文档前置处理 + RAG 增量建库")
    parser.add_argument("--input", default=None, help="输入目录（默认 data/docs）")
    parser.add_argument("--force", action="store_true", help="清空集合与台账，全量重建")
    parser.add_argument("--schemes", default=None, help="逗号分隔的方案 id（缺省为配置的全部方案）")
    args = parser.parse_args()

    in_dir = Path(args.input) if args.input else DEFAULT_INPUT_DIR
    out_dir = DEFAULT_OUTPUT_DIR

    # 1. 前置处理：解析 + 清洗 + 报告
    clean_docs, report = run_pipeline(in_dir, out_dir)
    summary = report.summary()
    print(
        f"[preprocess] 共 {summary['total']} 份："
        f"入库候选 {summary['ok']}、重复 {summary['superseded']}、"
        f"隔离 {summary['quarantined']}、失败(DLQ) {summary['dlq']}"
    )
    for doc in report.docs:
        mark = {"ok": "✓", "superseded": "≈", "quarantined": "!", "dlq": "✗", "container": "▸"}.get(doc.status, " ")
        line = f"  {mark} {doc.path} [{doc.status}]"
        if doc.quality_score is not None:
            line += f" 质量分={doc.quality_score}"
        if doc.error:
            line += f" — {doc.error}"
        print(line)

    # 2. 台账增量入库：无可入库文档且台账为空（全新空跑）直接结束
    ledger = LedgerStore(out_dir / "ledger.json")
    if not clean_docs and not ledger.sources():
        print("[ingest] 无可入库文档，结束。")
        return

    scheme_ids = [s.strip() for s in args.schemes.split(",")] if args.schemes else None
    embeddings = create_embeddings(fake=False)  # 未配 Embedding Key 时抛 ConfigError
    rag = RagManager(settings, embeddings, top_k=settings.rag_top_k, scheme_ids=scheme_ids)
    if args.force:
        for scheme in rag.schemes.values():
            scheme.store.clear()
        ledger.clear()
    elif not ledger.sources() and any(len(scheme.store) > 0 for scheme in rag.schemes.values()):
        # 首次启用台账但集合已有历史数据（旧版全量重建的块无真实 source）：清空重建保证归属一致
        for scheme in rag.schemes.values():
            scheme.store.clear()
        print("[ledger] 首次启用台账：清空历史集合，全量重建")

    result = run_incremental_ingest(clean_docs, report, rag, ledger, in_dir)
    ledger.save()

    if result["removed"]:
        print(f"[ledger] 清理过期来源 {len(result['removed'])} 条")
        for source in result["removed"]:
            print(f"  - {source}")
    print(f"[ledger] 未变更跳过 {result['skipped']} 份，重建 {len(result['rebuilt'])} 份")
    for source in result["rebuilt"]:
        print(f"  + {source}")
    for entry in rag.list():
        print(f"[ingest] {entry['collection']}: {entry['count']} 条")


if __name__ == "__main__":
    main()
