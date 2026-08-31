"""建库脚本：真实文档前置处理（解析 + 清洗）→ 各 RAG 方案入库。

前置链路：扫描 data/docs/ → 格式识别 → 复杂度路由（扫描件走 qwen3.5-flash OCR）
→ 五阶段清洗（归一化/页眉页脚/乱码/质量/去重）→ 干净文本交给 RagManager 入库。
处理报告落盘 data/ingest/report.json，失败文档归档 data/ingest/dlq/。

在 backend/ 目录下执行：
  python scripts/ingest_documents.py [--input data/docs] [--force]
    [--schemes naive,advanced,modular]
与内嵌语料通道（scripts/ingest_*.py → build_corpus）互不影响。
"""
import argparse

from app.config import settings
from app.llm.client import create_embeddings
from app.rag.manager import RagManager
from app.rag.preprocess import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="真实文档前置处理 + RAG 建库")
    parser.add_argument("--input", default=None, help="输入目录（默认 data/docs）")
    parser.add_argument("--force", action="store_true", help="强制清空后重建（忽略幂等）")
    parser.add_argument("--schemes", default=None, help="逗号分隔的方案 id（缺省为配置的全部方案）")
    args = parser.parse_args()

    # 1. 前置处理：解析 + 清洗 + 报告
    clean_docs, report = run_pipeline(args.input)
    summary = report.summary()
    print(
        f"[preprocess] 共 {summary['total']} 份："
        f"入库候选 {summary['ok']}、重复 {summary['superseded']}、"
        f"隔离 {summary['quarantined']}、失败(DLQ) {summary['dlq']}"
    )
    for doc in report.docs:
        mark = {"ok": "✓", "superseded": "≈", "quarantined": "!", "dlq": "✗", "": " "}.get(doc.status, " ")
        line = f"  {mark} {doc.path} [{doc.status}]"
        if doc.quality_score is not None:
            line += f" 质量分={doc.quality_score}"
        if doc.error:
            line += f" — {doc.error}"
        print(line)
    if not clean_docs:
        print("[ingest] 无可入库文档，结束。")
        return

    # 2. 入库：交给各 RAG 方案（复用现有幂等机制）
    scheme_ids = [s.strip() for s in args.schemes.split(",")] if args.schemes else None
    embeddings = create_embeddings(fake=False)  # 未配 Embedding Key 时抛 ConfigError
    rag = RagManager(settings, embeddings, top_k=settings.rag_top_k, scheme_ids=scheme_ids)
    if args.force:
        for scheme in rag.schemes.values():
            scheme.store.clear()
    rag.ingest_all([cd.text for cd in clean_docs])
    for entry in rag.list():
        print(f"[ingest] {entry['collection']}: {entry['count']} 条")


if __name__ == "__main__":
    main()
