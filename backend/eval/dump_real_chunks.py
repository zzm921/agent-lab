"""Dump modular 方案分块产物（本地确定性重建，不依赖网络）。

modular 建库走 AdvancedRagScheme.ingest → _structure_chunks（纯文本处理、确定性），
本脚本对同一语料按同一顺序调用同一分块函数，重建出与 Qdrant 库一致的子块列表
（text + metadata），并按语料顺序编 chunk_id（real-0001...），供金标测试集标注用。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory.corpus import KNOWLEDGE_DOCS  # noqa: E402
from app.rag.schemes.advanced import AdvancedRagScheme  # noqa: E402


def main() -> None:
    scheme = AdvancedRagScheme.__new__(AdvancedRagScheme)  # 分块为实例方法但不用存储/向量
    items: list[dict] = []
    seq = 0
    for vol_title, vol_text in KNOWLEDGE_DOCS.items():
        chunks = scheme._structure_chunks(vol_text)
        if not chunks:
            continue
        for text, meta in chunks:
            seq += 1
            items.append(
                {
                    "chunk_id": f"real-{seq:04d}",
                    "volume": vol_title,
                    "text": text,
                    "metadata": meta,
                }
            )
    out = Path(__file__).resolve().parent / "real_chunks.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"chunks: {len(items)} -> {out}")
    for m in items[:3]:
        print("meta:", json.dumps(m["metadata"], ensure_ascii=False)[:200])
    tables = [m for m in items if str(m["text"]).lstrip().startswith("|")]
    if tables:
        print("table meta:", json.dumps(tables[0]["metadata"], ensure_ascii=False)[:200])


if __name__ == "__main__":
    main()
