"""朴素 RAG 方案：固定长度切块 + 纯稠密向量语义检索（最简基线）。

严格保持 Naive RAG 的「朴素」特征：
- 固定长度切块：长文本按字符数硬切，不做语义感知 → 长规则被拆散、语义断裂，
  正是后续 Advanced(语义分块) / Graph(图谱) 等升级要解决的缺陷；
- 纯稠密向量检索：不做查询改写、混合检索、重排、多跳推理。
"""
from __future__ import annotations

from typing import Any

from app.rag.base import RagScheme

# 固定切块长度（字符）：长文本被硬切为多块，演示固定切块切断长文本语义的缺陷
CHUNK_SIZE = 150
# 固定切块无重叠，保持最朴素形态（重叠/语义分块属 Advanced 阶段的优化）
CHUNK_OVERLAP = 0


class NaiveRagScheme(RagScheme):
    """Naive RAG：固定切块 + 纯稠密相似度检索。"""

    id: str = "naive"
    name: str = "朴素 RAG"
    description: str = "固定切块 + 纯稠密向量检索（最简基线）"

    def ingest(self, texts: list[str]) -> None:
        expected = [chunk for text in texts for chunk in self._fixed_chunks(text)]
        self._rebuild_if_changed(expected)

    @staticmethod
    def _fixed_chunks(text: str) -> list[str]:
        """固定长度切块：按字符数硬切，无语义感知、无重叠。"""
        if len(text) <= CHUNK_SIZE:
            return [text]
        return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.store.search(query, top_k or self.top_k)
