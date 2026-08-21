"""内存向量存储：基于 Embedding + 余弦相似度的检索，供 RAG 与长期记忆共用。"""
from __future__ import annotations

from typing import Any

import numpy as np


class VectorStore:
    """把文本向量化并存储，支持按查询返回最相关的 top-k 结果。"""

    def __init__(self, embeddings, name: str = "default"):
        self.embeddings = embeddings
        self.name = name
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self.vectors: list[list[float]] = []

    def add(self, text: str, metadata: dict | None = None) -> None:
        vec = self.embeddings.embed_query(text)
        self.texts.append(text)
        self.metadatas.append(metadata or {})
        self.vectors.append(vec)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.texts:
            return []
        qv = np.array(self.embeddings.embed_query(query), dtype=float)
        scores: list[float] = []
        for v in self.vectors:
            a = np.array(v, dtype=float)
            denom = float(np.linalg.norm(a) * np.linalg.norm(qv))
            score = float(np.dot(a, qv) / denom) if denom else 0.0
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "text": self.texts[i],
                "score": round(scores[i], 4),
                "metadata": self.metadatas[i],
            }
            for i in ranked
        ]

    def __len__(self) -> int:
        return len(self.texts)
