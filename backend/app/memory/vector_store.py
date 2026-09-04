"""内存向量存储：基于 Embedding + 余弦相似度的检索，供 RAG 与长期记忆共用。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class VectorStore:
    """把文本向量化并存储，支持按查询返回最相关的 top-k 结果。

    长期记忆（LongMemoryStore）依赖其持久化能力：save/load 以 JSONL 保存
    「文本 + 元数据 + 向量」，load 直接重建内存索引（不重算 embedding）。
    """

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

    def update(self, index: int, text: str | None = None, metadata: dict | None = None) -> None:
        """按索引更新文本/元数据；文本变更时重算向量（保持语义索引正确）。"""
        if text is not None:
            self.texts[index] = text
            self.vectors[index] = self.embeddings.embed_query(text)
        if metadata is not None:
            self.metadatas[index] = metadata

    def delete(self, index: int) -> None:
        del self.texts[index]
        del self.metadatas[index]
        del self.vectors[index]

    def search(self, query: str, top_k: int = 3, threshold: float = 0.0) -> list[dict[str, Any]]:
        """按余弦相似度返回 top-k；score < threshold 的命中被过滤（0 即不过滤）。"""
        if not self.texts:
            return []
        qv = np.array(self.embeddings.embed_query(query), dtype=float)
        scored: list[tuple[float, int]] = []
        for i, v in enumerate(self.vectors):
            a = np.array(v, dtype=float)
            denom = float(np.linalg.norm(a) * np.linalg.norm(qv))
            score = float(np.dot(a, qv) / denom) if denom else 0.0
            if score >= threshold:
                scored.append((score, i))
        scored.sort(key=lambda t: t[0], reverse=True)
        ranked = scored[:top_k]
        return [
            {
                "text": self.texts[i],
                "score": round(score, 4),
                "metadata": self.metadatas[i],
            }
            for score, i in ranked
        ]

    def save(self, path: str) -> None:
        """JSONL 持久化：每行 {text, metadata, vector}（向量随记录落盘）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for text, meta, vec in zip(self.texts, self.metadatas, self.vectors):
                fh.write(
                    json.dumps({"text": text, "metadata": meta, "vector": vec}, ensure_ascii=False) + "\n"
                )

    def load(self, path: str) -> None:
        """从 JSONL 载入（重建列表，不调用 embedding）。"""
        p = Path(path)
        if not p.exists():
            return
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.texts.append(rec["text"])
                self.metadatas.append(rec.get("metadata") or {})
                self.vectors.append(rec["vector"])

    def __len__(self) -> int:
        return len(self.texts)
