"""上下文压缩（Context Compression）：多路召回后控制进入 LLM 的上下文噪声。

对齐 Modular RAG 企业级架构的「后处理模块组-上下文压缩（6.2）」：
多路召回必然带来重复与噪声，进入 LLM 的上下文应当「既相关又精简」——
- 去重：跨路/跨子查询按文本去重（保留最高分）；
- 截断：按 top_k 只保留分数最高的若干条；
- 超长截断：对单条超长 chunk 做提取式截断（优先句边界），避免超长块稀释向量语义。

压缩是幂等的：未发生去重/截断/超长时，metrics.original == metrics.kept 且 truncated == 0，
调用方据此决定是否在前端标注「压缩」徽标。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

# 句边界：中文句号/问号/感叹号/分号 + 换行
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；；])\s*|\n+")


class ContextCompressor(ABC):
    """上下文压缩抽象：输入候选命中，输出精简后的命中 + 压缩统计。"""

    @abstractmethod
    def compress(
        self, query: str, hits: list[dict[str, Any]], top_k: int
    ) -> tuple[list[dict[str, Any]], dict]:
        """返回 (压缩后 hits, metrics)；metrics = {"original", "kept", "truncated"}。"""


class ExtractiveContextCompressor(ContextCompressor):
    """提取式压缩：去重（精确 + 语义）+ top_k 截断 + 超长块句边界截断（纯本地，确定性）。

    embeddings：提供时追加语义去重——与已保留块向量相似度超过 semantic_threshold 的
    重复表达只保留分数最高的一条（真 Embedding 下生效；未提供则仅精确去重）。
    """

    def __init__(self, max_chars: int = 400, embeddings=None, semantic_threshold: float = 0.95):
        self.max_chars = max_chars
        self.embeddings = embeddings
        self.semantic_threshold = semantic_threshold

    def compress(
        self, query: str, hits: list[dict[str, Any]], top_k: int
    ) -> tuple[list[dict[str, Any]], dict]:
        original = len(hits)
        truncated = 0

        # 1) 精确去重（保留最高分）
        deduped: dict[str, dict[str, Any]] = {}
        for h in hits:
            text = h.get("text", "")
            if text not in deduped or self._score(h) > self._score(deduped[text]):
                deduped[text] = h
        unique = sorted(deduped.values(), key=self._score, reverse=True)

        # 2) 语义去重：跨路召回的同义重复表达（换词复述）按向量相似度滤除，只留最高分
        if self.embeddings is not None:
            try:
                unique = self._semantic_dedup(unique)
            except Exception:  # noqa: BLE001 — 语义去重为增强项，Embedding 失败时回退精确去重结果
                pass

        # 3) top_k 截断 + 超长截断（原文保留至 metadata["raw_text"]，供溯源与评测判定）
        kept = []
        for h in unique[: max(1, top_k)]:
            text = h.get("text", "") or ""
            if len(text) > self.max_chars:
                truncated_text = self._truncate(text)
                meta = dict(h.get("metadata") or {})
                meta.setdefault("raw_text", text)
                kept.append(dict(h, text=truncated_text, metadata=meta))
                truncated += 1
            else:
                kept.append(dict(h))

        metrics = {"original": original, "kept": len(kept), "truncated": truncated}
        return kept, metrics

    def _semantic_dedup(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按分数降序贪心去重：与已保留块余弦相似度超阈值即丢弃（保留更高分）。"""
        kept: list[dict[str, Any]] = []
        kept_vecs: list[list[float]] = []
        for h in hits:
            vec = self.embeddings.embed_query(h.get("text", "") or "")
            if any(self._cosine(vec, kept_vec) > self.semantic_threshold for kept_vec in kept_vecs):
                continue
            kept.append(h)
            kept_vecs.append(vec)
        return kept

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _score(h: dict[str, Any]) -> float:
        try:
            return float(h.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _truncate(self, text: str) -> str:
        """超长块截断：优先按空行分段（条文/条目天然分段），预算内整段保留、不留半条；
        被丢弃段落以标题提示形式附在末尾，避免检索证据（条文标题）随截断丢失。
        无分段结构时按句边界截断；无边界时硬切。"""
        parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(parts) > 1:
            kept_parts: list[str] = []
            used = 0
            for p in parts:
                if kept_parts and used + len(p) > self.max_chars:
                    break
                kept_parts.append(p)
                used += len(p) + 2
            if len(kept_parts) < len(parts):
                dropped_titles = [
                    self._head_title(p) for p in parts[len(kept_parts):]
                ]
                head = "\n\n".join(kept_parts)
                head = head.rstrip() + "\n…（后文略：" + "；".join(dropped_titles) + "）"
                return head
            return text

        head = ""
        for sentence in _SENTENCE_BOUNDARY.split(text):
            if len(head) + len(sentence) > self.max_chars:
                break
            head += sentence
        if not head:
            head = text[: self.max_chars]
        if len(head) < len(text):
            head = head.rstrip() + "…"
        return head

    @staticmethod
    def _head_title(part: str) -> str:
        """段落标题：首行剥掉「第X条（…）」以外的正文，保留标题与首句要点（≤24 字）。"""
        first = part.strip().split("\n", 1)[0].strip()
        m = re.match(r"^(第[一二三四五六七八九十百零]+条（[^）]+）)", first)
        title = m.group(1) if m else first
        return title[:24]


def build_compressor(max_chars: int = 400, embeddings=None, semantic_threshold: float = 0.95) -> ContextCompressor:
    """构造提取式上下文压缩器（纯本地，无外部依赖；embeddings 提供时启用语义去重）。"""
    return ExtractiveContextCompressor(
        max_chars=max_chars, embeddings=embeddings, semantic_threshold=semantic_threshold
    )
