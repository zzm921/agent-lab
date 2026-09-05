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
    """提取式压缩：去重（精确 + 语义）+ top_k 截断 + 超长块 query 相关句/段保留（纯本地，确定性）。

    embeddings：提供时追加语义去重——与已保留块向量相似度超过 semantic_threshold 的
    重复表达只保留分数最高的一条（真 Embedding 下生效；未提供则仅精确去重）。

    超长截断（企业级内容处理）：不再「从头硬切」，而是按 query 相关度选取句子/段落
    （关键词二元组重叠打分），预算内保留关键信息、按原文顺序输出——直接截断导致的
    「关键条款/差异点落在截断点之后而丢失」被消除；被丢弃段以标题提示附在末尾。
    summarizer（可选）：提供时对超长命中先生成 query 针对性摘要（LLM 摘要方案，默认关）。
    """

    def __init__(self, max_chars: int = 400, embeddings=None, semantic_threshold: float = 0.95, summarizer=None):
        self.max_chars = max_chars
        self.embeddings = embeddings
        self.semantic_threshold = semantic_threshold
        self.summarizer = summarizer  # callable(query, text) -> str；None = 关闭摘要，走相关句保留

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
                truncated_text = self._truncate(text, query)
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

    def _truncate(self, text: str, query: str) -> str:
        """超长块截断：query 相关句/段保留（不从头硬切）。

        - summarizer 已配置：先生成 query 针对性摘要（LLM 摘要方案，失败回退相关句保留）；
        - 段落级：优先按空行分段（条文/条目天然分段），按相关度选取段、原文顺序输出；
        - 句级：无分段结构时按句边界切分，按相关度选取句；
        - 均保留「后文略：标题」提示（检索证据标题不随截断丢失），原文存 metadata.raw_text。
        """
        if self.summarizer is not None:
            try:
                summary = self.summarizer(query, text)
                if summary and summary != text:
                    return (summary[: self.max_chars].rstrip() + "…") if len(summary) > self.max_chars else summary
            except Exception:  # noqa: BLE001 — 摘要失败回退相关句保留
                pass
        parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(parts) > 1:
            return self._pick_relevant(parts, query, "\n\n", self._head_title)
        sentences = [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
        if len(sentences) > 1:
            return self._pick_relevant(sentences, query, "", lambda s: s.strip()[:24])
        return text[: self.max_chars] + "…"

    def _pick_relevant(self, units: list[str], query: str, join: str, label_fn) -> str:
        """按 query 相关度选取句子/段落：预算内保留关键信息，输出按原文顺序。

        相关度 = 与 query 的中文相邻 2 字词重叠占比（词法信号，确定性、零成本）；
        全部单元均超预算（picked 为空）时回退取首个单元按句界截断。
        """
        q_terms = self._terms(query)
        rel = [self._relevance(u, q_terms) for u in units] if q_terms else [0.0] * len(units)
        order = sorted(range(len(units)), key=lambda i: rel[i], reverse=True)
        picked: list[int] = []
        used = 0
        for i in order:
            u = units[i]
            if used + len(u) > self.max_chars:
                continue
            picked.append(i)
            used += len(u) + len(join)
        if not picked:
            head = units[0][: self.max_chars]
            return head.rstrip() + "…" if len(units[0]) > self.max_chars else head
        picked.sort()
        head = join.join(units[i] for i in picked)
        if len(picked) < len(units):
            dropped_titles = [label_fn(units[i]) for i in range(len(units)) if i not in picked]
            if dropped_titles:
                head = head.rstrip() + "\n…（后文略：" + "；".join(dropped_titles) + "）"
        return head

    @staticmethod
    def _terms(text: str) -> set[str]:
        """中文相邻 2 字词集合：查询/文本相关性粗判（与 agentic tools.terms2 同口径）。"""
        seg = re.findall(r"[\u4e00-\u9fff]+", text)
        return {s[i : i + 2] for s in seg for i in range(len(s) - 1)}

    @staticmethod
    def _relevance(unit: str, q_terms: set[str]) -> float:
        """单元与查询的相关度：与查询共现的 2 字词占单元词数比（[0,1]，无词重叠=0）。"""
        if not q_terms:
            return 0.0
        ut = ExtractiveContextCompressor._terms(unit)
        if not ut:
            return 0.0
        return len(ut & q_terms) / max(1, len(ut))

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
