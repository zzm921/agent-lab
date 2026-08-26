"""重排序（Rerank）：对多路召回候选做精排，把真正相关的片段顶到前面。

- DashScopeReranker：交叉编码器（qwen3-rerank），对 (query, 候选) 逐对打分，精度最高；
- LexicalReranker：离线确定性回退（词法重叠 + 原分融合），保证未开通重排模型/离线可测。
- build_reranker：有 Embedding API Key 优先交叉编码器，否则词法回退（对齐稀疏向量回退模式）。
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# 交叉编码器失败仅警告一次，避免检索时反复刷屏
_RERANK_FALLBACK_WARNED = False


def _warn_rerank_fallback(model: str, exc) -> None:
    global _RERANK_FALLBACK_WARNED
    if _RERANK_FALLBACK_WARNED:
        return
    _RERANK_FALLBACK_WARNED = True
    logger.warning("重排模型 %s 不可用（%s），advanced 方案回退本地词法重排（仅警告一次）", model, exc)


class Reranker(ABC):
    """重排序抽象：输入检索命中的候选，输出按精排分数降序的新列表。"""

    @abstractmethod
    def rerank(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """重排 hits 并更新每个 hit 的 score（越大越相关），返回新列表（不修改入参）。"""


class LexicalReranker(Reranker):
    """离线确定性重排：原检索分 + 查询/文本字符二元组重合度加权融合。"""

    _W_BASE = 0.5
    _W_LEX = 0.5

    def rerank(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q_bigrams = self._bigrams(query)
        if not q_bigrams:
            return list(hits)
        scored = []
        for h in hits:
            base = self._normalize(h.get("score"))
            overlap = self._overlap(q_bigrams, self._bigrams(h.get("text", "")))
            scored.append(dict(h, score=round(self._W_BASE * base + self._W_LEX * overlap, 4)))
        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored

    @staticmethod
    def _bigrams(text: str) -> list[str]:
        """中文无空格语言的字符二元组（含英文单词），作为词法信号。"""
        grams = [text[i : i + 2] for i in range(len(text) - 1)]
        grams += re.findall(r"[A-Za-z0-9]+", text)
        return grams

    @staticmethod
    def _overlap(a: list[str], b: list[str]) -> float:
        if not a:
            return 0.0
        count = sum(1 for g in a if g in b)
        return count / len(a)

    @staticmethod
    def _normalize(score) -> float:
        """把原始分数（Qdrant cosine/RRF 融合等）归一到 [0,1]，无分数时取 0。"""
        try:
            s = float(score)
        except (TypeError, ValueError):
            return 0.0
        if s >= 0:
            return max(0.0, min(1.0, s))
        return 0.0


class DashScopeReranker(Reranker):
    """交叉编码器重排：DashScope qwen3-rerank，对 (query, 候选) 逐对打分。

    任何异常（未开通/网络失败/导入失败）回退 LexicalReranker，保证检索链路不中断。
    """

    def __init__(self, api_key: str = "", model: str = "qwen3-rerank"):
        self.api_key = api_key
        self.model = model
        self._fallback = LexicalReranker()

    def rerank(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hits:
            return []
        try:
            return self._call(query, hits)
        except Exception as exc:  # noqa: BLE001 — 交叉编码器不可用时静默回退词法重排
            _warn_rerank_fallback(self.model, exc)
            return self._fallback.rerank(query, hits)

    def _call(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import dashscope  # 延迟导入：无该包时走词法回退

        rerank_cls = getattr(dashscope, "TextReRank", None) or getattr(dashscope, "Rerank", None)
        if rerank_cls is None:
            raise ImportError("dashscope 未提供 TextReRank/Rerank")
        if self.api_key:
            dashscope.api_key = self.api_key
        resp = rerank_cls.call(
            model=self.model,
            query=query,
            documents=[h.get("text", "") for h in hits],
            top_n=len(hits),
            return_documents=False,
        )
        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(f"重排调用失败(status={getattr(resp, 'status_code', None)})")
        results = (resp.output or {}).get("results") if isinstance(resp.output, dict) else getattr(resp.output, "results", None)
        if not results:
            raise RuntimeError("重排结果为空")
        by_index = {int(r["index"]): float(r["relevance_score"]) for r in results}
        reordered = [dict(hits[i], score=round(by_index[i], 4)) for i in range(len(hits)) if i in by_index]
        reordered.sort(key=lambda h: h["score"], reverse=True)
        return reordered


def build_reranker(embeddings, model: str = "qwen3-rerank") -> Reranker:
    """有 Embedding API Key 优先交叉编码器重排，否则本地词法重排（离线可测）。"""
    api_key = getattr(embeddings, "api_key", "") or ""
    if api_key:
        return DashScopeReranker(api_key=api_key, model=model)
    return LexicalReranker()
