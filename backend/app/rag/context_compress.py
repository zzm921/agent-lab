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
    """提取式压缩：去重 + top_k 截断 + 超长块句边界截断（纯本地，确定性）。"""

    def __init__(self, max_chars: int = 200):
        self.max_chars = max_chars

    def compress(
        self, query: str, hits: list[dict[str, Any]], top_k: int
    ) -> tuple[list[dict[str, Any]], dict]:
        original = len(hits)
        truncated = 0

        # 1) 去重（保留最高分）
        deduped: dict[str, dict[str, Any]] = {}
        for h in hits:
            text = h.get("text", "")
            if text not in deduped or self._score(h) > self._score(deduped[text]):
                deduped[text] = h
        unique = sorted(deduped.values(), key=self._score, reverse=True)

        # 2) top_k 截断 + 超长截断
        kept = []
        for h in unique[: max(1, top_k)]:
            text = h.get("text", "") or ""
            if len(text) > self.max_chars:
                text = self._truncate(text)
                truncated += 1
            kept.append(dict(h, text=text))

        metrics = {"original": original, "kept": len(kept), "truncated": truncated}
        return kept, metrics

    @staticmethod
    def _score(h: dict[str, Any]) -> float:
        try:
            return float(h.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _truncate(self, text: str) -> str:
        """按句边界截到 max_chars，超出加省略号；无边界时硬切。"""
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


def build_compressor(max_chars: int = 200) -> ContextCompressor:
    """构造提取式上下文压缩器（纯本地，无外部依赖）。"""
    return ExtractiveContextCompressor(max_chars=max_chars)
