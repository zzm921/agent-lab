"""检索融合：倒数排名融合（Reciprocal Rank Fusion, RRF）。

多路 / 多查询 / 多跳召回的分数可能来自不同体系（纯向量余弦分 vs 混合检索内部的 RRF 分、
不同后端各自的打分），直接比较大小或取最大值没有意义。RRF 只依赖「排名位置」，
天然跨体系可比：

    rrf_score(doc) = Σ 1 / (K + rank_i)   （对每个召回该文档的列表累加）

- K 取 60（Cornack et al. 2009 提出的标准常数）；
- 出现在越多路、越靠前的文档融合分越高，恰好突出多路 / 多跳的共同证据。
"""
from __future__ import annotations

from typing import Any

# RRF 标准常数（Cornack, Clarke & Buettcher 2009）
_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = _RRF_K,
) -> list[dict[str, Any]]:
    """把多路排名列表按文本融合：去重 + 按 RRF 分降序。

    - 每路内部按出现顺序（1 基）计秩；
    - 同文本跨路累计 rrf 分（score 字段覆写为融合分，供下游排序/截断）；
    - 返回全部候选（不截断），截断由调用方按需处理。
    """
    hits_by_text: dict[str, dict[str, Any]] = {}  # 首现命中（保留 text/metadata 等字段）
    rrf_by_text: dict[str, float] = {}  # 文本 → 累计 RRF 分（与原始分无关，跨路可比）
    for lst in ranked_lists:
        for rank, hit in enumerate(lst, start=1):
            text = hit.get("text", "")
            if not text:
                continue
            hits_by_text.setdefault(text, dict(hit))
            rrf_by_text[text] = rrf_by_text.get(text, 0.0) + 1.0 / (k + rank)
    result: list[dict[str, Any]] = []
    for text, hit in hits_by_text.items():
        hit = dict(hit)
        hit["score"] = rrf_by_text[text]
        result.append(hit)
    return sorted(result, key=lambda h: h.get("score") or 0.0, reverse=True)
