"""阶段四：文档级去重（精确 SHA256 + 近似 bottom-k sketch）。

知识库中同一制度的多版本并存会让 LLM 同时看到新旧限额，回答自相矛盾。
- 精确去重：归一化文本 SHA256 一致 → 只留一份；
- 近似去重：5-gram shingle 的 bottom-k（k=128 最小哈希）Jaccard 估计
  ≥ 0.85 → 判定为近似重复；
- 版本策略：保留 mtime 最新者（调用方按旧→新传入），旧版标记 superseded
  （不删除，合规场景可追溯历史版本）。

bottom-k 是无偏 Jaccard 估计器（标准差 ≈ √(J(1-J)/k) ≈ 0.03），比置换
MinHash 更简单稳健，且纯 Python 实现、不引入 datasketch 依赖。
"""
from __future__ import annotations

import hashlib
import re

SKETCH_SIZE = 128
SHINGLE_SIZE = 5
DUPLICATE_THRESHOLD = 0.85

_WHITESPACE = re.compile(r"\s+")


def _shingles(text: str) -> set[int]:
    """字符级 5-gram → 64 位哈希集合。"""
    compact = _WHITESPACE.sub("", text)
    if not compact:
        return set()
    if len(compact) < SHINGLE_SIZE:
        return {int.from_bytes(hashlib.blake2b(compact.encode(), digest_size=8).digest(), "big")}
    return {
        int.from_bytes(hashlib.blake2b(compact[i : i + SHINGLE_SIZE].encode(), digest_size=8).digest(), "big")
        for i in range(len(compact) - SHINGLE_SIZE + 1)
    }


def exact_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def signature(text: str) -> tuple[int, ...]:
    """bottom-k sketch：shingle 哈希集合中最小的 k 个（有序元组）。"""
    return tuple(sorted(_shingles(text))[:SKETCH_SIZE])


def jaccard_estimate(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """bottom-k Jaccard 估计：并集的 k 最小哈希中属于 A 的占比。"""
    set_a = set(sig_a)
    union_bottom = sorted(set(sig_a) | set(sig_b))[:SKETCH_SIZE]
    if not union_bottom:
        return 1.0  # 两边都为空 → 视为完全一致
    return sum(1 for h in union_bottom if h in set_a) / len(union_bottom)


def find_duplicates(texts: list[str]) -> dict[int, int]:
    """返回 {被淘汰旧版下标: 保留新版下标}（链式覆盖已解析）。

    texts 顺序应为旧→新：内容重复时，新版本代表该文档进入 kept 集合，
    旧版本标记淘汰；若新版本之后又出现更新版本，链式解析保证最终
    每个旧版都指向最新的保留版。
    """
    superseded: dict[int, int] = {}
    kept: list[tuple[tuple[int, ...], int, str]] = []  # (sketch, 下标, 精确 hash)

    for i, text in enumerate(texts):
        h = exact_hash(text)
        target = None
        sketch: tuple[int, ...] | None = None
        for k_sketch, k_idx, k_hash in kept:
            if k_hash == h:  # 精确重复
                target = k_idx
                break
            if sketch is None:
                sketch = signature(text)
            if jaccard_estimate(sketch, k_sketch) >= DUPLICATE_THRESHOLD:  # 近似重复
                target = k_idx
                break
        if sketch is None:
            sketch = signature(text)
        if target is not None:
            superseded[target] = i
            kept = [(s, j, hh) for (s, j, hh) in kept if j != target]  # 新版取代旧版
        kept.append((sketch, i, h))

    # 链式覆盖解析：0→1 且 1→2 ⇒ 0→2（最终都指向最新保留版）
    resolved: dict[int, int] = {}
    for old, new in superseded.items():
        visited = {old}
        while new in superseded and new not in visited:
            visited.add(new)
            new = superseded[new]
        resolved[old] = new
    return resolved
