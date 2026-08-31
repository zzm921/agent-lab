"""阶段一：Unicode 归一化与断行修复。

- NFKC 归一化 + 去零宽/控制字符：同一字符多种写法统一，保证后续
  hash 去重、重复行检测口径一致；
- 断行合并：行尾无标点且次行为正文续写时合并，防止「规章/制度」
  这类词被 PDF 硬换行切断，导致向量与 BM25 都匹配不上。
"""
from __future__ import annotations

import re
import unicodedata

# 零宽字符与 BOM
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
# 控制字符（保留 \n \t）
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 行内空白折叠
_WS = re.compile(r"[ \t\u3000]+")
# 连续空行折叠
_BLANK = re.compile(r"\n{3,}")
# 句末标点（中英文）——行尾有标点视为完整行，不再合并次行
_END_PUNCT = ("。", "！", "？", "；", "：", ".", "!", "?", ";", ":")
# 次行起始的结构标记：标题/列表/表格/条款号——不与前一行合并
_NEXT_STRUCT = re.compile(r"^(#{1,6}\s|[-*+]\s|\d+\.\s|\||（[一二三四五六七八九十\d]+）|\d+、)")


def normalize(text: str) -> tuple[str, dict]:
    """归一化 + 断行合并，返回 (处理后文本, 统计)。"""
    stats = {"removed_control": 0, "merged_lines": 0}

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _ZERO_WIDTH.sub("", cleaned)
    before = len(cleaned)
    cleaned = _CONTROL.sub("", cleaned)
    stats["removed_control"] = before - len(cleaned)

    lines = [_WS.sub(" ", line).strip() for line in cleaned.split("\n")]
    merged: list[str] = []
    for line in lines:
        prev = merged[-1] if merged else ""
        can_merge = (
            bool(prev)
            and bool(line)
            and not prev.endswith(_END_PUNCT)  # 上一行未完结
            and not prev.startswith(("#", "|"))  # 上一行是标题/表格行，不吸收次行
            and not _NEXT_STRUCT.match(line)  # 次行是结构起始（标题/列表/表格/条款号）
        )
        if can_merge:
            merged[-1] = prev + line
            stats["merged_lines"] += 1
        else:
            merged.append(line)

    result = "\n".join(merged)
    result = _BLANK.sub("\n\n", result).strip()
    return result, stats
