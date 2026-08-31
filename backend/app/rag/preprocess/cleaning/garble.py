"""阶段三：乱码检测与拦截。

检测到乱码直接拒绝入库（进 DLQ），不「凑合入库」——一个空结果比一个
错误结果更安全：乱码文本的嵌入向量是随机噪声，可能在向量空间中意外
匹配到不相关查询；BM25 侧用户搜任何内容都命中不了。
"""
from __future__ import annotations

import re

from app.rag.preprocess.models import GarbledDocument

# 判定阈值（与《复杂情况应对手册》保持同步）
REPLACEMENT_RATIO = 0.03  # Unicode 替换符 � 占比
MOJIBAKE_RATIO = 0.05  # GBK→UTF8 mojibake 特征字符占比

# mojibake 高频特征字符：GBK 字节被 UTF-8 误解码后大量出现的 Latin-1 扩展区字符
_MOJIBAKE_CHARS = set("æøåþðß€‚„…†‡ˆ‰‹Œ‘’“”•–—˜™š›œžŸ¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿×÷")
# 常见 mojibake 双字节序列特征（如 æ–‡ ä»¶ ç®¡ç†）
_MOJIBAKE_SEQ = re.compile(r"[æçåèéäßø][\x80-\xff\u0080-\u00ff]", re.IGNORECASE)


def check_garble(text: str) -> tuple[bool, dict]:
    """返回 (是否乱码/空, 统计)。判定为乱码时由管线抛 GarbledDocument。"""
    stripped = text.strip()
    stats = {"chars": len(stripped)}
    if not stripped:
        stats["reason"] = "提取文本为空（可能是无法识别的格式或空白文档）"
        return True, stats

    n = len(stripped)
    replacement_ratio = stripped.count("\ufffd") / n
    stats["replacement_ratio"] = round(replacement_ratio, 4)
    if replacement_ratio > REPLACEMENT_RATIO:
        stats["reason"] = f"替换符 � 占比 {replacement_ratio:.1%} 超过阈值 {REPLACEMENT_RATIO:.0%}（编码错误）"
        return True, stats

    mojibake_hits = sum(1 for ch in stripped if ch in _MOJIBAKE_CHARS)
    mojibake_hits += len(_MOJIBAKE_SEQ.findall(stripped))
    mojibake_ratio = mojibake_hits / n
    stats["mojibake_ratio"] = round(mojibake_ratio, 4)
    if mojibake_ratio > MOJIBAKE_RATIO:
        stats["reason"] = f"mojibake 特征占比 {mojibake_ratio:.1%} 超过阈值 {MOJIBAKE_RATIO:.0%}（GBK 被 UTF-8 误解码）"
        return True, stats

    # 全文档既无 CJK 也无字母数字 → 纯符号噪声
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", stripped):
        stats["reason"] = "文本不含任何有效文字（仅符号/噪声）"
        return True, stats

    return False, stats
