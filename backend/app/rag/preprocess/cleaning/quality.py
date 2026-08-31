"""阶段五：质量评分与分流。

日志片段这类「碎片文档」不是没用，而是不该混进主索引污染通用问答：
- score ≥ 70 → 正常入库；
- 50-69    → quarantined（隔离，不入主索引，原文件保留）；
- < 50     → DLQ（近乎无有效信息，入库只会带来噪声）。

不直接删除的原因：隔离区的文档在专门场景（如排障日志检索）仍可能有用。
"""
from __future__ import annotations

import re

# 分流阈值（与《复杂情况应对手册》保持同步）
SCORE_PASS = 70
SCORE_QUARANTINE = 50

# 日志/时间戳行特征
_LOG_LINE = re.compile(r"\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\]|^\s*(INFO|WARN|ERROR|DEBUG)\b", re.MULTILINE)
_SENTENCE_END = "。！？.!?"


def score_text(text: str) -> tuple[int, dict]:
    """返回 (0-100 质量分, 分项明细)。"""
    stripped = text.strip()
    breakdown = {"chars": len(stripped)}
    if not stripped:
        return 0, breakdown

    # 1) 有效体量：≥500 字满分，线性衰减到 0 字
    volume = min(len(stripped) / 500, 1.0)

    # 2) 完整句占比：以句末标点收尾的句子 / 总句子
    sentences = [s for s in re.split(r"[。！？.!?]\s*", stripped) if s.strip()]
    if sentences:
        complete = sum(1 for s in re.findall(r"[^。！？.!?]*[。！？.!?]", stripped) if len(s.strip()) >= 8)
        complete_ratio = min(complete / max(len(sentences), 1), 1.0)
    else:
        complete_ratio = 0.0
    breakdown["complete_ratio"] = round(complete_ratio, 2)

    # 3) 平均句长：10-100 字最佳（过短碎片化、过长难读）
    avg_len = len(stripped) / max(len(sentences), 1)
    if 10 <= avg_len <= 100:
        length_score = 1.0
    else:
        length_score = max(0.0, 1 - abs(avg_len - 55) / 200)
    breakdown["avg_sentence_len"] = round(avg_len, 1)

    # 4) 信息密度：时间戳/日志行占比越低越好
    log_lines = len(_LOG_LINE.findall(stripped))
    total_lines = max(stripped.count("\n") + 1, 1)
    noise = min(log_lines / total_lines, 1.0)
    breakdown["noise_ratio"] = round(noise, 2)

    score = round(
        100 * (0.35 * volume + 0.30 * complete_ratio + 0.20 * length_score + 0.15 * (1 - noise))
    )
    return score, breakdown
