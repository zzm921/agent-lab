"""清洗编排：单文档五阶段顺序执行（归一化 → 页眉页脚 → 乱码 → 质量）。

去重为跨文档阶段，由全链路 orchestrator（preprocess/pipeline.py）执行。
乱码/空文本抛 GarbledDocument（进 DLQ），正常返回 (text, stats)。
"""
from __future__ import annotations

from app.rag.preprocess.cleaning.boilerplate import remove_boilerplate
from app.rag.preprocess.cleaning.garble import check_garble
from app.rag.preprocess.cleaning.normalizer import normalize
from app.rag.preprocess.cleaning.quality import score_text
from app.rag.preprocess.models import GarbledDocument, ParsedDocument


def elements_to_text(elements) -> str:
    """结构元素 → 纯文本：标题转 Markdown 层级，表格原样保留。"""
    lines: list[str] = []
    for el in elements:
        if el.type == "title":
            lines.append("#" * min(el.level or 1, 6) + " " + el.text)
        elif el.type == "text":
            lines.append(el.text)
        elif el.type == "table":
            lines.append(el.text)
    return "\n\n".join(lines)


def clean_document(parsed: ParsedDocument) -> tuple[str, dict]:
    """清洗单个已解析文档，返回 (干净文本, 各阶段统计)。乱码抛 GarbledDocument。"""
    stats: dict = {}

    # 阶段 1：归一化 + 断行修复（在元素文本上逐元素处理，保持结构）
    norm_stats = {"removed_control": 0, "merged_lines": 0}
    for el in parsed.elements:
        if el.type in ("text", "table"):
            el.text, s = normalize(el.text)
            norm_stats["removed_control"] += s["removed_control"]
            norm_stats["merged_lines"] += s["merged_lines"]
    stats["normalize"] = norm_stats

    # 阶段 2：页眉页脚 / 页码移除
    parsed.elements, bp_stats = remove_boilerplate(parsed)
    stats["boilerplate"] = bp_stats

    # 阶段 3：乱码检测（在拼装后的全文上判定，口径统一）
    text = elements_to_text(parsed.elements)
    garbled, garble_stats = check_garble(text)
    stats["garble"] = garble_stats
    if garbled:
        raise GarbledDocument(garble_stats.get("reason", "乱码检测未通过"))

    # 阶段 5：质量评分（分流决定由 orchestrator 做：分数透传报告）
    score, quality_stats = score_text(text)
    stats["quality"] = quality_stats
    stats["quality_score"] = score
    return text, stats
