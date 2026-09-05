"""轮末自动提取巩固：把本轮对话中值得长期记住的事实提炼并写入记忆库。

对齐 3.5 提取模板：只提取用户画像/偏好/项目决策/外部资源；
每条由 LLM 判定 scope（global=跨会话长期偏好/约束 → 写常驻库；session=本会话临时上下文 → 写会话库）；
importance < memory_consolidate_min_importance 丢弃。

写入采用「提取 → 匹配 → 合并」三段式（企业级 Mem0 式，替代简单覆盖）：
- 高相似度（≥ dedup_threshold）→ 直接合并：旧值入 history 归档、新表述作当前值；
- 模糊带（MERGE_LOW ~ 高阈值）→ 轻量 LLM 批量裁决 merge/conflict/add；裁决失败回退规则
  （含改口触发词 → conflict 合并归档，否则保守新增，宁重不漏）；
- 低相似度 → 视为不同事实另存。
整体 try/except 吞错，任何失败都不影响主对话链路（记忆是增强项，非必要项）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.memory.long_memory import MERGE_LOW, is_conflict_rewrite

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = (
    "从以下对话中提取值得长期记住的事实。\n"
    "只提取：用户画像、偏好、项目决策、外部资源。\n"
    "不提取：可从代码/文件/命令历史推导的信息、临时状态。\n"
    "每条输出 JSON：{text, type, importance(0~1), scope}。\n"
    "  scope=global：跨会话长期有效的偏好/约束/稳定画像（如「以后所有项目都用 X」）；\n"
    "  scope=session：仅本会话相关的临时上下文/一次性事件；\n"
    "importance 低于 0.5 的不要输出。\n"
    "输出必须严格是 JSON 数组，不要输出任何其他文字。"
)

# 参考记忆分类；无法映射的 type 一律归为 fact
_VALID_KINDS = ("fact", "preference", "episodic", "procedural")

# 模糊带合并裁决：给「已有记忆 + 新提取事实」，判定 merge（补充/更新）/ conflict（用户改口）/
# add（不同事实另存）；merge/conflict 时由 LLM 给出合并后的统一表述 text。
MERGE_JUDGE_PROMPT = (
    "你是记忆合并判断器。以下是若干组「已有记忆」与其最相似的「新提取事实」，逐条判断应如何处理：\n"
    "- merge：新事实是对已有记忆的补充/更新（同一事实的新表述）→ 输出合并后的统一表述 text；\n"
    "- conflict：新事实推翻/取代旧记忆（用户改口、纠正）→ 输出新表述 text，旧表述将归档不再召回；\n"
    "- add：新事实与已有记忆是不同事实，应另存为一条新记忆。\n"
    '只输出严格 JSON 数组，不要输出任何其他文字：\n'
    '[{"index": 0, "action": "merge|conflict|add", "reason": "一句话理由", "text": "合并后的表述"}]'
)


def _judge_payload(ambiguous: list[dict[str, Any]]) -> str:
    """把模糊带候选组装为裁决输入（编号对齐 JSON 里的 index）。"""
    lines = []
    for a in ambiguous:
        lines.append(
            f"{a['idx']}. 已有记忆：{a['existing']}\n"
            f"   新提取事实：{a['item']['text']}"
        )
    return "\n\n".join(lines)


def _parse_judgments(content: str) -> dict[int, dict[str, str]]:
    """解析裁决输出 JSON 数组 → {index: {action, reason, text}}；整体非法返回空（走规则回退）。"""
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, list):
        return {}
    out: dict[int, dict[str, str]] = {}
    for d in data:
        if not isinstance(d, dict):
            continue
        try:
            idx = int(d.get("index"))
        except (TypeError, ValueError):
            continue
        action = str(d.get("action") or "").strip().lower()
        if action not in ("merge", "conflict", "add"):
            continue
        out[idx] = {
            "action": action,
            "reason": str(d.get("reason") or ""),
            "text": str(d.get("text") or "").strip(),
        }
    return out


def _extract_items(content: str) -> list[dict[str, Any]]:
    """从 LLM 输出中提取 JSON 数组并规范化；无数组/非法则返回空列表。

    scope 非法/缺失时保守归为 session（避免误写全局常驻库）。
    """
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    items = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        kind = str(item.get("type") or "fact")
        if kind not in _VALID_KINDS:
            kind = "fact"
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        scope = str(item.get("scope") or "session").strip().lower()
        if scope not in ("global", "session"):
            scope = "session"
        items.append(
            {
                "text": text,
                "kind": kind,
                "importance": max(0.0, min(1.0, importance)),
                "scope": scope,
            }
        )
    return items


def _transcript(messages) -> str:
    """把最近若干条消息拼为提取输入（只取文本内容，截断控制 token）。"""
    lines = []
    for m in messages[-20:]:
        content = getattr(m, "content", None)
        if not content:
            continue
        speaker = "用户" if getattr(m, "type", "") == "human" else "助手"
        lines.append(f"{speaker}: {str(content)[:2000]}")
    return "\n".join(lines)


async def maybe_consolidate(store, constant_store, messages, llm, settings, session_id: str) -> list[dict[str, Any]]:
    """轮末提取巩固：门控开关；提取 + 过滤 + 按 scope 分流 + 三段式合并落库；任何异常静默吞掉。

    scope=global 的条目写入 constant_store（当前客户端的常驻库，跨会话生效）；
    scope=session 的条目写入 store（当前会话库）。返回实际写入的记忆条目列表（含 scope）。

    三段式（Mem0 式提取→匹配→合并）：
    - 高相似度（≥ dedup_threshold）→ 确定性合并（旧值入 history 归档、新表述作当前值）；
    - 模糊带（MERGE_LOW ~ dedup_threshold）→ 攒批后轻量 LLM 裁决 merge/conflict/add；
      裁决失败回退规则（含改口触发词 → conflict 合并归档，否则保守新增，宁重不漏）；
    - 低相似度 / 无匹配 → 视为不同事实另存。
    """
    if not getattr(settings, "memory_consolidate_enabled", True):
        return []
    if not messages:
        return []
    text = _transcript(messages)
    if not text:
        return []
    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=EXTRACT_PROMPT),
                HumanMessage(content=text),
            ]
        )
        content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
        items = _extract_items(content)
    except Exception as exc:  # noqa: BLE001 — 提取失败不影响主链路
        logger.warning("memory consolidate 提取失败（已忽略）: %s", exc)
        return []
    written = []
    min_importance = getattr(settings, "memory_consolidate_min_importance", 0.5)
    ambiguous: list[dict[str, Any]] = []  # 模糊带候选：攒批统一交 LLM 裁决
    for item in items:
        if item["importance"] < min_importance:
            continue
        target = constant_store if (item["scope"] == "global" and constant_store is not None) else store
        # 匹配（不触碰访问统计）：落入模糊带则攒批，其余（高/低/无匹配）直接确定性落库
        matched = target.match(item["text"], top_k=1)
        if matched:
            sim = float(matched[0].get("score", 0.0))
            if MERGE_LOW <= sim < target.dedup_threshold:
                ambiguous.append(
                    {
                        "idx": len(ambiguous),
                        "target": target,
                        "item": item,
                        "match_id": (matched[0].get("metadata") or {}).get("id"),
                        "existing": matched[0].get("text"),
                    }
                )
                continue
        try:
            target.add(
                item["text"],
                kind=item["kind"],
                importance=item["importance"],
                source_session=session_id,
            )
            written.append(item)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory consolidate 写入失败（已忽略）: %s", exc)

    # 模糊带批量裁决：轻量 LLM 判定 merge/conflict/add（一次调用处理整批）；
    # 裁决缺失/失败回退走 add 的确定性合并（高阈值合并 / 改口 conflict / 保守新增）。
    if ambiguous:
        try:
            resp = await llm.ainvoke(
                [
                    SystemMessage(content=MERGE_JUDGE_PROMPT),
                    HumanMessage(content=_judge_payload(ambiguous)),
                ]
            )
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            judgments = _parse_judgments(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory consolidate 合并裁决失败（已忽略）: %s", exc)
            judgments = {}
        for a in ambiguous:
            target = a["target"]
            item = a["item"]
            jud = judgments.get(a["idx"])
            try:
                if jud and jud["action"] in ("merge", "conflict"):
                    target.add_judged(
                        jud.get("text") or item["text"],
                        kind=item["kind"],
                        importance=item["importance"],
                        source_session=session_id,
                        decision=jud["action"],
                        reason=jud.get("reason") or jud["action"],
                        match_id=a["match_id"],
                    )
                else:  # add 或裁决缺失/失败 → 确定性回退
                    target.add(
                        item["text"],
                        kind=item["kind"],
                        importance=item["importance"],
                        source_session=session_id,
                    )
                written.append(item)
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory consolidate 裁决写入失败（已忽略）: %s", exc)
    return written
