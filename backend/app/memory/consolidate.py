"""轮末自动提取巩固：把本轮对话中值得长期记住的事实提炼并写入记忆库。

对齐 3.5 提取模板：只提取用户画像/偏好/项目决策/外部资源；
每条由 LLM 判定 scope（global=跨会话长期偏好/约束 → 写常驻库；session=本会话临时上下文 → 写会话库）；
importance < memory_consolidate_min_importance 丢弃；经 store.add 语义去重自动纠偏。
整体 try/except 吞错，任何失败都不影响主对话链路（记忆是增强项，非必要项）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

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
    """轮末提取巩固：门控开关；提取 + 过滤 + 按 scope 分流落库；任何异常静默吞掉。

    scope=global 的条目写入 constant_store（当前客户端的常驻库，跨会话生效）；
    scope=session 的条目写入 store（当前会话库）。返回实际写入的记忆条目列表（含 scope）。
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
    for item in items:
        if item["importance"] < min_importance:
            continue
        target = constant_store if (item["scope"] == "global" and constant_store is not None) else store
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
    return written
