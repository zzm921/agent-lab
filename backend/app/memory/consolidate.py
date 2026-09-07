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


async def _consolidate_core(store, constant_store, messages, llm, settings, session_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """提取→过滤→匹配→裁决→落库 共享核心：返回 (written, report)。

    - written：实际写入的记忆条目列表（scope 合法化后），供静默轮末路径使用；
    - report：结构化中间结果报告（「记忆梦游」手动触发时逐项展示用）：
      extracted（LLM 提取原文）/ filtered（importance 过滤丢弃）/ matches（逐条匹配与相似度 /
      所属区带）/ judgments（模糊带裁决）/ written（落库明细 + 动作）/ steps（阶段摘要）/
      summary（前后数量对比）。任何异常静默吞掉（记忆是增强项）。
    """
    report: dict[str, Any] = {
        "summary": {
            "transcript_messages": 0,
            "extracted": 0,
            "kept": 0,
            "dropped": 0,
            "written": 0,
            "added": 0,
            "merged": 0,
            "conflicted": 0,
            "before": {"session": 0, "global": 0},
            "after": {"session": 0, "global": 0},
        },
        "extracted": [],
        "filtered": [],
        "matches": [],
        "judgments": [],
        "written": [],
        "steps": [],
    }
    if not messages:
        return [], report
    text = _transcript(messages)
    if not text:
        return [], report
    # 先捕获写入前数量，供报告做前后对比（before 必须在任何落库之前快照）
    report["summary"]["before"] = {"session": len(store), "global": len(constant_store) if constant_store is not None else 0}
    report["summary"]["transcript_messages"] = len(messages[-20:])
    report["steps"].append({"name": "提取", "detail": f"取最近 {report['summary']['transcript_messages']} 条消息交给 LLM 提取事实"})
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
        return [], report
    report["summary"]["extracted"] = len(items)
    report["extracted"] = [dict(it) for it in items]
    report["steps"].append({"name": "提取", "detail": f"LLM 提取出 {len(items)} 条候选事实"})
    written: list[dict[str, Any]] = []
    min_importance = getattr(settings, "memory_consolidate_min_importance", 0.5)
    ambiguous: list[dict[str, Any]] = []  # 模糊带候选：攒批统一交 LLM 裁决
    for item in items:
        if item["importance"] < min_importance:
            report["filtered"].append({**dict(item), "reason": f"importance {item['importance']:.2f} < 阈值 {min_importance}"})
            continue
        target = constant_store if (item["scope"] == "global" and constant_store is not None) else store
        target_name = "global" if (item["scope"] == "global" and constant_store is not None) else "session"
        # 匹配（不触碰访问统计）：落入模糊带则攒批，其余（高/低/无匹配）直接确定性落库
        matched = target.match(item["text"], top_k=1)
        sim = float(matched[0].get("score", 0.0)) if matched else None
        if matched and sim is not None and MERGE_LOW <= sim < target.dedup_threshold:
            zone = "ambiguous"
            report["matches"].append(
                {
                    "text": item["text"],
                    "scope": target_name,
                    "sim": round(sim, 4),
                    "zone": zone,
                    "existing": matched[0].get("text"),
                }
            )
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
        zone = "high" if (matched and sim is not None and sim >= target.dedup_threshold) else ("none" if not matched else "low")
        report["matches"].append(
            {
                "text": item["text"],
                "scope": target_name,
                "sim": round(sim, 4) if sim is not None else None,
                "zone": zone,
                "existing": matched[0].get("text") if matched else None,
            }
        )
        try:
            result = target.add(
                item["text"],
                kind=item["kind"],
                importance=item["importance"],
                source_session=session_id,
            )
            action = result.get("action", "add")
            written.append(item)
            report["written"].append(
                {
                    "text": item["text"],
                    "kind": item["kind"],
                    "importance": item["importance"],
                    "scope": target_name,
                    "action": action,
                    "reason": "同义事实，合并更新" if action == "merge" else ("用户改口，归档旧值" if action == "conflict" else "新事实，直接写入"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory consolidate 写入失败（已忽略）: %s", exc)
            report["written"].append({"text": item["text"], "kind": item["kind"], "importance": item["importance"], "scope": target_name, "action": "error", "reason": str(exc)})

    # 模糊带批量裁决：轻量 LLM 判定 merge/conflict/add（一次调用处理整批）；
    # 裁决缺失/失败回退走 add 的确定性合并（高阈值合并 / 改口 conflict / 保守新增）。
    if ambiguous:
        report["steps"].append({"name": "匹配", "detail": f"{len(ambiguous)} 条落在模糊带，交 LLM 批量裁决 merge/conflict/add"})
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
            target_name = "global" if (item["scope"] == "global" and constant_store is not None) else "session"
            jud = judgments.get(a["idx"])
            report["judgments"].append(
                {
                    "index": a["idx"],
                    "text": item["text"],
                    "existing": a["existing"],
                    "action": jud["action"] if jud else "add",
                    "reason": jud.get("reason") if jud else "裁决缺失，保守新增",
                }
            )
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
                    action = jud["action"]
                else:  # add 或裁决缺失/失败 → 确定性回退
                    target.add(
                        item["text"],
                        kind=item["kind"],
                        importance=item["importance"],
                        source_session=session_id,
                    )
                    action = "add"
                written.append(item)
                report["written"].append(
                    {
                        "text": item["text"],
                        "kind": item["kind"],
                        "importance": item["importance"],
                        "scope": target_name,
                        "action": action,
                        "reason": (jud.get("reason") if jud else None) or ("同义事实，合并更新" if action == "merge" else ("用户改口，归档旧值" if action == "conflict" else "不同事实，另存一条")),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory consolidate 裁决写入失败（已忽略）: %s", exc)
                report["written"].append({"text": item["text"], "kind": item["kind"], "importance": item["importance"], "scope": target_name, "action": "error", "reason": str(exc)})

    # 汇总统计 + 写入后数量对比（before 已在入口快照，这里只补 after）
    report["summary"]["kept"] = report["summary"]["extracted"] - len(report["filtered"])
    report["summary"]["dropped"] = len(report["filtered"])
    report["summary"]["written"] = len(written)
    for w in report["written"]:
        if w["action"] == "add":
            report["summary"]["added"] += 1
        elif w["action"] == "merge":
            report["summary"]["merged"] += 1
        elif w["action"] == "conflict":
            report["summary"]["conflicted"] += 1
    report["summary"]["after"] = {"session": len(store), "global": len(constant_store) if constant_store is not None else 0}
    report["steps"].append({"name": "落库", "detail": f"写入 {report['summary']['written']} 条（新增 {report['summary']['added']} / 合并 {report['summary']['merged']} / 改口 {report['summary']['conflicted']}）"})
    return written, report


async def maybe_consolidate(store, constant_store, messages, llm, settings, session_id: str) -> list[dict[str, Any]]:
    """轮末静默提取巩固：门控开关后走共享核心，返回实际写入的记忆条目列表。

    scope=global 的条目写入 constant_store（当前客户端的常驻库，跨会话生效）；
    scope=session 的条目写入 store（当前会话库）。任何异常静默吞掉。
    """
    if not getattr(settings, "memory_consolidate_enabled", True):
        return []
    if not messages:
        return []
    written, _ = await _consolidate_core(store, constant_store, messages, llm, settings, session_id)
    return written


async def dream_consolidate(store, constant_store, messages, llm, settings, session_id: str) -> dict[str, Any]:
    """手动触发「记忆梦游 · 对话提炼」：执行一次完整提取→匹配→裁决→落库，返回结构化中间结果报告。

    与轮末静默路径共用 _consolidate_core，仅返回视角不同（报告 vs 写入列表）。
    """
    _, report = await _consolidate_core(store, constant_store, messages, llm, settings, session_id)
    return report


# ---- 记忆梦游 · 整理现有记忆（对齐 Claude Dreaming：读库 → 归并/替换/挖模式 → 直接落库） ----

TIDY_PROMPT = (
    "你是记忆整理器（对齐 Claude Dreaming 的睡眠整理）。以下是当前记忆库的全部条目，"
    "请只做「整理」，不新增对话里没有的信息。逐条处理：\n"
    "- merge：若干条目语义重复/几乎相同 → 合并为一条统一表述（保留信息量最全的）；\n"
    "- replace：某条已过时/被更准确表述取代 → 用新表述替换它；\n"
    "- pattern：多条条目可以归纳出一条更高层的模式/规律（不破坏原条目时也可同时给出）→ 新增该模式；\n"
    "- keep：其余正常条目保留（不要输出 keep）。\n"
    "注意：\n"
    "1. 只能合并同一 scope 内的条目（session 归 session、global 归 global），不得跨库合并；\n"
    "2. 每条建议的 ids 必须来自输入列表，且至少一个；merge 至少两个；\n"
    "3. 保守优先：不确定的保持不动。\n"
    "只输出严格 JSON 数组，不要输出任何其他文字：\n"
    '[{"action": "merge|replace|pattern", "scope": "session|global", "ids": ["id1", "id2"], '
    '"text": "整理后的表述", "kind": "fact|preference|episodic|procedural", '
    '"importance": 0.8, "reason": "一句话理由"}]'
)


def _tidy_payload(items: list[dict[str, Any]]) -> str:
    """把记忆库全部条目组装为整理输入（含 scope 与 id）。"""
    lines = []
    for it in items:
        lines.append(
            f"- id={it['id']} scope={it['scope']} kind={it.get('kind')} "
            f"importance={it.get('importance', 0):.2f}\n  内容：{it['text']}"
        )
    return "\n".join(lines)


def _parse_tidy(content: str) -> list[dict[str, Any]]:
    """解析整理输出 JSON 数组并规范化；非法字段剔除。"""
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for d in data:
        if not isinstance(d, dict):
            continue
        action = str(d.get("action") or "").strip().lower()
        if action not in ("merge", "replace", "pattern"):
            continue
        scope = str(d.get("scope") or "session").strip().lower()
        if scope not in ("session", "global"):
            scope = "session"
        ids = [str(x) for x in (d.get("ids") or []) if str(x).strip()]
        # pattern 为「新归纳出的模式」可无 ids；merge/replace 必须引用现有条目
        if not ids and action != "pattern":
            continue
        text = str(d.get("text") or "").strip()
        if not text:
            continue
        kind = str(d.get("kind") or "fact")
        if kind not in _VALID_KINDS:
            kind = "fact"
        try:
            importance = max(0.0, min(1.0, float(d.get("importance", 0.5))))
        except (TypeError, ValueError):
            importance = 0.5
        out.append(
            {
                "action": action,
                "scope": scope,
                "ids": ids,
                "text": text,
                "kind": kind,
                "importance": importance,
                "reason": str(d.get("reason") or "").strip(),
            }
        )
    return out


async def dream_tidy(store, constant_store, llm, settings, session_id: str) -> dict[str, Any]:
    """手动触发「记忆梦游 · 整理记忆」（对齐 Claude Dreaming）：读取现有记忆库全部条目，
    LLM 给出 merge/replace/pattern 整理建议并**直接落库**，返回整理报告。

    输入库原样快照于报告（before），落库后给出 after，便于对照「整理前后」。
    任何异常静默吞掉（记忆是增强项）。
    """
    # 组装输入：会话库 + 常驻库条目（标注 scope），LLM 只能合并同 scope 内条目
    store_items = [{"id": it["id"], "text": it["text"], "kind": it.get("kind"), "importance": it.get("importance", 0.5), "scope": "session"} for it in store.list()]
    const_items = (
        [{"id": it["id"], "text": it["text"], "kind": it.get("kind"), "importance": it.get("importance", 0.5), "scope": "global"} for it in constant_store.list()]
        if constant_store is not None
        else []
    )
    all_items = store_items + const_items
    report: dict[str, Any] = {
        "summary": {
            "before": {"session": len(store_items), "global": len(const_items)},
            "after": {"session": len(store), "global": len(constant_store) if constant_store is not None else 0},
            "merge": 0,
            "replace": 0,
            "pattern": 0,
        },
        "actions": [],
        "steps": [],
    }
    if not all_items:
        report["steps"].append({"name": "整理", "detail": "记忆库为空，无需整理"})
        return report
    report["steps"].append({"name": "读取", "detail": f"读取现有记忆 {len(all_items)} 条（会话 {len(store_items)} / 常驻 {len(const_items)}）"})
    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=TIDY_PROMPT),
                HumanMessage(content=_tidy_payload(all_items)),
            ]
        )
        content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
        actions = _parse_tidy(content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory tidy 整理失败（已忽略）: %s", exc)
        report["steps"].append({"name": "整理", "detail": f"LLM 整理失败（已忽略）：{exc}"})
        return report
    report["steps"].append({"name": "整理", "detail": f"LLM 给出 {len(actions)} 条整理建议（merge/replace/pattern）"})

    # 按 scope 校验 id 归属并执行：merge 合并进首个 id、其余删除；replace 合并进该 id（旧值归档）；
    # pattern 新增一条。所有建议直接落库（用户已确认「直接整理落库」）。
    id_to_store = {it["id"]: store for it in store_items}
    id_to_scope = {it["id"]: "session" for it in store_items}
    if constant_store is not None:
        id_to_store.update({it["id"]: constant_store for it in const_items})
        id_to_scope.update({it["id"]: "global" for it in const_items})

    executed: list[dict[str, Any]] = []
    for act in actions:
        # pattern 为跨条目归纳的新模式：按 scope 写入目标库（无 id 约束）；
        # merge/replace 必须引用当前库内、且与建议 scope 一致的有效 id，否则忽略。
        if act["action"] == "pattern":
            target = constant_store if (act["scope"] == "global" and constant_store is not None) else store
            entry = {**act, "executed": True}
            try:
                target.add(
                    act["text"],
                    kind=act["kind"],
                    importance=act["importance"],
                    source_session=session_id,
                )
                report["summary"]["pattern"] += 1
            except Exception as exc:  # noqa: BLE001
                entry["executed"] = False
                entry["error"] = str(exc)
                logger.warning("memory tidy 落库失败（已忽略）: %s", exc)
            executed.append(entry)
            continue
        valid_ids = [i for i in act["ids"] if i in id_to_store and id_to_scope[i] == act["scope"]]
        if not valid_ids:
            continue
        target = id_to_store[valid_ids[0]]
        entry = {**act, "executed": True}
        try:
            if act["action"] == "merge" and len(valid_ids) >= 2:
                target.add_judged(
                    act["text"],
                    kind=act["kind"],
                    importance=act["importance"],
                    source_session=session_id,
                    decision="merge",
                    reason=act["reason"] or "记忆整理合并",
                    match_id=valid_ids[0],
                )
                for rid in valid_ids[1:]:
                    target.delete(rid)
                report["summary"]["merge"] += 1
            elif act["action"] == "replace":
                target.add_judged(
                    act["text"],
                    kind=act["kind"],
                    importance=act["importance"],
                    source_session=session_id,
                    decision="merge",
                    reason=act["reason"] or "记忆整理替换（过时）",
                    match_id=valid_ids[0],
                )
                report["summary"]["replace"] += 1
            else:
                entry["executed"] = False
                continue
        except Exception as exc:  # noqa: BLE001
            entry["executed"] = False
            entry["error"] = str(exc)
            logger.warning("memory tidy 落库失败（已忽略）: %s", exc)
        executed.append(entry)
    report["actions"] = executed
    report["summary"]["after"] = {"session": len(store), "global": len(constant_store) if constant_store is not None else 0}
    report["steps"].append(
        {"name": "落库", "detail": f"整理完成：合并 {report['summary']['merge']} / 替换 {report['summary']['replace']} / 新增模式 {report['summary']['pattern']}"}
    )
    return report
