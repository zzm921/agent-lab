"""L2 主动语义召回（企业级「每轮前置把对话转 query 召回相关记忆」）。

区别于 L1 常驻注入（启动按重要度挑、不依赖查询），本模块由系统驱动、
基于当前对话内容主动召回（记忆为系统前置能力，不暴露为模型工具）：

  ① 触发判断（selector 轻量 LLM）——本轮是否需要记忆背景，判否直接跳过（省检索+注入成本）
  ② 合并召回——会话记忆库 + 常驻库各召一次，按 id 去重、按分数合并
  ③ 已见去重——同一会话内已注入过的记忆不重复注入（会话级 injected_ids 集合）
  ④ 预算封顶——top-k 上限 + 注入字符预算（超预算截断）

注入到 user 消息而非 system（只对本轮生效，不污染后续轮次、省 token）；
命中即由 search 内部 touch 更新访问频率（供 LRU/重要度漂移）。
任何异常吞掉，绝不阻断主链路。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SELECTOR_PROMPT = """你是记忆触发判断器。判断「用户最新一条消息」是否可能需要结合过往记忆才能更好回答。

需要召回（need=true）的情形：
- 用户用个人化措辞：我/我的、我的偏好、我记得、之前/上次/以前做过、你了解我
- 身份/归属类查询：我的主管/上级/领导、我的部门/团队、我的薪资/工资/福利、我的生日/档案/工号、我负责什么等——回答这类问题必须知道用户身份，务必召回
- 问题涉及用户资料、历史决策、过往项目、长期偏好或约束
- 与用户个人历史明显相关的开放问题

不需要召回（need=false）的情形：
- 纯闲聊、寒暄
- 通用知识 / 技术常识问题（与个人历史无关）
- 一次性的简单请求，无需背景
- 记忆写入 / 删除指令（记住、记一下、保存、忘掉、删除记忆等）——记录或删除记忆不需要旧背景

只输出 JSON：{"need": true 或 false, "reason": "一句话理由"}"""

# 记忆写指令（确定性规则，先于 selector，省一次轻量 LLM 调用）：
# 「记住/忘掉 X」是写操作，只需记录、不需要背景召回；注入无关旧记忆反而干扰模型。
# 用读指令做反向护栏：只收窄「含写动词但实为读取」的措辞（如「你还记得…记住的偏好吗」、
# 「记住的东西有哪些」），不含写动词的纯读问题天然不会被判为写指令，无需进护栏。
_WRITE_INTENT_RE = re.compile(r"(记住|记一下|记牢|记起来|记下|记好了|帮我记|帮我记住|请记住|请记|保存)")
_FORGET_INTENT_RE = re.compile(r"(忘了|忘掉|忘记|删除记忆|移除记忆|删掉)")
_READ_GUARD_RE = re.compile(r"(你还记得|记得(吗|么|不)|我之前让你记住|我之前让你记|上次让我记|以前让你记|记住的|记住了什么|记住哪些|记住什么)")

# 个人化归属查询（确定性护栏，先于 selector，省一次轻量 LLM 调用）：
# 「我的主管是谁」「我的部门/薪资/生日/工号…」这类身份归属查询必须召回用户身份记忆，
# 否则后续检索会把「当前登录用户」当泛化实体空转（选召 LLM 可能误判为通用查询而跳过）。
# 误召回由阈值 + top-k 预算兜底（宁多召，不漏召）。
_PERSONAL_QUERY_RE = re.compile(r"(我的|本人|我自己的|我自己)")


def is_write_intent(query: str) -> bool:
    """判定是否为记忆写入/删除指令（无需背景召回）；读指令由反向护栏排除。"""
    q = (query or "").strip()
    if not q:
        return False
    if _READ_GUARD_RE.search(q):
        return False
    return bool(_WRITE_INTENT_RE.search(q) or _FORGET_INTENT_RE.search(q))


def _parse_verdict(content: str) -> dict[str, Any]:
    """解析 selector 输出 JSON；解析失败保守返回 need=true（宁多召，交给阈值/预算兜底）。"""
    text = (content or "").strip()
    try:
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text else text
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return {"need": bool(data.get("need")), "reason": str(data.get("reason", ""))}
    except Exception:  # noqa: BLE001 — 解析失败保守召回
        return {"need": True, "reason": "selector 输出解析失败，保守召回"}


def _fmt_date(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "?"


def _fmt_line(h: dict) -> str:
    meta = h.get("metadata") or {}
    return (
        f"- ({meta.get('kind', 'fact')}·重要度{meta.get('importance', 0):.1f}) "
        f"{h['text']} —— 记录于 {_fmt_date(meta.get('created_at'))}"
    )


def _injection_block(hits: list[dict]) -> str:
    return (
        "## 用户记忆（本轮主动召回，仅供参考）\n"
        "以下记忆来自历史对话，可能过时或不准确，仅作背景；若与用户本次说明冲突，以本次说明为准：\n"
        + "\n".join(_fmt_line(h) for h in hits)
    )


async def maybe_recall(
    llm,
    store,
    constant_store,
    query: str,
    *,
    top_k: int = 3,
    threshold: float = 0.3,
    max_chars: int = 400,
    injected_ids: set[str],
    emit=None,
    selector_enabled: bool = True,
) -> tuple[str | None, list[dict], bool]:
    """执行一次主动语义召回，返回 (注入块 or None, 命中的 hits, 是否被 selector 拦下)。

    - selector_enabled 且判断 need=false → 直接跳过（不发注入，事件带 need=false）
    - 合并会话库 + 常驻库召回，按 id 去重、按分数降序
    - 过滤本会话已注入记忆，按 top-k + 字符预算封顶
    - 有注入内容时把命中 id 记入 injected_ids（会话级已见集合）
    - 任何异常吞掉返回 (None, [], False)
    """
    # ① 触发判断
    # 0) 写指令确定性跳过：记住/忘掉类指令只需记录、不需要背景召回（先于 selector，省一次 LLM 调用）
    if is_write_intent(query):
        reason = "记忆写指令（记住/忘掉）无需背景召回"
        if emit is not None:
            emit({"type": "memory_read", "query": query, "hits": [], "source": "proactive", "need": False, "reason": reason})
        return None, [], True
    need, reason = True, ""
    if _PERSONAL_QUERY_RE.search(query):
        # 个人化归属查询确定性召回：跳过 selector，保证用户身份记忆进入链路
        # （身份丢失会让后续检索把「当前登录用户」当泛化实体空转）
        reason = "个人化归属查询，需召回用户身份记忆"
    elif selector_enabled:
        try:
            resp = await llm.ainvoke([SystemMessage(content=SELECTOR_PROMPT), HumanMessage(content=query)])
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            verdict = _parse_verdict(content)
            need = verdict.get("need", True)
            reason = verdict.get("reason", "")
        except Exception as exc:  # noqa: BLE001 — selector 失败保守召回
            logger.warning("memory selector 调用失败（保守召回）: %s", exc)
    if not need:
        if emit is not None:
            emit({"type": "memory_read", "query": query, "hits": [], "source": "proactive", "need": False, "reason": reason})
        return None, [], True

    # ② 合并召回：会话库 + 常驻库
    hits: list[dict] = []
    seen_ids: set[str] = set()
    for s in (store, constant_store):
        if s is None:
            continue
        try:
            for h in s.search(query, top_k=top_k, threshold=threshold):
                rid = (h.get("metadata") or {}).get("id")
                if rid and rid in seen_ids:
                    continue
                if rid:
                    seen_ids.add(rid)
                hits.append(h)
        except Exception as exc:  # noqa: BLE001 — 单库失败不影响另一库
            logger.warning("memory proactive 召回失败（已忽略）: %s", exc)
    hits.sort(key=lambda h: h.get("score", 0), reverse=True)

    # ③ 已见去重（本会话已注入过的记忆不再注入）
    fresh = [h for h in hits if (h.get("metadata") or {}).get("id") not in injected_ids]

    # ④ 预算封顶：top-k 限制 + 字符上限
    kept: list[dict] = []
    used_chars = 0
    for h in fresh[:top_k]:
        line = _fmt_line(h)
        if used_chars + len(line) > max_chars:
            break
        kept.append(h)
        used_chars += len(line)

    block = None
    if kept:
        for h in kept:
            rid = (h.get("metadata") or {}).get("id")
            if rid:
                injected_ids.add(rid)
        block = _injection_block(kept)
    if emit is not None:
        emit(
            {
                "type": "memory_read",
                "query": query,
                "hits": [{"text": h["text"], "score": h["score"]} for h in kept],
                "source": "proactive",
                "need": True,
                "reason": reason,
            }
        )
    return block, kept, False
