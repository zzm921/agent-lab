"""上下文管理与压缩管线：snip-compact（对话修剪）/ micro-compact（工具结果占位）/ auto-compact（LLM 摘要）/ 大文件落盘。

设计原则（详见 .trae/documents/上下文管理与压缩落地实施方案.md）：
- 「便宜的先跑，贵的后跑」：前三层全部本地、确定性、零 API 成本；auto-compact 走 LLM，默认配置关闭。
- 只作用于发给模型的副本，不写回 checkpointer：原始历史保留可回滚，不动 Prompt Cache 前缀。
- 边界保护：裁剪绝不拆散 AI(tool_use) 与其对应 ToolMessage 配对，保证工具调用逻辑完整。
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

# 旧工具结果截断后缀（micro-compact：保留头部关键信息，而非清空成零信息占位符）
_MICRO_SUFFIX = "\n…（工具结果已压缩，保留头部关键信息）"


def _backend_root() -> Path:
    """backend 根目录（本文件位于 backend/app/agents/ 下）。"""
    return Path(__file__).resolve().parents[2]


def maybe_offload(text: str, *, session_id: str, tool_name: str, settings) -> tuple[str, dict | None]:
    """超大单条工具输出落盘：文本超过阈值时写入磁盘，上下文只保留指针文本。

    返回 (指针文本, 落盘信息 dict | None)。未触发时原样返回 (text, None)。
    """
    threshold = int(getattr(settings, "context_offload_threshold", 3000))
    if len(text) <= threshold:
        return text, None

    preview = int(getattr(settings, "context_offload_preview", 200))
    max_per_session = int(getattr(settings, "context_offload_max_per_session", 50))
    rel_dir = getattr(settings, "context_offload_dir", "./data/offload")
    out_dir = Path(rel_dir)
    if not out_dir.is_absolute():
        out_dir = _backend_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
    stamp = int(time.time() * 1000)
    prefix = (session_id or "anon")[:8]
    filename = f"{prefix}_{stamp}_{digest}.txt"
    out_path = out_dir / filename
    out_path.write_text(text, encoding="utf-8")

    # 治理：按会话前缀统计文件数，超过上限删最旧（LRU）
    session_files = sorted(out_dir.glob(f"{prefix}_*.txt"), key=lambda p: p.stat().st_mtime)
    while len(session_files) > max_per_session:
        session_files.pop(0).unlink(missing_ok=True)
        session_files = sorted(out_dir.glob(f"{prefix}_*.txt"), key=lambda p: p.stat().st_mtime)

    rel = str(Path(rel_dir) / filename)
    ptr = (
        f"[工具输出已落盘] 完整输出共 {len(text)} 字符，已保存至 {rel}。"
        f"开头预览：{text[:preview]}… 需要细节时可用 run_command 读取该文件。"
    )
    return ptr, {"chars": len(text), "file": rel}


def _no_tool_calls(msg) -> bool:
    """消息不带 tool_calls（可安全用于去重/合并的普通文本消息）。"""
    return not getattr(msg, "tool_calls", None)


def _same_key(a, b) -> bool:
    """两条消息是否"完全重复"：ToolMessage 按 tool_call_id 判同（同一调用结果唯一），
    其余按 type + content 判同。带 tool_calls 的 AI 由调用方另行保护。"""
    if isinstance(a, ToolMessage) and isinstance(b, ToolMessage):
        return str(getattr(a, "tool_call_id", "")) == str(getattr(b, "tool_call_id", ""))
    return a.type == b.type and str(getattr(a, "content", "")) == str(getattr(b, "content", ""))


def _drop_exact_duplicates(messages: list) -> list:
    """仅清理"相邻"的完全重复消息（流式重试/断点重放产物），不删跨轮次相同内容，
    保持 user/assistant 交替结构与工具调用配对完整。"""
    result: list = []
    for m in messages:
        if not result:
            result.append(m)
            continue
        if _same_key(result[-1], m):
            # 相邻重复：带 tool_calls 的 AI 保守保留（不拆配对），其余删除
            if isinstance(m, AIMessage) and not _no_tool_calls(m):
                result.append(m)
        else:
            result.append(m)
    return result


def _merge_consecutive_assistant(messages: list) -> list:
    """合并连续、无 tool_calls、内容相同的助手消息为一条。"""
    result: list = []
    for m in messages:
        if (
            result
            and isinstance(result[-1], AIMessage)
            and isinstance(m, AIMessage)
            and _no_tool_calls(result[-1])
            and _no_tool_calls(m)
            and result[-1].content == m.content
        ):
            continue
        result.append(m)
    return result


def snip_compact(
    messages: list,
    *,
    max_messages: int,
    keep_head: int,
    keep_tail: int,
) -> tuple[list, dict | None]:
    """snip-compact：对话历史过长时「掐头去尾」裁掉中间旧消息。

    只做条数治理（去重 / 合并连续同内容助手消息）→ 掐头去尾 → 边界保护
    （tail 起始是 ToolMessage 时向前并入其 tool_use AIMessage）→ 头尾衔接保护。
    工具结果「长度」压缩职责已移交 micro_compact，避免同一内容被重复压缩。
    全程产生新消息副本，不修改入参。
    """
    if len(messages) <= max_messages:
        return messages, None

    # 1) 轻量清理（只去重/合并，不截断工具输出）
    msgs = _drop_exact_duplicates(messages)
    msgs = _merge_consecutive_assistant(msgs)

    # 2) 掐头去尾
    start = max(keep_head, len(msgs) - keep_tail)

    # 3) 边界保护：tail 起始是 ToolMessage 时向前并入其 tool_use AIMessage
    while start < len(msgs) and msgs[start].type == "tool":
        if start <= keep_head:
            break
        start -= 1

    # 4) 头尾衔接：head[-1] 若是带 tool_calls 的 AI 消息（其 ToolMessage 被裁掉）→ 从 head 移除
    head = msgs[:keep_head]
    if head and isinstance(head[-1], AIMessage) and not _no_tool_calls(head[-1]):
        head = head[:-1]

    result = head + msgs[start:]
    metrics = {"original": len(messages), "kept": len(result), "dropped": len(messages) - len(result)}
    return result, metrics


def micro_compact(messages: list, *, keep_recent: int, truncate_chars: int = 300) -> tuple[list, dict | None]:
    """micro-compact：压缩「较旧工具结果」的体积——保留最近 keep_recent 条原文，
    更早的超长结果截断到头部 truncate_chars 字符（保留关键信息，而非清空成零信息占位符）。
    落盘指针与短文本不动（指针含路径，受保护）。
    只改 content，消息与 tool_call_id 配对关系保持，不破坏工具调用逻辑。
    """
    if truncate_chars <= 0:
        return messages, None
    tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if len(tool_idx) <= keep_recent:
        return messages, None
    result = list(messages)
    truncated = 0
    for i in tool_idx[:-keep_recent]:
        m = result[i]
        content = getattr(m, "content", None)
        if not isinstance(content, str) or len(content) <= truncate_chars:
            continue  # 短文本：无需压缩
        if content.startswith("[工具输出已落盘]"):
            continue  # 落盘指针受保护：只留指针，不截断（保留路径可回读）
        result[i] = m.model_copy(update={"content": content[:truncate_chars] + _MICRO_SUFFIX})
        truncated += 1
    if truncated == 0:
        return messages, None
    metrics = {"original": len(tool_idx), "truncated": truncated, "kept": keep_recent}
    return result, metrics


def _serialize(messages: list) -> str:
    lines = []
    for m in messages:
        if isinstance(m, ToolMessage):
            lines.append(f"[工具结果]{m.content}")
        elif isinstance(m, AIMessage) and not _no_tool_calls(m):
            lines.append(f"[助手调用工具]{m.tool_calls}")
        else:
            lines.append(f"{m.type}: {m.content}")
    return "\n".join(lines)


class ContextManager:
    """四层上下文压缩管线：snip → micro → auto（成本递增）。"""

    def __init__(self, settings):
        self.settings = settings
        # auto-compact 摘要记忆（按会话）：{session_id: {"count": int, "last_id": str, "summary": str}}
        self._compact_state: dict[str, dict] = {}
        # auto-compact 熔断：连续失败达到上限后该会话禁用
        self._fails: dict[str, int] = {}
        self._disabled: set[str] = set()

    # ---- 总入口 ----
    async def build(self, messages: list, *, llm=None, session_id: str | None = None, keep_rounds: int = 0) -> tuple[list, list]:
        """按 snip → micro → auto 顺序处理消息副本，返回 (新消息, context 事件负载列表)。

        keep_rounds > 0 时进入「每轮压缩」演示模式：保留最近 keep_rounds 轮原文
        （每轮按 2 条消息估算），更早的历史每轮都裁剪/截断，便于页面上持续观察压缩卡片；
        否则使用 Settings 中的默认阈值（达到阈值才触发）。
        """
        events: list[dict] = []
        msgs = list(messages)

        if getattr(self.settings, "context_snip_enabled", True):
            if keep_rounds > 0:
                keep_tail = max(2, keep_rounds * 2)
                keep_head = int(getattr(self.settings, "context_snip_keep_head", 3))
                max_messages = keep_head + keep_tail  # 历史超过「保留量」即触发，之后每轮都压缩
            else:
                max_messages = int(getattr(self.settings, "context_snip_max_messages", 50))
                keep_head = int(getattr(self.settings, "context_snip_keep_head", 3))
                keep_tail = int(getattr(self.settings, "context_snip_keep_tail", 47))
            msgs, metrics = snip_compact(
                msgs,
                max_messages=max_messages,
                keep_head=keep_head,
                keep_tail=keep_tail,
            )
            if metrics is not None:
                ev = {"kind": "snip_compact", "metrics": metrics, "threshold": max_messages}
                if keep_rounds > 0:
                    ev["keep_rounds"] = keep_rounds
                events.append(ev)

        if getattr(self.settings, "context_micro_enabled", True):
            if keep_rounds > 0:
                keep_recent = max(1, keep_rounds * 2)  # 保留最近 keep_rounds 轮工具结果原文
            else:
                keep_recent = int(getattr(self.settings, "context_micro_keep_recent", 6))
            msgs, metrics = micro_compact(
                msgs,
                keep_recent=keep_recent,
                truncate_chars=int(getattr(self.settings, "context_micro_truncate_chars", 300)),
            )
            if metrics is not None:
                events.append({"kind": "micro_compact", "metrics": metrics})

        if getattr(self.settings, "context_auto_compact_enabled", False) and llm is not None:
            msgs, metrics = await self.auto_compact(
                msgs,
                llm=llm,
                keep_recent=int(getattr(self.settings, "context_auto_compact_keep_recent", 20)),
                session_id=session_id,
            )
            if metrics is not None:
                events.append({"kind": "auto_compact", "metrics": metrics, "summary": metrics.get("summary", "")})

        return msgs, events

    # ---- auto-compact（LLM 摘要，默认配置关闭）----
    async def auto_compact(self, messages: list, *, llm, keep_recent: int, session_id: str | None) -> tuple[list, dict | None]:
        sid = session_id or "anon"
        threshold = int(getattr(self.settings, "context_auto_compact_threshold", 100))
        if sid in self._disabled or len(messages) <= threshold:
            return messages, None

        older, recent = messages[:-keep_recent], messages[-keep_recent:]
        cached = self._compact_state.get(sid)
        # 摘要记忆复用：上次已压缩到相同长度且尾部消息 id 一致（checkpointer 追加式，前缀不变）
        if (
            cached is not None
            and cached["count"] == len(older)
            and cached["last_id"] == getattr(older[-1], "id", None)
        ):
            summary = cached["summary"]
        else:
            summary = await self._llm_summarize(llm, older)
            if summary is None:
                self._fails[sid] = self._fails.get(sid, 0) + 1
                if self._fails[sid] >= 3:
                    self._disabled.add(sid)
                return messages, None
            self._fails[sid] = 0
            self._compact_state[sid] = {
                "count": len(older),
                "last_id": getattr(older[-1], "id", None),
                "summary": summary,
            }

        result = [SystemMessage(content=f"【历史摘要】\n{summary}")] + list(recent)
        metrics = {"original": len(messages), "kept": len(result), "summary": summary}
        return result, metrics

    async def _llm_summarize(self, llm, older: list) -> str | None:
        """调用 LLM 对旧消息做结构化摘要；失败返回 None。"""
        system = (
            "你是对话摘要器。把用户提供的多轮对话压缩为结构化摘要，只保留对后续任务仍有用的信息。"
            "严格按以下 JSON 输出，不添加原文没有的信息，每条不超过 20 字：\n"
            "{\"goal\": \"用户当前真实目标，一句话\", \"decided\": [\"已确认的事实/决策，逐条\"], "
            "\"failed\": [\"已尝试方案与失败原因，逐条\"], \"todo\": [\"未完成事项与下一步行动，逐条\"]}"
        )
        try:
            resp = await llm.ainvoke(
                [
                    SystemMessage(content=system),
                    SystemMessage(content=f"待摘要的历史对话：\n{_serialize(older)}"),
                ]
            )
            return str(getattr(resp, "content", "") or "").strip() or None
        except Exception:
            return None
