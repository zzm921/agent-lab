"""向用户提问工具：Agent 缺少用户私有信息时调用，触发 HITL 澄清卡片。

该工具的「执行」由 tools_node（手写 StateGraph）与 events_mw（react/multi-agent）
特殊拦截：调用时触发 LangGraph interrupt 暂停等待用户回复，恢复后把用户输入
（或跳过标记）格式化为问答文本返回给模型。此处仅提供工具声明与共享的
归一化/格式化助手，正常路径不会被调用到。

设计要点：一次提出全部缺失的关键信息（questions 列表），避免一问一答多次往返；
澄清交互不计入 Agent 轮数上限（由各模式在计数处豁免）。
"""
from __future__ import annotations

import json
import re

from langchain_core.tools import tool


@tool
def ask_user(questions: list[dict]) -> str:
    """当任务缺少只有用户能提供的信息（如出发地、预算、偏好、时间、同行人员等）时，一次性提出全部澄清问题并等待回复。

    - questions：需要用户补充的关键信息列表（一次列全所有缺失项，不要一问一答多次往返），每项为：
      {"question": "问题文本", "options": ["候选答案1", "候选答案2", ...]}
      options 为可选的候选答案（2-4 项，每项是完整答案文本），用户可直接点选，降低回答成本。
    该工具会暂停执行等待用户回复；用户可能选择「跳过/无法回答」，此时返回「用户跳过未回答」。
    """
    return "（等待用户回复）"


def normalize_options(options) -> list[str]:
    """把模型返回的选项归一化为 list[str]。

    模型经常把数组参数以字符串形式返回（甚至双重编码的 JSON 数组）。
    若直接透传，前端 v-for 会按字符拆分成单个按钮；这里统一清洗后再进中断。
    """
    if options is None:
        return []
    if isinstance(options, str):
        s = options.strip()
        if not s:
            return []
        if s.startswith("["):  # 双重编码的 JSON 数组：'["a", "b"]'
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except ValueError:
                pass
        parts = [p.strip() for p in re.split(r"[\n,，;；|]", s) if p.strip()]
        return parts if parts else [s]
    if isinstance(options, (list, tuple)):
        return [str(x).strip() for x in options if str(x).strip()]
    return [str(options)]


def _coerce_questions_list(raw):
    """把可能是 JSON 字符串形态的 questions 参数解析为列表；无法解析返回 None。

    模型经常把数组参数以字符串返回，甚至双重编码 JSON 数组、末尾带杂质
    （如 ")\n"）。先剥离尾部多余字符再尝试 json.loads。
    """
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            candidates = [s]
            if s.endswith(")"):
                candidates.append(s[:-1].rstrip())
            for cand in candidates:
                try:
                    parsed = json.loads(cand)
                    if isinstance(parsed, list):
                        return parsed
                except ValueError:
                    continue
    return None


def normalize_questions(args) -> list[dict]:
    """把 ask_user 的工具参数归一化为 questions 列表。

    新格式：{"questions": [{"question": "...", "options": [...]}, ...]}
    兼容旧格式（单问）：{"question": "...", "options": [...]} → 包装为单元素列表。
    模型偶发把 questions 整体以 JSON 字符串返回（双重编码、带杂质），先解析为列表；
    args 本身也可能是被序列化成的字符串，同样先解析。
    逐项清洗：question 转字符串、options 经 normalize_options 归一化，跳过空问题。
    """
    if isinstance(args, str):  # args 偶发是整个 arguments 被序列化成的字符串
        try:
            parsed = json.loads(args)
            args = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            args = {}
    raw = args.get("questions")
    if raw is None and args.get("question"):
        raw = [{"question": args.get("question"), "options": args.get("options")}]
    elif raw is not None:
        parsed = _coerce_questions_list(raw)
        raw = parsed if parsed is not None else []
    questions = []
    for item in raw:
        if isinstance(item, str):  # 模型偶发把每项直接写成字符串
            q = item.strip()
            options: list = []
        else:
            item = item if isinstance(item, dict) else {}
            q = str(item.get("question") or "").strip()
            options = normalize_options(item.get("options"))
        if not q:
            continue
        questions.append({"question": q, "options": options})
    return questions


def format_ask_reply(questions: list[dict], answers: list | None) -> str:
    """把用户回复按问题对齐格式化为问答文本，作为 ToolMessage 返回给模型。

    answers 与 questions 按下标对齐；每项可为 dict（{"answer": ..., "skip": bool}）
    或纯文本字符串；缺省/空/标记 skip 视为「用户跳过未回答」。
    """
    lines = []
    for i, q in enumerate(questions):
        text = (q or {}).get("question", "")
        ans = answers[i] if isinstance(answers, list) and i < len(answers) else None
        if isinstance(ans, dict):
            a = str(ans.get("answer") or "").strip()
            skipped = bool(ans.get("skip")) or not a
        elif ans is None:
            a, skipped = "", True
        else:
            a, skipped = str(ans).strip(), False
        lines.append(f"Q{i + 1} {text}：{'用户跳过未回答' if skipped else a}")
    return "\n".join(lines) if lines else "用户未填写回复"


def is_ask_reply_msg(msg) -> bool:
    """判断消息是否为 ask_user 的回复 ToolMessage（tools_node/events_mw 在返回时打了 ask_reply 标记）。

    各模式节点据此豁免澄清轮次：ask 回复后接续的模型调用不计入轮数上限。
    """
    return getattr(msg, "type", None) == "tool" and bool(
        (getattr(msg, "additional_kwargs", None) or {}).get("ask_reply")
    )
