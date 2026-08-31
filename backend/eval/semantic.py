"""语义评测公共件：上下文拼装、生成 prompt、库外拒答 judge（供全面评测复用）。

生成质量评分统一由 RAGAS 标准指标承担（见 eval/full.py），本模块不再手写
faithfulness/answer_relevance 评分；仅保留 RAGAS 覆盖不了的部分：
- 生成链路公共件：检索命中 → 与线上一致的上下文注入与生成 prompt；
- 库外拒答判定：LLM-as-a-Judge 判定库外用例是否「拒绝回答而非编造」（grounded）。
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

# judge 输出 JSON 片段（含可能被模型包裹的散文）
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _format_context(hits: list[dict[str, Any]]) -> str:
    """把检索命中拼成编号资料（judge/生成可见），与线上上下文注入口径一致。"""
    lines = []
    for i, h in enumerate(hits, 1):
        cid = (h.get("metadata") or {}).get("chunk_id", "-")
        lines.append(f"[{i}]({cid}) {h.get('text', '')}")
    return "\n".join(lines)


def _gen_prompt(query: str, context: str) -> list:
    system = (
        "你是企业员工制度助手。仅依据下方「资料」回答用户问题，不得编造资料中没有的信息；"
        "若资料不足以回答，直接说明「资料中未找到相关信息」，不要猜测。"
    )
    user = f"资料：\n{context or '（无）'}\n\n问题：{query}"
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _judge_prompt(query: str, answer: str, context: str) -> list:
    """库外用例 judge prompt：正确做法是明确拒绝/说明资料不足，强行作答即编造。"""
    system = (
        "你是 RAG 语义评测判官。本问题是知识库外的问题，资料中没有对应条目——"
        "正确做法是明确拒绝/说明资料不足；若模型强行给出资料之外的答案即视为编造。\n"
        "只输出一条 JSON，不要输出其他文字。\n"
        "- grounded（true/false）：回答是否做到「说明资料不足/拒绝回答」而非编造。\n"
        "输出格式："
        '{"grounded": true/false, "reason": "一句话"}'
    )
    user = (
        f"资料：\n{context or '（无）'}\n\n"
        f"问题：{query}\n\n助手回答：{answer or '（空）'}"
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _extract_json(content: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象；失败抛 ValueError（由调用方按解析失败处理）。"""
    match = _JSON_RE.search(content)
    if not match:
        raise ValueError("judge 输出中未找到 JSON")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("judge 输出 JSON 必须是对象")
    return data


def judge_out_of_kb(query: str, answer: str, context: str, judge) -> dict[str, Any]:
    """库外用例拒答判定：返回 {"grounded": bool, "reason": str}；解析失败抛 ValueError。"""
    resp = judge.invoke(_judge_prompt(query, answer, context))
    content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
    data = _extract_json(content)
    return {
        "grounded": bool(data.get("grounded", True)),
        "reason": str(data.get("reason", "") or ""),
    }
