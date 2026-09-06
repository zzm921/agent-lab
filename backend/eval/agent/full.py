"""Agent 全量评测 · 答案质量层（企业级 LLM-as-a-Judge，不复用 RAGAS）。

在 runner（L0 任务层约束 + L1 架构层不变量，全部离线确定性）之上叠加 L2 答案语义评分，
形成「行为层离线断言 + 语义层 LLM 评分」的企业级双层评测：

- 答案来源：事件流最终 message（最后一次工具执行之后的所有 message delta 拼接），
  由 FakeChatModel 脚本化决策驱动真实 Harness 产出——隔离生成随机性，聚焦评分链路本身；
- 事实证据：事件流 tool_end 的 tool(args) → result，构造「工具执行证据」列表；
- 评分方式：自定义 agent_judge 场景逐指标判卷（企业级 LLM-as-Judge，不依赖 RAGAS）：
    answer_correctness（对照金标 reference + 要点覆盖：事实一致性 0.5 + 要点覆盖 0.5）
    answer_relevancy（回答是否直接、完整回应问题）
    faithfulness（主张级验证：答案拆原子断言，逐条对照工具证据判定支持度，防幻觉核心；
                  仅对产出工具证据的用例评分，无证据用例只评前两者）
  + 确定性旁证：answer_keywords 关键词覆盖（子串匹配，无 LLM，正确性的低层符号信号）；
- 两种模式：fake 离线冒烟（占位文本解析必败，分数 None，仅验证链路）/
  真实 LLM（仅需 LLM_API_KEY，agent_judge 场景评分，不依赖 Embedding）；
- 审批 / 轮数上限用例（无最终答案）不进入 L2 评分，标记 answer_scored=False。

报告：合并 L0/L1/L2 三层，逐用例带行为断言 + 答案质量分 + 工具证据 + 关键词覆盖，按模式汇总均分。
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.errors import ConfigError
from app.core.version import get_app_version
from app.llm.client import get_chat_model
from app.llm.fake_model import FakeChatModel

from eval.agent import runner

# judge 输出 JSON 片段（含可能被模型包裹的散文）
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# L2 答案质量指标（逐用例字段与报告键共用）
_ANSWER_METRICS = ("answer_correctness", "answer_relevancy", "faithfulness")


def _to_score(value: Any) -> float | None:
    """judge 失败行（None / NaN / 非数值）→ 归一为 None，便于汇总与报告。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 3)


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in records if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _extract_json(content: str) -> dict[str, Any]:
    """从 judge 输出提取 JSON 对象；失败抛 ValueError（由调用方按解析失败处理）。"""
    match = _JSON_RE.search(content)
    if not match:
        raise ValueError("judge 输出中未找到 JSON")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("judge 输出 JSON 必须是对象")
    return data


def _judge_invoke(judge, messages: list) -> dict[str, Any]:
    """调用 judge 并提取结构化 JSON 输出；解析失败抛 ValueError。"""
    resp = judge.invoke(messages)
    content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
    return _extract_json(content)


# ---------------------------------------------------------------------------
# 答案 / 证据提取
# ---------------------------------------------------------------------------
def _extract_answer(events: list[dict[str, Any]]) -> str:
    """提取最终答案：最后一次工具执行（tool_start/tool_end 或 worker dispatch）之后产生的最终文本。

    react / plan_execute / multi_agent：工具轮之后的 message 即最终回答（中间工具轮 message 为空）；
    reflection：无工具事件 → 首稿为 message 增量；修订场景末次修订稿走 revise 增量，
    以修订稿为准（revise 优先于 message），否则拼接全部 message（首稿正文）。
    """
    last_act = -1
    for i, ev in enumerate(events):
        t = ev.get("type")
        if t in ("tool_start", "tool_end") or (t == "agent_event" and ev.get("status") == "dispatch"):
            last_act = i
    draft = ""
    revise = ""
    for ev in events[last_act + 1:]:
        t = ev.get("type")
        if t == "message":
            draft += str(ev.get("delta") or "")
        elif t == "revise":
            revise += str(ev.get("delta") or "")
    return (revise or draft).strip()


def _extract_evidence(events: list[dict[str, Any]]) -> list[str]:
    """从事件流提取工具执行证据：tool(args) → result（仅成功执行，多轮工具调用逐条保留）。"""
    evs: list[str] = []
    for ev in events:
        if ev.get("type") != "tool_end" or not ev.get("success"):
            continue
        args = ev.get("args") or {}
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
        evs.append(f"{ev.get('tool')}({arg_str}) → {ev.get('result')}")
    return evs


# ---------------------------------------------------------------------------
# 企业级 judge（自定义 agent_judge 场景，逐指标判卷，不复用 RAGAS）
# ---------------------------------------------------------------------------
def _judge_correctness(query: str, answer: str, reference: str, judge) -> dict[str, Any]:
    """正确性：对照金标 reference——事实一致性 0.5 + 要点覆盖 0.5。"""
    system = (
        "你是 Agent 答案质量判官，负责「正确性」评分。把「助手回答」与「金标参考答案」对照：\n"
        "1. 事实一致性：回答中的关键事实是否与参考答案一致（语义等价即可，不要求逐字）；\n"
        "2. 要点覆盖：参考答案包含哪些关键要点，回答覆盖了多少。\n"
        "score = 事实一致性×0.5 + 要点覆盖×0.5。\n"
        "只输出一条 JSON，不要输出其他文字：\n"
        '{"score": 0.0~1.0, "key_points_hit": ["..."], "missing_points": ["..."], "reason": "一句话"}'
    )
    user = (
        f"用户问题：{query}\n\n"
        f"金标参考答案：{reference or '（无）'}\n\n"
        f"助手回答：{answer or '（空）'}"
    )
    return _judge_invoke(judge, [SystemMessage(content=system), HumanMessage(content=user)])


def _judge_relevancy(query: str, answer: str, judge) -> dict[str, Any]:
    """相关性：回答是否直接、完整回应问题（不答非所问 / 不回避核心诉求）。"""
    system = (
        "你是 Agent 答案质量判官，负责「相关性」评分。判断「助手回答」是否直接、完整地回应了「用户问题」：\n"
        "1. 是否切题（不答非所问）；\n"
        "2. 是否覆盖问题的核心诉求（信息够用、有无回避）。\n"
        "只输出一条 JSON，不要输出其他文字：\n"
        '{"score": 0.0~1.0, "reason": "一句话"}'
    )
    user = f"用户问题：{query}\n\n助手回答：{answer or '（空）'}"
    return _judge_invoke(judge, [SystemMessage(content=system), HumanMessage(content=user)])


def _judge_faithfulness(query: str, answer: str, evidence: list[str], judge) -> dict[str, Any]:
    """忠实度：主张级验证——答案拆原子断言，逐条对照工具执行证据判定支持度（防幻觉核心）。"""
    system = (
        "你是 Agent 答案质量判官，负责「忠实度」评分（主张级验证）。\n"
        "步骤 1：把「助手回答」拆解为若干原子事实断言（每个断言是一个可被独立验证的事实陈述）；\n"
        "步骤 2：逐个断言对照「工具执行证据」判定是否被支持：\n"
        "  - 证据中存在该事实 / 可由证据直接推断 → supported\n"
        "  - 证据中找不到依据 / 与证据矛盾 → unsupported（视为编造）\n"
        "注意：支持依据只能是工具执行证据，不得用先验常识替证据补位。\n"
        "score = supported 断言数 ÷ 断言总数。\n"
        "只输出一条 JSON，不要输出其他文字：\n"
        '{"claims": [{"claim": "...", "supported": true/false}, ...], "score": 0.0~1.0, "reason": "一句话"}'
    )
    user = (
        f"用户问题：{query}\n\n"
        f"工具执行证据：\n{chr(10).join(evidence) if evidence else '（无）'}\n\n"
        f"助手回答：{answer or '（空）'}"
    )
    return _judge_invoke(judge, [SystemMessage(content=system), HumanMessage(content=user)])


def _keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    """确定性旁证：金标要点关键词在答案中的覆盖比例（子串匹配，无 LLM）。"""
    kws = [k for k in keywords if k]
    if not kws:
        return None
    hits = sum(1 for k in kws if k in (answer or ""))
    return round(hits / len(kws), 3)


def _build_judge(fake: bool):
    """按模式构建评分 judge：fake 占位（解析必败，冒烟链路）或真实 agent_judge 场景（关闭思考）。"""
    if fake:
        return FakeChatModel(script=[])
    judge = get_chat_model("agent_judge")
    if judge is None:
        raise ConfigError(
            "Agent 答案质量评测需要真实 LLM（agent_judge 场景评分）：请配置 LLM_API_KEY。"
            "离线冒烟可用 --fake（仅验证链路，无评测意义）。"
        )
    return judge


def _judge_records(records: list[dict[str, Any]], fake: bool) -> None:
    """L2 逐用例判卷：有工具证据的用例评三维（含 faithfulness）；无证据用例只评正确性/相关性。

    单指标解析失败（如 fake 占位文本）记为 None，不中断全量。
    """
    judge = _build_judge(fake)
    for rec in records:
        if not rec["answer_scored"]:
            continue
        ans = rec["answer"]
        try:
            data = _judge_correctness(rec["query"], ans, rec["reference"], judge)
            rec["answer_correctness"] = _to_score(data.get("score"))
            rec["correct_hit"] = data.get("key_points_hit")
            rec["correct_miss"] = data.get("missing_points")
        except Exception:  # noqa: BLE001 — 单指标失败不中断全量
            rec["answer_correctness"] = None
        try:
            data = _judge_relevancy(rec["query"], ans, judge)
            rec["answer_relevancy"] = _to_score(data.get("score"))
        except Exception:  # noqa: BLE001
            rec["answer_relevancy"] = None
        if rec["evidence"]:
            try:
                data = _judge_faithfulness(rec["query"], ans, rec["evidence"], judge)
                rec["faithfulness"] = _to_score(data.get("score"))
                rec["claims"] = data.get("claims")
            except Exception:  # noqa: BLE001
                rec["faithfulness"] = None


# ---------------------------------------------------------------------------
# 主流程 / 报告
# ---------------------------------------------------------------------------
def run(fake: bool = False, runs: int = 1, real: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑全量 Agent 评测（L0/L1 离线断言 + L2 LLM-judge 答案评分）。

    fake=True：评分用 Fake 模型占位，离线冒烟链路（分数无评测意义）；
    fake=False：真实 LLM（agent_judge 场景评分），缺 Key 抛 ConfigError；
    runs>1：每用例执行多次并合并成功率（answers/证据基于第一次运行事件流）；
    real=True：行为层用真实 LLM（chat 场景）驱动（需 Key），success_rate 有鲁棒性意义。
    """
    runs = max(1, runs)
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in runner._load_cases():
        subs: list[dict[str, Any]] = []
        first_events: list[dict[str, Any]] | None = None
        for i in range(runs):
            events, elapsed_ms, stats = asyncio.run(runner._execute_case(case, real=real))
            rec = runner._evaluate(case, events, elapsed_ms)
            rec["model_calls"] = stats["calls"]
            rec["prompt_tokens"] = stats["prompt_tokens"]
            rec["completion_tokens"] = stats["completion_tokens"]
            if i == 0:
                first_events = events
            subs.append(rec)
            latencies.append(elapsed_ms)
        rec = runner._merge_runs(subs, runs)
        events = first_events or []
        rec["answer"] = _extract_answer(events)
        rec["reference"] = case.get("reference")
        rec["answer_keywords"] = case.get("answer_keywords") or []
        rec["keyword_hit"] = _keyword_coverage(rec["answer"], rec["answer_keywords"])
        rec["evidence"] = _extract_evidence(events)
        rec["answer_scored"] = bool(rec["answer"]) and bool(rec["reference"])
        for m in _ANSWER_METRICS:
            rec[m] = None
        records.append(rec)

    _judge_records(records, fake)

    l1 = runner._aggregate(records, latencies)
    scored = [r for r in records if r["answer_scored"]]
    faithful = [r for r in records if r["answer_scored"] and r["evidence"]]
    modes: dict[str, dict[str, Any]] = {}
    for mode in sorted({r["mode"] for r in records}):
        mode_scored = [r for r in records if r["mode"] == mode and r["answer_scored"]]
        mode_all = [r for r in records if r["mode"] == mode]
        modes[mode] = {
            **l1["modes"].get(mode, {}),
            "avg_success_rate": runner._mean(mode_all, "success_rate"),
            "avg_answer_correctness": _mean(mode_scored, "answer_correctness"),
            "avg_answer_relevancy": _mean(mode_scored, "answer_relevancy"),
            "avg_faithfulness": _mean(mode_scored, "faithfulness"),
        }

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "version": get_app_version(),
            "mode": (
                "fake 冒烟（无评测意义）"
                if fake
                else "真实 LLM（agent_judge 场景评分，不复用 RAGAS）"
            ),
            "layers": "L0 任务层 + L1 架构层（离线确定性） + L2 答案质量（LLM-as-a-Judge）",
            "judge": "自定义 agent_judge 场景逐指标判卷（correctness / relevancy / faithfulness，温度 0.1 / 关闭思考）",
            "driver": "真实 LLM（chat 场景）" if real else "FakeChatModel 脚本化决策",
            "runs": runs,
            "robustness": "success_rate = 通过次数 / runs（真实 LLM 驱动下才有统计意义）",
            "cost_source": "usage_metadata（input/output tokens），fake 模式下恒为 0",
            "keyword_source": "answer_keywords 关键词覆盖（确定性旁证，无 LLM）",
            "eval_cases": len(records),
            "answer_source": "事件流最终 message（脚本化决策驱动真实 Harness）",
            "evidence_source": "tool_end 工具执行证据（Faithfulness 事实源）",
            "reference_source": "task_set.jsonl#reference",
        },
        "overall": {
            **l1,
            "modes": modes,
            "answer": {
                **{m: _mean(scored, m) for m in _ANSWER_METRICS},
                "scored_cases": len(scored),
                "faithful_cases": len(faithful),
            },
        },
        "cases": records,
    }
    return records, report


def save_report(report: dict[str, Any], path: str) -> None:
    """把报告写入 JSON（含逐用例行为断言 + 答案质量分 + 工具证据，供失败样本回流分析）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
