"""Agent 全量评测 · 答案质量层（LLM-as-a-Judge）。

在 runner（L0 任务层约束 + L1 架构层不变量，全部离线确定性）之上叠加 L2 答案语义评分，
形成「行为层离线断言 + 语义层 LLM 评分」的企业级双层评测：

- 答案来源：事件流最终 message（最后一次工具执行之后的所有 message delta 拼接），
  由 FakeChatModel 脚本化决策驱动真实 Harness 产出——隔离生成随机性，聚焦评分链路本身；
- 事实证据：事件流 tool_end 的 tool(args) → result，构造「工具执行证据」列表，填入 RAGAS
  retrieved_contexts 槽位——Faithfulness 退化为「答案是否忠实于工具结果」（对应 RAG 的忠实度）；
- 评分指标（复用 eval/rag/full.py 的 RAGAS 基建，不新写 judge）：
    answer_correctness（对照金标 reference）/ answer_relevancy（问题相关性）/
    faithfulness（事实一致性，仅对产出工具证据的用例评分，无证据用例只评前两者）；
- 两种模式：fake 离线冒烟（占位文本解析必败，仅前 FAKE_EVAL_LIMIT 个样本验证链路）/
  真实 LLM（需 LLM_API_KEY 与 EMBEDDING_API_KEY，rag_ragas 场景评分）；
- 审批 / 轮数上限用例（无最终答案）不进入 L2 评分，标记 answer_scored=False。

报告：合并 L0/L1/L2 三层，逐用例带行为断言 + 答案质量分，按模式汇总均分。
"""
from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
# legacy 指标：兼容 langchain BaseLanguageModel 直接注入（与 eval/rag/full.py 一致）
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._faithfulness import Faithfulness

from app.core.errors import ConfigError
from app.core.version import get_app_version
from app.llm.client import create_embeddings, get_chat_model
from app.llm.fake_model import FakeChatModel, FakeEmbeddings

from eval.agent import runner

# fake 模式下最多送入 evaluate() 的样本数（占位文本解析必败，只验证链路不追求覆盖）
FAKE_EVAL_LIMIT = 3

# L2 答案质量指标（逐用例字段与报告键共用）
_ANSWER_METRICS = ("answer_correctness", "answer_relevancy", "faithfulness")
_METRIC_FACTORY = {
    "answer_correctness": AnswerCorrectness,
    "answer_relevancy": AnswerRelevancy,
    "faithfulness": Faithfulness,
}


def _to_score(value: Any) -> float | None:
    """RAGAS 失败行返回 NaN → 归一为 None，便于汇总与报告。"""
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
# L2 评分
# ---------------------------------------------------------------------------
def _build_models(fake: bool) -> tuple:
    """按模式构建（评分 llm, embeddings, run_config）。评分用 rag_ragas 场景（关闭思考）。"""
    if fake:
        llm = FakeChatModel(script=[])  # 队列空 → 返回默认占位回答，解析必败快速冒烟
        embeddings = FakeEmbeddings()
        run_config = RunConfig(timeout=30, max_retries=1, max_wait=0, max_workers=4)
        return llm, embeddings, run_config

    llm = get_chat_model("rag_ragas")
    if llm is None:
        raise ConfigError(
            "Agent 答案质量评测需要真实 LLM（rag_ragas 场景评分）：请配置 LLM_API_KEY。"
            "离线冒烟可用 --fake（仅验证链路，无评测意义）。"
        )
    try:
        embeddings = create_embeddings(fake=False)
    except ConfigError as exc:
        raise ConfigError(
            "Agent 答案质量评测需要 Embedding（AnswerRelevancy/AnswerCorrectness）："
            "请配置 EMBEDDING_API_KEY。"
        ) from exc
    run_config = RunConfig(timeout=120, max_retries=3, max_wait=10, max_workers=8)
    return llm, embeddings, run_config


def _evaluate_batch(recs: list[dict[str, Any]], metrics: list[str], fake: bool, llm, embeddings, run_config) -> None:
    """对一批记录跑 RAGAS 指定指标并回填（fake 模式仅前 FAKE_EVAL_LIMIT 个样本）。"""
    if not recs:
        return
    samples = [
        SingleTurnSample(
            user_input=r["query"],
            response=r["answer"],
            reference=r["reference"],
            retrieved_contexts=r["evidence"],
        )
        for r in recs
    ]
    eval_samples = samples[:FAKE_EVAL_LIMIT] if fake else samples
    metric_objs = [_METRIC_FACTORY[m]() for m in metrics]
    result = evaluate(
        dataset=EvaluationDataset(samples=eval_samples),
        metrics=metric_objs,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        show_progress=False,
    )
    for j, rec in enumerate(recs[: len(eval_samples)]):
        s = result.scores[j]
        for m in metrics:
            rec[m] = _to_score(s.get(m))


def _stage_answer_score(records: list[dict[str, Any]], fake: bool) -> None:
    """L2 评分：有工具证据的用例评三维（含 faithfulness）；无证据用例只评正确性/相关性。"""
    llm, embeddings, run_config = _build_models(fake)
    evid = [r for r in records if r["answer_scored"] and r["evidence"]]
    plain = [r for r in records if r["answer_scored"] and not r["evidence"]]
    if evid:
        _evaluate_batch(evid, ["faithfulness", "answer_correctness", "answer_relevancy"], fake, llm, embeddings, run_config)
    if plain:
        _evaluate_batch(plain, ["answer_correctness", "answer_relevancy"], fake, llm, embeddings, run_config)


# ---------------------------------------------------------------------------
# 主流程 / 报告
# ---------------------------------------------------------------------------
def run(fake: bool = False, runs: int = 1, real: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑全量 Agent 评测（L0/L1 离线断言 + L2 LLM-judge 答案评分）。

    fake=True：评分用 Fake 模型占位，离线冒烟链路（分数无评测意义）；
    fake=False：真实 LLM（rag_ragas 场景评分），缺 Key 抛 ConfigError。
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
        rec["evidence"] = _extract_evidence(events)
        rec["answer_scored"] = bool(rec["answer"]) and bool(rec["reference"])
        for m in _ANSWER_METRICS:
            rec[m] = None
        records.append(rec)

    _stage_answer_score(records, fake)

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
                else "真实 LLM（rag_ragas 场景评分）"
            ),
            "layers": "L0 任务层 + L1 架构层（离线确定性） + L2 答案质量（LLM-as-a-Judge）",
            "driver": "真实 LLM（chat 场景）" if real else "FakeChatModel 脚本化决策",
            "runs": runs,
            "robustness": "success_rate = 通过次数 / runs（真实 LLM 驱动下才有统计意义）",
            "cost_source": "usage_metadata（input/output tokens），fake 模式下恒为 0",
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
