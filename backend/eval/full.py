"""modular RAG 全面评测：一份报告覆盖路由/检索/生成/拒答四类维度。

替代原「L2 手写 judge + L3 RAGAS」双层语义评测（指标重叠、需交叉对比），统一为单层：
- 路由+检索+闸门：复用 runner 的确定性模块链（期望路由注入 + BM25，离线可复现），
  产出路由准确率、Recall/Precision/MRR、关键词覆盖、可答/澄清率、耗时；
- 生成：真实 LLM（chat 场景）基于检索命中生成答案，与线上上下文注入口径一致；
- 生成质量：RAGAS 标准指标 Faithfulness（忠实度）/ AnswerRelevancy（相关性）/
  AnswerCorrectness（答案正确性，对照金标）/ ContextPrecision（上下文精确度，
  检索块对金标的支撑度），金标参考答案维护在 eval_set.jsonl 的 reference 字段；
- 拒答行为：库外问题由手写 judge 判定「拒绝回答而非编造」（grounded），
  RAGAS 无此维度；库外用例不进 RAGAS 评分（期望答案为拒答，标准指标无意义）。

两种模式（scripts/eval_full.py 选择）：
- 默认（真实 LLM）：生成用 chat 场景、拒答 judge 用 rag_judge 场景、RAGAS 内部 LLM
  用 rag_ragas 场景（关闭思考：高频小 JSON 提取，开思考会整批超时）、
  embeddings 用 DashScopeEmbeddings，需配置 LLM_API_KEY 与 EMBEDDING_API_KEY；
- fake 模式：FakeChatModel / FakeEmbeddings 占位，离线冒烟验证链路可跑通，
  分数为占位值、无评测意义（仅 CI/离线自检用）。fake 占位文本非合法 JSON，
  RAGAS 结构化输出解析必然失败且触发 fix 重试级联，因此 fake 模式仅对前
  FAKE_EVAL_LIMIT 个样本调用 evaluate() 验证链路，检索与生成仍覆盖全量用例。

依赖说明：RAGAS 0.4.x 硬导入 langchain_community.chat_models.vertexai，
需 langchain-community<0.4（0.4 起移除该模块）；本项目 App 不依赖
langchain-community，锁版无副作用。
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
# legacy 指标：兼容 langchain BaseLanguageModel 直接注入（evaluate 内部包装驱动）；
# collections 版新指标强制 instructor 架构 LLM，与项目自定义模型不兼容。
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._faithfulness import Faithfulness

from app.core.errors import ConfigError
from app.llm.client import create_embeddings, get_chat_model
from app.llm.fake_model import FakeChatModel, FakeEmbeddings

from eval import runner, semantic

# fake 模式下最多送入 evaluate() 的样本数（占位文本解析必败，只验证链路不追求覆盖）
FAKE_EVAL_LIMIT = 3

# 生成质量指标（RAGAS 标准分，逐用例字段与报告键共用）
_GEN_METRICS = ("faithfulness", "answer_relevancy", "answer_correctness", "context_precision")


def _to_score(value: Any) -> float | None:
    """RAGAS 失败行返回 NaN → 归一为 None，便于汇总与报告。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 3)


def _build_models(fake: bool) -> tuple:
    """按模式构建（generator, judge, ragas_llm, embeddings, run_config）。"""
    if fake:
        generator = FakeChatModel(
            script=[AIMessage(content="（fake 生成占位：依据资料作答）")] * 64
        )
        judge = FakeChatModel(
            script=[AIMessage(content='{"grounded": true, "reason": "fake 冒烟占位"}')] * 64
        )
        llm = FakeChatModel(script=[])  # 队列空 → 返回默认占位回答
        embeddings = FakeEmbeddings()
        # fake 占位文本解析必败：重试=1、不等待，快速失败冒烟链路
        run_config = RunConfig(timeout=30, max_retries=1, max_wait=0, max_workers=4)
        return generator, judge, llm, embeddings, run_config

    generator = get_chat_model("chat")
    judge = get_chat_model("rag_judge")
    llm = get_chat_model("rag_ragas")
    if generator is None or judge is None or llm is None:
        raise ConfigError(
            "全面评测需要真实 LLM（chat 生成 / rag_judge 拒答判定 / rag_ragas 评分）："
            "请配置 LLM_API_KEY。离线冒烟可用 --fake（仅验证链路，无评测意义）。"
        )
    try:
        embeddings = create_embeddings(fake=False)
    except ConfigError as exc:
        raise ConfigError(
            "全面评测需要 Embedding（RAGAS AnswerRelevancy/AnswerCorrectness）："
            "请配置 EMBEDDING_API_KEY。"
        ) from exc
    # 真实模式：失败重试 3 次、退避上限 10s、8 并发（默认 10 次/60s 退避会让
    # 偶发解析失败累积成数分钟级延迟）
    run_config = RunConfig(timeout=120, max_retries=3, max_wait=10, max_workers=8)
    return generator, judge, llm, embeddings, run_config


def _stage_semantic(top_k: int, fake: bool) -> list[dict[str, Any]]:
    """生成+评分阶段：检索用例生成答案 → RAGAS 标准评分 / 库外拒答判定。"""
    store = runner._build_store()
    cases = runner._load_cases()
    generator, judge, llm, embeddings, run_config = _build_models(fake)

    records: list[dict[str, Any]] = []
    samples: list[SingleTurnSample] = []
    sample_rec_idx: list[int] = []  # 每个样本对应 records 的下标

    for case in cases:
        expected = case["expected"]
        rec: dict[str, Any] = {
            "id": case["id"],
            "branch": case["branch"],
            "query": case["query"],
            "out_of_kb": bool(case.get("out_of_kb")),
            "reference": case.get("reference"),
            "answer": None,
            **{m: None for m in _GEN_METRICS},
            "grounded": None,
            "judge_reason": None,
            "gen_elapsed_ms": None,
            "error": None,
        }
        if not expected["retrieval_need"]:
            # 不检索分支不进入语义评分（生成无需上下文，暂不计分）
            rec["skipped"] = True
            records.append(rec)
            continue

        router = runner._RecordingRouter(runner._ExpectedRouter(runner._decision(expected)))
        scheme = runner._build_scheme(top_k, store, router)
        result = scheme.retrieve_full(case["query"], top_k, context=case.get("context"))
        context = semantic._format_context(result.hits)
        rec["retrieved_ids"] = sorted(
            {(h.get("metadata") or {}).get("chunk_id") for h in result.hits}
        )

        # 生成：真实 LLM 或 fake 占位
        try:
            t0 = time.perf_counter()
            resp = generator.invoke(semantic._gen_prompt(case["query"], context))
            rec["gen_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            answer = resp.content if isinstance(resp.content, str) else str(resp.content or "")
        except Exception as exc:  # noqa: BLE001 — 单用例失败不中断全量
            rec["error"] = f"generate 失败: {exc}"
            records.append(rec)
            continue
        rec["answer"] = answer

        if rec["out_of_kb"]:
            # 库外用例：只做拒答判定（期望答案为拒答，RAGAS 标准指标无意义）
            try:
                verdict = semantic.judge_out_of_kb(case["query"], answer, context, judge)
                rec["grounded"] = verdict["grounded"]
                rec["judge_reason"] = verdict["reason"]
            except Exception as exc:  # noqa: BLE001 — 解析失败记为 error，不中断全量
                rec["error"] = f"judge 解析失败: {exc}"
            records.append(rec)
            continue

        # 常规检索用例：进入 RAGAS 标准评分（携带金标 reference）
        idx = len(records)
        records.append(rec)
        samples.append(
            SingleTurnSample(
                user_input=case["query"],
                retrieved_contexts=[h.get("text", "") for h in result.hits],
                response=answer,
                reference=case.get("reference"),
            )
        )
        sample_rec_idx.append(idx)

    if samples:
        # fake 占位文本解析必败、重试级联极慢，仅评估前 FAKE_EVAL_LIMIT 个样本验证链路；
        # 真实模式评估全部样本。
        eval_samples = samples[:FAKE_EVAL_LIMIT] if fake else samples
        dataset = EvaluationDataset(samples=eval_samples)
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), AnswerCorrectness(), ContextPrecision()],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            show_progress=False,
        )
        # result.scores 与 eval_samples 顺序一一对应；回填到对应记录的槽位
        for j, rec_i in enumerate(sample_rec_idx[: len(eval_samples)]):
            rec = records[rec_i]
            if rec.get("error") is not None:
                continue
            s = result.scores[j]
            for m in _GEN_METRICS:
                rec[m] = _to_score(s.get(m))

    return records


def run(top_k: int = 3, fake: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑全面评测（确定性层 + 生成语义层），返回 (逐用例记录, 汇总报告)。

    fake=True：生成/judge/RAGAS 用 Fake 模型占位，离线冒烟链路（分数无评测意义）；
    fake=False：真实 LLM（chat 生成 + rag_judge 判定 + RAGAS 评分），缺 Key 抛 ConfigError。
    """
    # 确定性层：路由/检索/闸门（runner 内部重建 store，两次检索结果确定一致）
    _, l1_report = runner.run(top_k=top_k, real_router=False)
    records = _stage_semantic(top_k, fake)

    # 合并逐用例记录：语义记录挂上确定性层的检索指标
    l1_by_id = {r["id"]: r for r in l1_report["cases"]}
    for r in records:
        base = l1_by_id.get(r["id"])
        if base:
            r["retrieval"] = {
                k: base[k]
                for k in ("recall", "precision", "mrr", "keyword_hit", "answerable", "elapsed_ms")
                if k in base
            }

    scored = [r for r in records if not r.get("skipped") and r["error"] is None and not r["out_of_kb"]]
    oob = [r for r in records if r["out_of_kb"] and r["error"] is None and not r.get("skipped")]
    generation = {
        f"avg_{m}": (round(sum(r[m] for r in scored if r[m] is not None) / len(scored), 3) if scored else None)
        for m in _GEN_METRICS
    }

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "top_k": top_k,
            "mode": "fake 冒烟（无评测意义）" if fake else "real LLM (chat 生成 + rag_judge 拒答判定 + RAGAS 评分)",
            "metrics": list(_GEN_METRICS) + ["out_of_kb_grounded"],
            "retrieval": "bm25-bigram (确定性，非生产向量检索)",
            "reference_source": "eval_set.jsonl#reference",
            "corpus_size": len(runner.CORPUS),
            "eval_cases": len(records),
        },
        "routing": {"routing_accuracy": l1_report["routing_accuracy"]},
        "retrieval": {"overall": l1_report["overall"], "branches": l1_report["branches"]},
        "generation": {**generation, "scored_cases": len(scored)},
        "out_of_kb": {
            "grounded_rate": round(sum(1 for r in oob if r["grounded"]) / len(oob), 3) if oob else None,
            "cases": len(oob),
        },
        "cases": records,
    }
    return records, report


def save_report(report: dict[str, Any], path: str) -> None:
    """把全面评测报告写入 JSON（含逐用例答案/检索指标/评分，供失败样本回流分析）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
