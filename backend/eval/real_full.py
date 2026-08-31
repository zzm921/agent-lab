"""真实 modular RAG 全面评测：真实方案链路（Qdrant+ES 双后端）+ 真实 LLM + RAGAS。

与 full.py（BM25 确定性链路）的差别：
- 检索：ModularRagScheme.retrieve_full 走真实语义路由 → 执行计划 → Qdrant 稠密 +
  ES BM25 多路召回 + 重排/多跳，不是 BM25 手造 store；
- 路由：不注入期望路由，改为观测方案自身的编排行为（rewrite/decompose/multi_hop/
  compress 使用率），语义路由质量不单独打分；
- 检索指标：evidence 覆盖口径（金标集 real_eval_set.jsonl 的 evidence 条款原文是否被
  检回），容忍 modular 父子扩展返回 parent 级文本；
- 生成/评分/拒答：与 full.py 同构（chat 生成、rag_judge 拒答判定、rag_ragas 评 RAGAS）。

库内混有 105 条 builtin 种子语料（非云帆制度汇编），评测时过滤其命中（不进指标、
不进生成上下文），并在报告中记录过滤量。

fake 模式只占位生成/评分 LLM，检索仍走真实 Qdrant（query 向量化需真实 embeddings；
严禁用 FakeEmbeddings 构造 QdrantStore——维度不符会触发 _ensure_collection 删库重建）。
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._faithfulness import Faithfulness

from app.config import settings
from app.core.errors import ConfigError
from app.llm.client import create_embeddings, get_chat_model
from app.llm.fake_model import FakeChatModel
from app.rag.manager import RagManager

from eval import semantic
from eval.full import _GEN_METRICS, _to_score

BASE = Path(__file__).resolve().parent
SET_PATH = BASE / "real_eval_set.jsonl"

# fake 模式下最多送入 evaluate() 的样本数（与 full.py 同理：占位解析必败，只验证链路）
FAKE_EVAL_LIMIT = 3


def _norm(s: str) -> str:
    return "".join(s.split())


def _load_cases() -> list[dict[str, Any]]:
    return [json.loads(l) for l in SET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def _filter_builtin(hits: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """过滤内置种子语料命中（评测对象是云帆制度汇编检索）。"""
    kept = [h for h in hits if (h.get("metadata") or {}).get("source") != "builtin"]
    return kept, len(hits) - len(kept)


def _ir_metrics(evidence: list[str], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """检索指标（evidence 覆盖口径，容忍 modular 父子扩展返回 parent 级文本）：

    - recall：被任一命中覆盖的 evidence 比例（norm 后包含判定）；
    - precision：覆盖了至少一条 evidence 的命中占比；
    - mrr：首个覆盖任一 evidence 的命中的名次倒数。
    """
    norm_evs = [_norm(ev) for ev in evidence if _norm(ev)]
    if not norm_evs:
        return {"recall": None, "precision": None, "mrr": None, "covered_evidence": 0}
    norm_hits = [_norm(h.get("text", "")) for h in hits]
    covered = sum(1 for ev in norm_evs if any(ev in nh for nh in norm_hits))
    first_cover_rank = 0
    cover_hits = 0
    for rank, nh in enumerate(norm_hits, start=1):
        if any(ev in nh for ev in norm_evs):
            cover_hits += 1
            if first_cover_rank == 0:
                first_cover_rank = rank
    return {
        "recall": round(covered / len(norm_evs), 3),
        "precision": round(cover_hits / len(norm_hits), 3) if norm_hits else 0.0,
        "mrr": round(1.0 / first_cover_rank, 3) if first_cover_rank else 0.0,
        "covered_evidence": covered,
    }


def _build_real_scheme(top_k: int):
    """构建与线上完全一致的 modular 方案（真实 embeddings + Qdrant/ES 多后端）。"""
    embeddings = create_embeddings(fake=False)
    return RagManager(settings, embeddings, top_k=top_k, scheme_ids=["modular"]).schemes["modular"]


def _build_gen_models(fake: bool):
    """生成/评分模型：真实（chat + rag_judge + rag_ragas）或 fake 占位。"""
    if fake:
        return (
            FakeChatModel(script=[AIMessage(content="（fake 生成占位：依据资料作答）")] * 64),
            FakeChatModel(script=[AIMessage(content='{"grounded": true, "reason": "fake 冒烟占位"}')] * 64),
            FakeChatModel(script=[]),
        )
    generator = get_chat_model("chat")
    judge = get_chat_model("rag_judge")
    llm = get_chat_model("rag_ragas")
    if generator is None or judge is None or llm is None:
        raise ConfigError(
            "真实评测需要 LLM（chat 生成 / rag_judge 拒答判定 / rag_ragas 评分）："
            "请配置 LLM_API_KEY。"
        )
    return generator, judge, llm


def run(top_k: int = 3, fake: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑真实 modular 全面评测，返回 (逐用例记录, 汇总报告)。"""
    scheme = _build_real_scheme(top_k)
    cases = _load_cases()
    generator, judge, llm = _build_gen_models(fake)
    embeddings = create_embeddings(fake=False)  # RAGAS 相关性/正确性指标用
    run_config = (
        RunConfig(timeout=30, max_retries=1, max_wait=0, max_workers=4)
        if fake
        else RunConfig(timeout=180, max_retries=3, max_wait=10, max_workers=8)
    )

    records: list[dict[str, Any]] = []
    samples: list[SingleTurnSample] = []
    sample_rec_idx: list[int] = []
    builtin_filtered_total = 0

    for case in cases:
        rec: dict[str, Any] = {
            "id": case["id"],
            "branch": case["branch"],
            "difficulty": case["difficulty"],
            "query": case["query"],
            "out_of_kb": not case["answerable"],
            "reference": case.get("reference"),
            "evidence": case.get("evidence", []),
            "answer": None,
            **{m: None for m in _GEN_METRICS},
            "grounded": None,
            "judge_reason": None,
            "retrieve_elapsed_ms": None,
            "gen_elapsed_ms": None,
            "error": None,
        }
        # 检索：真实方案全链路（语义路由 → 执行计划 → 双后端召回 → 重排/多跳）
        try:
            t0 = time.perf_counter()
            result = scheme.retrieve_full(case["query"], top_k)
            rec["retrieve_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        except Exception as exc:  # noqa: BLE001 — 单用例失败不中断全量
            rec["error"] = f"retrieve 失败: {exc}"
            records.append(rec)
            continue
        hits, builtin_filtered = _filter_builtin(result.hits)
        builtin_filtered_total += builtin_filtered
        rec["builtin_hits"] = builtin_filtered
        rec["behavior"] = {
            "rewrites": len(result.rewrites),
            "decomposed": len(result.decomposed),
            "hops": len(result.hops),
            "compressed": bool(result.compressed),
        }
        if not rec["out_of_kb"]:
            ir = _ir_metrics(case.get("evidence", []), hits)
            rec.update(
                {
                    "recall": ir["recall"],
                    "precision": ir["precision"],
                    "mrr": ir["mrr"],
                    "covered_evidence": ir["covered_evidence"],
                }
            )
            ans_meta = result.answerability or {}
            rec["gate_answerable"] = bool(ans_meta.get("answerable"))

        # 生成（上下文用过滤后的云帆命中）
        context = semantic._format_context(hits)
        try:
            t0 = time.perf_counter()
            resp = generator.invoke(semantic._gen_prompt(case["query"], context))
            rec["gen_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            answer = resp.content if isinstance(resp.content, str) else str(resp.content or "")
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"generate 失败: {exc}"
            records.append(rec)
            continue
        rec["answer"] = answer

        if rec["out_of_kb"]:
            try:
                verdict = semantic.judge_out_of_kb(case["query"], answer, context, judge)
                rec["grounded"] = verdict["grounded"]
                rec["judge_reason"] = verdict["reason"]
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"judge 解析失败: {exc}"
            records.append(rec)
            continue

        idx = len(records)
        records.append(rec)
        samples.append(
            SingleTurnSample(
                user_input=case["query"],
                retrieved_contexts=[h.get("text", "") for h in hits],
                response=answer,
                reference=case.get("reference"),
            )
        )
        sample_rec_idx.append(idx)

    # RAGAS 评分
    if samples:
        eval_samples = samples[:FAKE_EVAL_LIMIT] if fake else samples
        result = evaluate(
            dataset=EvaluationDataset(samples=eval_samples),
            metrics=[Faithfulness(), AnswerRelevancy(), AnswerCorrectness(), ContextPrecision()],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            show_progress=False,
        )
        for j, rec_i in enumerate(sample_rec_idx[: len(eval_samples)]):
            rec = records[rec_i]
            if rec.get("error") is not None:
                continue
            s = result.scores[j]
            for m in _GEN_METRICS:
                rec[m] = _to_score(s.get(m))

    # 汇总
    scored = [r for r in records if r["error"] is None and not r["out_of_kb"]]
    oob = [r for r in records if r["out_of_kb"] and r["error"] is None]
    gen = {
        f"avg_{m}": (
            round(
                sum(r[m] for r in scored if r[m] is not None)
                / max(1, sum(1 for r in scored if r[m] is not None)),
                3,
            )
            if any(r[m] is not None for r in scored)
            else None
        )
        for m in _GEN_METRICS
    }
    behavior = {
        "rewrite_rate": round(sum(1 for r in scored if r["behavior"]["rewrites"]) / max(len(scored), 1), 3),
        "decompose_rate": round(sum(1 for r in scored if r["behavior"]["decomposed"]) / max(len(scored), 1), 3),
        "multi_hop_rate": round(sum(1 for r in scored if r["behavior"]["hops"]) / max(len(scored), 1), 3),
        "compress_rate": round(sum(1 for r in scored if r["behavior"]["compressed"]) / max(len(scored), 1), 3),
    }

    def _avg(rs: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "top_k": top_k,
            "mode": "fake 冒烟（真实检索 + 占位生成，无评测意义）"
            if fake
            else "real（Qdrant+ES 双后端 + chat 生成 + RAGAS）",
            "metrics": list(_GEN_METRICS) + ["out_of_kb_grounded"],
            "scheme": "modular（语义路由 + 执行计划 + 多路召回 + 重排/多跳）",
            "ir_metric_semantics": "evidence 覆盖口径：金标依据条款原文被任一命中覆盖（norm 包含）",
            "builtin_filtered": builtin_filtered_total,
            "reference_source": "real_eval_set.jsonl#reference",
            "eval_cases": len(records),
        },
        "retrieval": {
            "overall": {
                "cases": len(scored),
                "avg_recall": _avg(scored, "recall"),
                "avg_precision": _avg(scored, "precision"),
                "avg_mrr": _avg(scored, "mrr"),
                "gate_answerable_rate": round(
                    sum(1 for r in scored if r.get("gate_answerable")) / max(len(scored), 1), 3
                ),
                "avg_retrieve_ms": _avg(scored, "retrieve_elapsed_ms"),
            },
            "by_difficulty": {
                d: {
                    "cases": len([r for r in scored if r["difficulty"] == d]),
                    "avg_recall": _avg([r for r in scored if r["difficulty"] == d], "recall"),
                    "avg_mrr": _avg([r for r in scored if r["difficulty"] == d], "mrr"),
                }
                for d in ("basic", "advanced", "challenge")
            },
            "by_branch": {
                b: {
                    "cases": len([r for r in scored if r["branch"] == b]),
                    "avg_recall": _avg([r for r in scored if r["branch"] == b], "recall"),
                }
                for b in ("single_point", "compare", "multi_hop")
            },
        },
        "behavior": behavior,
        "generation": {**gen, "scored_cases": len(scored)},
        "out_of_kb": {
            "grounded_rate": round(sum(1 for r in oob if r["grounded"]) / len(oob), 3) if oob else None,
            "cases": len(oob),
        },
        "cases": records,
    }
    return records, report


def save_report(report: dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
