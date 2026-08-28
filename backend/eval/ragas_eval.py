"""L3 语义评测（RAGAS 库）：确定性检索 + 真实 LLM 生成 + RAGAS 标准指标评分。

在 L1（确定性回归，机制正确性）与 L2（手写 judge，语义质量）之上，
用 RAGAS 库的指标对生成答案做标准化评分，作为独立一层：
- 检索：复用 runner 的确定性模块链（期望路由 + BM25 + 验证闸门），上下文可复现；
- 生成：真实 LLM 基于检索命中生成答案（上下文注入用户消息，与线上一致）；
- 评分：RAGAS Faithfulness（忠实度）+ AnswerRelevancy（答案相关性），
  与 L2 手写 judge 的两维对齐，可对比「手写 judge vs 标准库」评分差异；
  两者均无需金标参考答案（ContextRecall/Precision 需金标，暂不纳入）。

两种模式（scripts/eval_ragas.py 选择）：
- 默认（真实 LLM）：生成与 RAGAS 内部 LLM 都用 chat 场景、embeddings 用
  DashScopeEmbeddings，需配置 LLM_API_KEY 与 EMBEDDING_API_KEY；
- fake 模式：FakeChatModel / FakeEmbeddings 占位，离线冒烟验证链路可跑通，
  分数为占位值、无评测意义（仅 CI/离线自检用）。fake 占位文本非合法 JSON，
  RAGAS 结构化输出解析必然失败且触发 fix 重试级联（单个样本可累计数十秒），
  因此 fake 模式仅对前 FAKE_EVAL_LIMIT 个样本调用 evaluate() 验证链路，
  检索与生成仍覆盖全量用例。

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

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
# legacy 指标：兼容 langchain BaseLanguageModel 直接注入（evaluate 内部包装驱动）；
# collections 版新指标强制 instructor 架构 LLM，与项目自定义模型不兼容。
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._faithfulness import Faithfulness

from app.core.errors import ConfigError
from app.llm.client import create_embeddings, get_chat_model
from app.llm.fake_model import FakeChatModel, FakeEmbeddings

from eval import runner, semantic

# fake 模式下最多送入 evaluate() 的样本数（占位文本解析必败，只验证链路不追求覆盖）
FAKE_EVAL_LIMIT = 3


def _to_score(value: Any) -> float | None:
    """RAGAS 失败行返回 NaN → 归一为 None，便于汇总与报告。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 3)


def _run_single(case: dict[str, Any], store, generator, top_k: int) -> tuple[dict[str, Any], SingleTurnSample | None]:
    """单个用例：确定性检索 → LLM 生成答案。返回 (记录, 待评测样本)。"""
    expected = case["expected"]
    if not expected["retrieval_need"]:
        # 不检索分支不进入 RAGAS 语义评分（生成无需上下文，暂不计分）
        return {"id": case["id"], "branch": case["branch"], "query": case["query"], "skipped": True}, None

    router = runner._RecordingRouter(runner._ExpectedRouter(runner._decision(expected)))
    scheme = runner._build_scheme(top_k, store, router)
    t0 = time.perf_counter()
    result = scheme.retrieve_full(case["query"], top_k, context=case.get("context"))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    hits = result.hits
    context = semantic._format_context(hits)
    rec: dict[str, Any] = {
        "id": case["id"],
        "branch": case["branch"],
        "query": case["query"],
        "retrieved_ids": sorted(
            {(h.get("metadata") or {}).get("chunk_id") for h in hits}
        ),
        "retrieved_count": len(hits),
        "answerable": bool(result.answerability.get("answerable")) if result.answerability else None,
        "elapsed_ms": round(elapsed_ms, 1),
        "answer": None,
        "faithfulness": None,
        "answer_relevancy": None,
        "error": None,
    }

    try:
        resp = generator.invoke(semantic._gen_prompt(case["query"], context))
        answer = resp.content if isinstance(resp.content, str) else str(resp.content or "")
    except Exception as exc:  # noqa: BLE001 — 单用例失败不中断全量
        rec["error"] = f"generate 失败: {exc}"
        return rec, None
    rec["answer"] = answer
    sample = SingleTurnSample(
        user_input=case["query"],
        retrieved_contexts=[h.get("text", "") for h in hits],
        response=answer,
    )
    return rec, sample


def run(top_k: int = 3, fake: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑 L3 RAGAS 语义评测（全部检索用例），返回 (逐用例记录, 汇总报告)。

    fake=True：生成/RAGAS 用 FakeChatModel、embeddings 用 FakeEmbeddings，
    离线冒烟链路（分数无评测意义）；
    fake=False：真实 LLM（chat 场景）+ DashScopeEmbeddings，缺 Key 抛 ConfigError。
    """
    store = runner._build_store()
    cases = runner._load_cases()

    if fake:
        n = len(cases)
        llm = FakeChatModel(script=[] * n)  # 队列空 → 返回默认占位回答
        embeddings = FakeEmbeddings()
        # fake 占位文本非合法 JSON，解析必然失败：重试=1、不等待，快速失败冒烟链路
        run_config = RunConfig(timeout=30, max_retries=1, max_wait=0, max_workers=4)
    else:
        llm = get_chat_model("chat")
        if llm is None:
            raise ConfigError(
                "L3 RAGAS 评测需要真实 LLM（chat 场景）：请配置 LLM_API_KEY。"
                "离线冒烟可用 --fake（仅验证链路，无评测意义）。"
            )
        try:
            embeddings = create_embeddings(fake=False)
        except ConfigError as exc:
            raise ConfigError(
                "L3 RAGAS 评测需要 Embedding（AnswerRelevancy 计算相似度）："
                "请配置 EMBEDDING_API_KEY。"
            ) from exc
        # 真实模式：失败重试 3 次、退避上限 10s、8 并发（默认 10 次/60s 退避会让
        # 偶发解析失败累积成数分钟级延迟）
        run_config = RunConfig(timeout=120, max_retries=3, max_wait=10, max_workers=8)

    records: list[dict[str, Any]] = []
    samples: list[SingleTurnSample] = []
    sample_rec_idx: list[int] = []  # 每个样本对应 records 的下标
    for i, case in enumerate(cases):
        rec, sample = _run_single(case, store, llm, top_k)
        records.append(rec)
        if sample is not None:
            samples.append(sample)
            sample_rec_idx.append(i)

    if samples:
        # fake 占位文本解析必败、重试级联极慢，仅评估前 FAKE_EVAL_LIMIT 个样本验证链路；
        # 真实模式评估全部样本。
        eval_samples = samples[:FAKE_EVAL_LIMIT] if fake else samples
        dataset = EvaluationDataset(samples=eval_samples)
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy()],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            show_progress=False,
        )
        # result.scores 与 eval_samples 顺序一一对应；回填到对应记录的槽位
        for idx, rec_i in enumerate(sample_rec_idx[: len(eval_samples)]):
            rec = records[rec_i]
            if rec.get("error") is not None:
                continue
            s = result.scores[idx]
            rec["faithfulness"] = _to_score(s.get("faithfulness"))
            rec["answer_relevancy"] = _to_score(s.get("answer_relevancy"))

    scored = [r for r in records if not r.get("skipped") and r.get("error") is None]
    fs = [r["faithfulness"] for r in scored if r["faithfulness"] is not None]
    ars = [r["answer_relevancy"] for r in scored if r["answer_relevancy"] is not None]

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "top_k": top_k,
            "mode": "fake 冒烟（无评测意义）" if fake else "real LLM (chat 生成) + RAGAS 评分",
            "metrics": ["faithfulness", "answer_relevancy"],
            "retrieval": "bm25-bigram (确定性，非生产向量检索)",
            "corpus_size": len(runner.CORPUS),
            "eval_cases": len(records),
        },
        "avg_faithfulness": round(sum(fs) / len(fs), 3) if fs else None,
        "avg_answer_relevancy": round(sum(ars) / len(ars), 3) if ars else None,
        "scored_cases": len(scored),
        "cases": records,
    }
    return records, report


def save_report(report: dict[str, Any], path: str) -> None:
    """把 L3 报告写入 JSON（含逐用例答案与评分，供失败样本回流分析）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
