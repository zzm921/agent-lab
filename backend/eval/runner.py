"""modular RAG 离线评测运行器：构建确定性方案 → 跑评测集 → 计算检索/路由/闸门指标。

供 scripts/eval_modular.py（CLI 报告）与 tests/test_eval_regression.py（回归门禁）共用。
- 默认（real_router=False）：路由注入评测集期望值，模块全部用规则/确定性实现，
  FakeEmbeddings 离线可跑、不依赖 Key——衡量「给定路由下模块链执行质量」；
- real_router=True：使用真实 LLM 路由（需 Key），额外计算路由准确率。
"""
from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.llm.client import llm_service
from app.llm.fake_model import FakeEmbeddings
from app.memory.stores.memory_store import MemoryStore
from app.rag.routing.query_hyde import RuleHydeExpander
from app.rag.routing.classifier import (
    CITATION,
    COMPARISON,
    DECOMPOSE,
    DIRECT,
    HYBRID,
    MULTIHOP,
    MULTI_RECALL,
    REWRITE,
    SIMPLE,
    VECTOR,
    RouteDecision,
    build_classifier,
)
from app.rag.retrieval.answerability import RuleAnswerabilityVerifier
from app.rag.retrieval.context_compress import ExtractiveContextCompressor
from app.rag.retrieval.iterative_retrieval import PlanExecuteRetriever
from app.rag.retrieval.planner import RuleMultiHopPlanner
from app.rag.retrieval.reranker import LexicalReranker
from app.rag.retrieval.verifier import RuleMultiHopVerifier
from app.rag.routing.query_decompose import RuleQueryDecomposer
from app.rag.routing.query_rewrite import RuleQueryRewriter
from app.rag.schemes.modular import ModularRagScheme

from eval.corpus import CORPUS

# 评测集/报告路径（相对本包）
_EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.jsonl"

# eval_set 中的字符串枚举 → 路由常量
_MODE: dict[str, str] = {"vector": VECTOR, "hybrid": HYBRID, "multi_recall": MULTI_RECALL}
_COMPLEXITY: dict[str, str] = {
    "simple": SIMPLE, "rewrite": REWRITE, "decompose": DECOMPOSE, "multihop": MULTIHOP,
}
_GEN: dict[str, str] = {"direct": DIRECT, "citation": CITATION, "comparison": COMPARISON}

# 语义去重阈值抬到 0.99：本评测语料多为内容相近的相邻制度条目，过低的阈值会把
# 内容不同的分块误判为同义去重，污染检索指标（语义去重另有单测覆盖，此处不评估它）。
_SEMANTIC_THRESHOLD = 0.99

# 字符段（中文/数字连续）：用于二元组提取
_SEG = re.compile(r"[\u4e00-\u9fff0-9]+")


def _bigrams(text: str) -> list[str]:
    """提取中文/数字连续段内的相邻二元组（单字符段单独成词），供词法打分。"""
    grams: list[str] = []
    for seg in _SEG.findall(text):
        if len(seg) == 1:
            grams.append(seg)
        else:
            grams.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return grams


class _BM25Store(MemoryStore):
    """确定性 BM25 词法检索后端（评测专用）：字符二元组词项 + BM25 打分。

    替代 FakeEmbeddings 的字符序号向量（后者对中文语义几乎无区分度，召回近似随机），
    也比 Dice 相似度更接近企业级词法检索（如 Elasticsearch 的 BM25）：
    对词频、逆文档频率、文档长度归一化更敏感，行为与生产检索更一致。
    仍保持确定性、可复现（仅用于评测回归，不代表生产向量检索质量，meta 已标注）。
    """

    name: str = "eval-bm25"
    _K1 = 1.5  # 词频饱和系数（BM25 标准参数）
    _B = 0.75  # 文档长度归一化系数（BM25 标准参数）

    def __init__(self, embeddings, collection: str = "eval_modular"):
        super().__init__(embeddings, collection)
        self._doc_terms: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._avgdl = 0.0

    def add(self, text: str, metadata: dict | None = None) -> None:
        super().add(text, metadata)
        terms = _bigrams(text)
        self._doc_terms.append(terms)
        for t in set(terms):
            self._df[t] = self._df.get(t, 0) + 1
        total = sum(len(ts) for ts in self._doc_terms)
        self._avgdl = total / len(self._doc_terms)

    def clear(self) -> None:
        super().clear()
        self._doc_terms = []
        self._df = {}
        self._avgdl = 0.0

    def _idf(self, term: str) -> float:
        """逆文档频率：出现文档越少，词项区分度越高。"""
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        n = len(self._doc_terms)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 3, volume_filter: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        q_terms = _bigrams(query)
        n = len(self._doc_terms)
        if not q_terms or n == 0:
            return []
        scored: list[tuple[str, float, dict[str, Any] | None]] = []
        for i in range(n):
            doc = self._doc_terms[i]
            dl = len(doc)
            if dl == 0:
                continue
            norm = self._K1 * (1.0 - self._B + self._B * dl / self._avgdl)
            score = 0.0
            for t in q_terms:
                tf = doc.count(t)
                if tf == 0:
                    continue
                score += self._idf(t) * tf * (self._K1 + 1.0) / (tf + norm)
            scored.append((self._store.texts[i], score, self._store.metadatas[i]))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [{"text": t, "score": s, "metadata": m} for t, s, m in scored[:top_k]]


class _ExpectedRouter:
    """按评测集期望值确定性返回路由决策（隔离真实 LLM 路由，衡量模块链质量）。"""

    def __init__(self, decision: RouteDecision):
        self.decision = decision

    def classify(self, query: str) -> RouteDecision:  # noqa: ARG002
        return self.decision


class _RecordingRouter:
    """包装真实/期望路由，记录最近一次决策（供路由准确率比对与可观测）。"""

    def __init__(self, inner):
        self.inner = inner
        self.last: RouteDecision | None = None

    def classify(self, query: str) -> RouteDecision:
        self.last = self.inner.classify(query)
        return self.last


class _EvalDeicticResolver:
    """确定性指代消解：上下文含「助手: <实体>」时把 他/她/它 替换为该实体（离线可测）。"""

    _PRONOUN = re.compile(r"他|她|它")

    def resolve(self, query: str, context: str | None) -> str:
        if not context or not self._PRONOUN.search(query):
            return query
        m = re.search(r"助手[:：]\s*([^\n]+)", context)
        if not m:
            return query
        entity = m.group(1).strip()
        return self._PRONOUN.sub(entity, query) if entity else query


def _decision(expected: dict[str, Any]) -> RouteDecision:
    return RouteDecision(
        retrieval_need=bool(expected["retrieval_need"]),
        retrieval_mode=_MODE[expected["retrieval_mode"]],
        complexity=_COMPLEXITY[expected["complexity"]],
        generation_mode=_GEN[expected["generation_mode"]],
        confidence=0.9,
        reason="评测集期望路由（确定性注入）",
    )


def _build_store():
    """构建评测存储：语料入库并携带稳定 chunk_id 元数据（供命中→相关分块映射）。

    用确定性 BM25 后端替代 FakeEmbeddings（见 _BM25Store），保证检索指标有意义。
    """
    store = _BM25Store(FakeEmbeddings(), collection="eval_modular")
    for item in CORPUS:
        store.add(item["text"], {"chunk_id": item["id"]})
    return store


def _build_scheme(top_k: int, store, router) -> ModularRagScheme:
    return ModularRagScheme(
        FakeEmbeddings(),
        store,
        top_k=top_k,
        classifier=router,
        deictic=_EvalDeicticResolver(),
        rewriter=RuleQueryRewriter(),
        reranker=LexicalReranker(),
        decomposer=RuleQueryDecomposer(),
        # 规则 HyDE 桩：原样返回查询（不触发真实 LLM 假想文档生成），保证离线确定性
        hyde=RuleHydeExpander(),
        compressor=ExtractiveContextCompressor(
            embeddings=FakeEmbeddings(), semantic_threshold=_SEMANTIC_THRESHOLD
        ),
        multi_hop=PlanExecuteRetriever(RuleMultiHopPlanner(), RuleMultiHopVerifier()),
        answerability=RuleAnswerabilityVerifier(),
    )


def _load_cases() -> list[dict[str, Any]]:
    cases = []
    for line in _EVAL_SET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _chunk_id(hit: dict[str, Any]) -> str | None:
    return (hit.get("metadata") or {}).get("chunk_id")


def _evaluate(case: dict[str, Any], result, elapsed_ms: float, route: RouteDecision) -> dict[str, Any]:
    """单用例指标：检索（Recall/Precision/MRR）、关键词覆盖、闸门行为、路由、耗时。"""
    expected = case["expected"]
    hits = result.hits
    need = bool(expected["retrieval_need"])
    relevant = set(case["relevant"])
    retrieved_ids = {cid for h in hits if (cid := _chunk_id(h))}

    record: dict[str, Any] = {
        "id": case["id"],
        "branch": case["branch"],
        "query": case["query"],
        "need_retrieval": need,
        "relevant": sorted(relevant),
        "retrieved_ids": sorted(retrieved_ids),
        "hits": len(hits),
        "recall": None,
        "precision": None,
        "mrr": None,
        "keyword_hit": None,
        "answerable": None,
        "recommendation": None,
        "missing_facts": [],
        "rewrites": result.rewrites,
        "decomposed": result.decomposed,
        "reranked": result.reranked,
        "hops": len(result.hops),
        "compressed": result.compressed,
        "elapsed_ms": round(elapsed_ms, 1),
        "route_used": {
            "retrieval_need": route.retrieval_need,
            "retrieval_mode": route.retrieval_mode,
            "complexity": route.complexity,
            "generation_mode": route.generation_mode,
        },
    }
    if need and relevant:
        inter = retrieved_ids & relevant
        record["recall"] = round(len(inter) / len(relevant), 3)
        record["precision"] = round(len(inter) / len(hits), 3) if hits else 0.0
        for i, h in enumerate(hits, 1):
            if _chunk_id(h) in relevant:
                record["mrr"] = round(1.0 / i, 3)
                break
        else:
            record["mrr"] = 0.0
        kws = case.get("answer_keywords") or []
        if kws:
            joined = "".join((h.get("text") or "") for h in hits)
            record["keyword_hit"] = any(kw in joined for kw in kws)
    if result.answerability:
        record["answerable"] = bool(result.answerability.get("answerable"))
        record["recommendation"] = result.answerability.get("recommendation")
        record["missing_facts"] = result.answerability.get("missing_facts") or []
    return record


def _mean(key: str, records: list[dict[str, Any]]) -> float | None:
    vals = [r[key] for r in records if r[key] is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval = [r for r in records if r["need_retrieval"]]
    n = len(records)
    return {
        "cases": n,
        "retrieval_cases": len(retrieval),
        "avg_recall": _mean("recall", retrieval),
        "avg_precision": _mean("precision", retrieval),
        "avg_mrr": _mean("mrr", retrieval),
        "keyword_coverage": _mean("keyword_hit", retrieval),
        "answerable_rate": _mean("answerable", retrieval),
        "clarify_rate": _mean(
            "recommendation", [dict(r, recommendation=1.0 if r.get("recommendation") == "clarify" else 0.0) for r in retrieval]
        ),
        "avg_elapsed_ms": round(sum(r["elapsed_ms"] for r in records) / n, 1) if n else 0.0,
    }


def run(top_k: int = 3, real_router: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑全量评测集，返回 (逐用例记录, 汇总报告)。

    - 默认：注入期望路由 + 规则模块（确定性回归基线，离线可测）；
    - real_router=True：用真实 LLM 路由（需 Key），额外计算路由准确率。
    """
    store = _build_store()
    cases = _load_cases()
    records: list[dict[str, Any]] = []
    routing_hits = 0

    # 真实路由模式：评测期决策温度归零（指标可复现）；线上默认 0.2 不受影响
    if real_router:
        llm_service.update_profile(
            "rag_classify", params={"temperature": 0, "max_tokens": 500, "enable_thinking": False}
        )

    # 真实路由模式下共用同一个（带记录）路由实例；注入模式按用例期望分别构造
    real_router_obj = _RecordingRouter(build_classifier()) if real_router else None

    for case in cases:
        expected = case["expected"]
        if real_router:
            router = real_router_obj
            scheme = _build_scheme(top_k, store, router)
        else:
            router = _RecordingRouter(_ExpectedRouter(_decision(expected)))
            scheme = _build_scheme(top_k, store, router)

        t0 = time.perf_counter()
        result = scheme.retrieve_full(case["query"], top_k, context=case.get("context"))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        route = router.last
        records.append(_evaluate(case, result, elapsed_ms, route))

        route_used = records[-1]["route_used"]
        if all(
            route_used[k] == expected[k]
            for k in ("retrieval_need", "retrieval_mode", "complexity", "generation_mode")
        ):
            routing_hits += 1

    branches: dict[str, dict[str, Any]] = {}
    for branch in sorted({r["branch"] for r in records}):
        branches[branch] = _aggregate([r for r in records if r["branch"] == branch])

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "top_k": top_k,
            "real_router": real_router,
            "modules": "rule + expected-router" if not real_router else "rule + real LLM router",
            "retrieval": "bm25-bigram (确定性，非生产向量检索)",
            "corpus_size": len(CORPUS),
            "eval_cases": len(records),
        },
        "routing_accuracy": round(routing_hits / len(records), 3),
        "branches": branches,
        "overall": _aggregate(records),
        "cases": records,
    }
    return records, report


def save_report(report: dict[str, Any], path: str) -> None:
    """把报告写入 JSON（含逐用例明细，供失败样本回流分析）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
