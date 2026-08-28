"""L2 语义评测运行器：真实 LLM 完整链路（检索 → 生成 → judge 评分）。

在 L1 确定性回归（机制正确性）之上补语义质量层，回答「手造数据测不出」的部分：
- 检索：复用 runner 的确定性模块链（期望路由 + BM25 + 验证闸门），上下文可复现；
- 生成：真实 LLM 基于检索命中生成答案（上下文注入用户消息，与线上一致）；
- 评分：LLM-as-a-Judge 评估 faithfulness（忠实度，有无编造）与 answer_relevance（相关性），
  对 out_of_kb 负向用例额外判定「拒绝回答而非编造」（grounded=true 且说明资料不足才算通过）。

两种模式（scripts/eval_semantic.py 选择）：
- 默认（真实 LLM）：生成用 chat 场景、judge 用 rag_judge 场景，需配置 LLM_API_KEY；
- fake 模式：FakeChatModel 占位，离线冒烟验证「检索→生成→judge→报告」链路可跑通，
  分数为占位值、无评测意义（仅 CI/离线自检用）。
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.errors import ConfigError
from app.llm.client import get_chat_model
from app.llm.fake_model import FakeChatModel

from eval import runner

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


def _judge_prompt(query: str, answer: str, context: str, out_of_kb: bool) -> list:
    system = (
        "你是 RAG 语义评测判官。评估助手回答的忠实度与相关性，只输出一条 JSON，不要输出其他文字。\n"
        "- faithfulness（0~5）：回答的每个关键论断是否都能在资料中找到依据；数字/时限/主体/口径是否与资料一致；"
        "编造、篡改、张冠李戴应给低分。\n"
        "- answer_relevance（0~5）：回答是否直接切中问题，有无答非所问、绕圈子。\n"
        "- grounded（true/false）：回答中是否存在资料里没有的编造事实。"
        + ("注意：本问题是知识库外的问题，资料中没有对应条目——正确做法是明确拒绝/说明资料不足；"
           "若模型强行给出资料之外的答案即视为编造（grounded=false）。" if out_of_kb else "")
        + "\n输出格式："
        '{"faithfulness": 0-5, "answer_relevance": 0-5, "grounded": true/false, "reason": "一句话"}'
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


def _run_single(case: dict[str, Any], store, generator, judge, top_k: int) -> dict[str, Any]:
    """单个用例：确定性检索 → LLM 生成 → judge 评分。返回逐用例记录。"""
    expected = case["expected"]
    if not expected["retrieval_need"]:
        # 不检索分支不进入 L2 语义评分（生成无需上下文，暂不计分）
        return {"id": case["id"], "branch": case["branch"], "query": case["query"], "skipped": True}

    router = runner._RecordingRouter(runner._ExpectedRouter(runner._decision(expected)))
    scheme = runner._build_scheme(top_k, store, router)
    t0 = time.perf_counter()
    result = scheme.retrieve_full(case["query"], top_k, context=case.get("context"))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    hits = result.hits
    context = _format_context(hits)
    is_oob = bool(case.get("out_of_kb"))

    rec: dict[str, Any] = {
        "id": case["id"],
        "branch": case["branch"],
        "query": case["query"],
        "out_of_kb": is_oob,
        "retrieved_ids": sorted(
            {(h.get("metadata") or {}).get("chunk_id") for h in hits}
        ),
        "retrieved_count": len(hits),
        "answerable": bool(result.answerability.get("answerable")) if result.answerability else None,
        "elapsed_ms": round(elapsed_ms, 1),
        "answer": None,
        "faithfulness": None,
        "answer_relevance": None,
        "grounded": None,
        "judge_reason": None,
        "error": None,
    }

    # 生成：真实 LLM 或 fake 占位
    gen_start = time.perf_counter()
    try:
        resp = generator.invoke(_gen_prompt(case["query"], context))
        answer = resp.content if isinstance(resp.content, str) else str(resp.content or "")
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"generate 失败: {exc}"
        return rec
    rec["gen_elapsed_ms"] = round((time.perf_counter() - gen_start) * 1000, 1)
    rec["answer"] = answer

    # judge 评分
    try:
        resp = judge.invoke(_judge_prompt(case["query"], answer, context, is_oob))
        content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
        data = _extract_json(content)
        rec["faithfulness"] = round(max(0, min(5, float(data.get("faithfulness", 0)))), 2)
        rec["answer_relevance"] = round(max(0, min(5, float(data.get("answer_relevance", 0)))), 2)
        rec["grounded"] = bool(data.get("grounded", True))
        rec["judge_reason"] = str(data.get("reason", "") or "")
    except Exception as exc:  # noqa: BLE001 — 解析失败记为 error，不中断全量
        rec["error"] = f"judge 解析失败: {exc}"
    return rec


def run(top_k: int = 3, fake: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑 L2 语义评测（全部检索用例），返回 (逐用例记录, 汇总报告)。

    fake=True：生成/judge 用 FakeChatModel 占位，离线冒烟链路（分数无评测意义）；
    fake=False：真实 LLM（生成=chat、judge=rag_judge），未配 LLM_API_KEY 时抛 ConfigError。
    """
    store = runner._build_store()
    cases = runner._load_cases()

    if fake:
        n = len(cases)
        generator = FakeChatModel(script=[AIMessage(content="（fake 生成占位：依据资料作答）")] * n)
        judge = FakeChatModel(
            script=[
                AIMessage(
                    content='{"faithfulness": 3, "answer_relevance": 3, "grounded": true, "reason": "fake 冒烟占位"}'
                )
                for _ in range(n)
            ]
        )
    else:
        generator = get_chat_model("chat")
        judge = get_chat_model("rag_judge")
        if generator is None or judge is None:
            raise ConfigError(
                "L2 语义评测需要真实 LLM（生成场景 chat / judge 场景 rag_judge）：请配置 LLM_API_KEY。"
                "离线冒烟可用 --fake（仅验证链路，无评测意义）。"
            )

    records = [_run_single(c, store, generator, judge, top_k) for c in cases]
    scored = [r for r in records if not r.get("skipped") and r.get("error") is None]
    oob = [r for r in scored if r["out_of_kb"]]

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "top_k": top_k,
            "mode": "fake 冒烟（无评测意义）" if fake else "real LLM (chat 生成 + rag_judge 评分)",
            "retrieval": "bm25-bigram (确定性，非生产向量检索)",
            "corpus_size": len(runner.CORPUS),
            "eval_cases": len(records),
        },
        "avg_faithfulness": round(sum(r["faithfulness"] for r in scored if r["faithfulness"] is not None) / len(scored), 3) if scored else None,
        "avg_answer_relevance": round(sum(r["answer_relevance"] for r in scored if r["answer_relevance"] is not None) / len(scored), 3) if scored else None,
        "out_of_kb_grounded_rate": round(sum(1 for r in oob if r["grounded"]) / len(oob), 3) if oob else None,
        "scored_cases": len(scored),
        "cases": records,
    }
    return records, report


def save_report(report: dict[str, Any], path: str) -> None:
    """把 L2 报告写入 JSON（含逐用例答案与评分，供失败样本回流分析）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
