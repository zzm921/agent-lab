"""全面评测链路冒烟回归（路由/检索/生成/拒答四维统一报告）。

用 fake 模式离线验证「确定性检索 → LLM 生成 → RAGAS evaluate / 拒答 judge → 报告」：
- 不依赖 Key（FakeChatModel / FakeEmbeddings 占位）；
- 断言逐用例均产出答案、链路无异常、报告结构完整；
- fake 模式下 RAGAS 结构化输出解析失败属预期（占位文本非合法 JSON），
  分数可能为 None、无评测意义；真实评分由 scripts/eval_full.py 在配置 Key 后运行。
"""
from __future__ import annotations

from eval import full, runner


def test_full_pipeline_smoke_runs() -> None:
    """fake 冒烟：全量检索用例都产出答案，链路无异常。"""
    records, _ = full.run(top_k=3, fake=True)

    scored = [r for r in records if not r.get("skipped")]
    assert len(scored) >= 22, f"检索用例数异常: {len(scored)}"
    for r in scored:
        assert r.get("error") is None, f"{r['id']} 出现错误: {r['error']}"
        assert r.get("answer"), f"{r['id']} 未产出答案"
        assert "retrieval" in r, f"{r['id']} 未合并检索指标"


def test_full_report_structure() -> None:
    """报告包含路由/检索/生成/拒答四维汇总与逐用例明细。"""
    _, report = full.run(top_k=3, fake=True)
    for key in ("meta", "routing", "retrieval", "generation", "out_of_kb", "cases"):
        assert key in report, f"报告缺少 {key}"
    assert report["routing"]["routing_accuracy"] == 1.0
    assert report["retrieval"]["overall"]["cases"] == 26
    assert report["out_of_kb"]["grounded_rate"] is not None
    for m in ("faithfulness", "answer_relevancy", "answer_correctness", "context_precision"):
        assert f"avg_{m}" in report["generation"], f"生成汇总缺少 {m}"
    assert len(report["cases"]) == report["meta"]["eval_cases"]


def test_reference_present_for_ragas_cases() -> None:
    """常规检索用例必须携带金标 reference（RAGAS 正确性/上下文精确度依赖）。"""
    cases = runner._load_cases()
    ragas_cases = [c for c in cases if c["expected"]["retrieval_need"] and not c.get("out_of_kb")]
    assert ragas_cases, "评测集应包含常规检索用例"
    for c in ragas_cases:
        assert c.get("reference"), f"{c['id']} 缺少金标 reference"
