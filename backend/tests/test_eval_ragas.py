"""L3 RAGAS 语义评测链路冒烟回归（独立层，可对比 L2 手写 judge）。

用 fake 模式离线验证「确定性检索 → LLM 生成 → RAGAS evaluate → 报告」整条链路：
- 不依赖 Key（FakeChatModel / FakeEmbeddings 占位）；
- 断言逐用例均产出答案、链路无异常、报告字段完整；
- fake 模式下 RAGAS 结构化输出解析失败属预期（占位文本非合法 JSON），
  分数可能为 None、无评测意义；真实评分由 scripts/eval_ragas.py 在配置 Key 后运行。
"""
from __future__ import annotations

from eval import ragas_eval


def test_l3_pipeline_smoke_runs() -> None:
    """fake 冒烟：全量检索用例都产出答案，链路无异常。"""
    records, _ = ragas_eval.run(top_k=3, fake=True)

    scored = [r for r in records if not r.get("skipped")]
    assert len(scored) >= 20, f"检索用例数异常: {len(scored)}"
    for r in scored:
        assert r.get("error") is None, f"{r['id']} 出现错误: {r['error']}"
        assert r.get("answer"), f"{r['id']} 未产出答案"


def test_l3_report_fields_present() -> None:
    """报告包含 meta/计分用例/逐用例明细。"""
    _, report = ragas_eval.run(top_k=3, fake=True)
    assert report["meta"]["metrics"] == ["faithfulness", "answer_relevancy"]
    assert report["scored_cases"] >= 20
    assert len(report["cases"]) == report["meta"]["eval_cases"]


def test_l3_skips_no_retrieval_branch() -> None:
    """不检索分支（寒暄/致谢）不进入 RAGAS 语义评分。"""
    records, _ = ragas_eval.run(top_k=3, fake=True)
    skipped = [r for r in records if r.get("skipped")]
    assert skipped, "评测集应包含不检索分支"
    for r in skipped:
        assert r["branch"] == "no_retrieval"
