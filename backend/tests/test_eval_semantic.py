"""L2 语义评测链路冒烟回归（Phase A 评估闭环第 2 层）。

用 fake 模式离线验证「检索→生成→judge→报告」整条链路可跑通：
- 不依赖 LLM_API_KEY（FakeChatModel 占位）；
- 断言逐用例均产出答案与评分、无错误、报告字段完整；
- 分数为占位值（无评测意义），真实语义评测由 scripts/eval_semantic.py 在配置 Key 后运行。
"""
from __future__ import annotations

from eval import semantic


def test_l2_pipeline_smoke_runs() -> None:
    """fake 冒烟：全量检索用例都产出答案+评分，无 error。"""
    records, report = semantic.run(top_k=3, fake=True)

    scored = [r for r in records if not r.get("skipped")]
    assert len(scored) >= 20, f"检索用例数异常: {len(scored)}"
    for r in scored:
        assert r.get("error") is None, f"{r['id']} 出现错误: {r['error']}"
        assert r.get("answer"), f"{r['id']} 未产出答案"
        assert r["faithfulness"] is not None and r["answer_relevance"] is not None
        assert r["grounded"] is not None


def test_l2_report_fields_present() -> None:
    """报告包含三个汇总指标 + 逐用例明细。"""
    _, report = semantic.run(top_k=3, fake=True)
    for key in ("avg_faithfulness", "avg_answer_relevance", "out_of_kb_grounded_rate"):
        assert report[key] is not None, f"报告缺少指标 {key}"
    assert report["scored_cases"] >= 20
    assert len(report["cases"]) == report["meta"]["eval_cases"]


def test_l2_skips_no_retrieval_branch() -> None:
    """不检索分支（寒暄/致谢）不进入语义评分。"""
    records, _ = semantic.run(top_k=3, fake=True)
    skipped = [r for r in records if r.get("skipped")]
    assert skipped, "评测集应包含不检索分支"
    for r in skipped:
        assert r["branch"] == "no_retrieval"
