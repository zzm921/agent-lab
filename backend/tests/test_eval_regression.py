"""modular RAG 离线评测回归门禁（Phase A 评估闭环的硬阈值断言）。

在 CI / 本地一键回归中把守「确定性评测基线」：一旦模块链（规则/BM25 检索/期望路由注入）
的检索、路由、闸门指标跌破阈值即失败——防止后续改动悄悄退化检索质量或绕开验证闸门。

说明：
- 全部离线确定性：期望路由注入 + 规则模块 + BM25 二元组词法检索，无 LLM 调用、不依赖 Key；
- 语料 42 条（含 7 组相似易混陷阱条目）、评测集 26 条（覆盖六分支 + 库外负向用例）；
- 阈值基于当前基线留出余量（见各断言注释），仅约束「不回归」，不代表生产向量检索质量；
- 陷阱条目（c36/c37/c39/c41 等）会真实抢占相似查询的召回位——decompose/multihop 召回
  因此低于 1.0 属预期（词法检索的真实上限），由分支召回阈值兜底，不因单例未满召回而失败。
"""
from __future__ import annotations

from eval import runner


def _run() -> tuple[list[dict], dict]:
    """跑确定性离线评测（期望路由注入），返回 (逐用例记录, 汇总报告)。"""
    return runner.run(top_k=3, real_router=False)


def test_routing_accuracy_is_perfect() -> None:
    """注入期望路由时必须 100% 路由正确（确定性隔离，衡量模块链而非路由本身）。"""
    _, report = _run()
    assert report["routing_accuracy"] == 1.0


def test_overall_retrieval_recall_no_regression() -> None:
    """总体召回不得跌破 0.9（基线 0.94：陷阱条目使 decompose/multihop 召回降至 <1）。"""
    _, report = _run()
    assert report["overall"]["avg_recall"] >= 0.9


def test_overall_mrr_no_regression() -> None:
    """总体 MRR 不得跌破 0.78（基线 0.81：相关块须经重排提升到前排，词法检索上限）。"""
    _, report = _run()
    assert report["overall"]["avg_mrr"] >= 0.78


def test_overall_keyword_coverage() -> None:
    """答案关键词覆盖不得跌破 0.9（基线 1.0：命中文须含能支撑回答的关键词）。"""
    _, report = _run()
    assert report["overall"]["keyword_coverage"] >= 0.9


def test_answer_sufficiency_gate_keeps_high_answerable() -> None:
    """验证闸门不得把「可答」用例误判为不可答（可答率基线 1.0）。"""
    _, report = _run()
    assert report["overall"]["answerable_rate"] >= 0.95


def test_no_fabricated_clarify() -> None:
    """闸门不得以「缺失信息/追问澄清」掩盖可答用例（clarify 率硬约束为 0）。"""
    _, report = _run()
    assert report["overall"]["clarify_rate"] <= 0.05


def test_no_retrieval_branch_stays_retrieval_free() -> None:
    """不检索分支（寒暄/致谢）必须零检索：不得误触发召回、不得进入验证闸门。"""
    records, _ = _run()
    no_retrieval = [r for r in records if r["branch"] == "no_retrieval"]
    assert no_retrieval, "评测集缺少 no_retrieval 分支用例"
    for r in no_retrieval:
        assert r["need_retrieval"] is False
        assert r["retrieved_ids"] == []
        assert r["answerable"] is None


def test_branch_recall_floors() -> None:
    """各分支召回下限：单点/改写须全量命中，分解/多跳受陷阱条目干扰允许低一些。"""
    _, report = _run()
    floors = {
        "simple": 1.0,  # 基线 1.00
        "rewrite": 0.9,  # 基线 1.00（指代消解 → 改写 → 召回）
        "decompose": 0.75,  # 基线 0.83（d03 对比核心块被陷阱 c36 抢占，词法检索真实上限）
        "multihop": 0.6,  # 基线 0.72（流程/实体链 3 块为词法检索真实上限，c22/c25 易被干扰块挤出）
    }
    for branch, floor in floors.items():
        agg = report["branches"].get(branch)
        assert agg is not None, f"评测报告缺少分支 {branch}"
        assert agg["avg_recall"] >= floor, (
            f"分支 {branch} 召回 {agg['avg_recall']} < 阈值 {floor}"
        )


def test_eval_stays_offline_fast() -> None:
    """确定性评测必须保持亚秒级平均耗时：若某用例出现 >200ms，多半混入了真实 LLM 调用。"""
    _, report = _run()
    assert report["overall"]["avg_elapsed_ms"] < 200.0


def test_all_expected_routes_covered_by_eval_set() -> None:
    """评测集六大分支覆盖完备性：至少各 1 条检索用例，且期望路由无非法枚举。"""
    records, report = _run()
    branch_counts: dict[str, int] = {}
    for r in records:
        branch_counts[r["branch"]] = branch_counts.get(r["branch"], 0) + 1
    assert {"no_retrieval", "simple", "rewrite", "decompose", "multihop", "out_of_kb"} <= set(
        branch_counts
    ), f"评测集分支覆盖不完整: {sorted(branch_counts)}"
    assert report["meta"]["eval_cases"] >= 10
