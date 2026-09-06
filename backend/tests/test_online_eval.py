"""在线评估闭环（P0）测试：样本采集落盘、反馈回填、回流评测集、/api/feedback 端点。

覆盖四层：
- 采集：append_online_sample 仅对检索命中落盘，字段完整，失败静默；
- 回填：apply_feedback 按 session_id+query 匹配、幂等更新，未匹配记 orphan；
- 回流：scripts/eval_online.py 分层筛选（点踩必回流 / 关键分支全量 / 兜底抽样 / 去重 / limit 裁剪）；
- 端点：POST /api/feedback 命中回填与 eval_online_enabled 关闭时返回 disabled。
全部离线确定性，使用 tmp 目录隔离落盘，不依赖 Key 与网络。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.telemetry import sample as sample_mod


BACKEND = Path(__file__).resolve().parents[1]


def _load_eval_online():
    """按文件路径加载 scripts/eval_online.py（scripts 非包，避免污染 sys.path）。"""
    spec = importlib.util.spec_from_file_location(
        "_eval_online_under_test", str(BACKEND / "scripts" / "eval_online.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sample_dir(tmp_path, monkeypatch):
    """把在线采样目录与回流输出重定向到 tmp，隔离真实 eval/ 目录。"""
    from app.config import settings

    monkeypatch.setattr(settings, "eval_online_enabled", True)
    monkeypatch.setattr(settings, "eval_sample_dir", str(tmp_path / "samples"))
    monkeypatch.setattr(settings, "eval_online_set_path", str(tmp_path / "online_eval_set.jsonl"))
    monkeypatch.setattr(settings, "eval_online_agent_set_path", str(tmp_path / "online_agent_eval_set.jsonl"))
    monkeypatch.setattr(settings, "eval_online_fix_set_path", str(tmp_path / "online_fix_set.jsonl"))
    d = tmp_path / "samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 采集：append_online_sample
# ---------------------------------------------------------------------------
def _meta(**kw) -> dict:
    base = {"message": "什么是多跳检索？", "mode": "react", "rag_scheme": "naive", "status": "done"}
    base.update(kw)
    return base


def _rag_signals(**kw) -> dict:
    base = {
        "effective_message": "什么是多跳检索？",
        "generation_mode": "citation",
        "complexity": "multihop",
        "retrieval_mode": "hybrid",
        "insufficient": False,
        "retrieved_ids": ["d1", "d2"],
    }
    base.update(kw)
    return base


def test_append_sample_with_hits_lands_row(sample_dir):
    """有检索命中 → 落盘一行，字段完整（含版本号，供报告溯源），capability=rag。"""
    events = [
        {"type": "meta", "session_id": "s1"},
        {"type": "retrieve", "hits": [{"id": "d1"}, {"id": "d2"}]},
        {"type": "message", "delta": "多跳检索是指……"},
        {"type": "done", "summary": "完成"},
    ]
    sample_mod.append_online_sample("s1", "cid:a", _meta(), events, _rag_signals())

    files = list(sample_dir.glob("online_*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text(encoding="utf-8"))
    assert row["session_id"] == "s1"
    assert row["client_key"] == "cid:a"
    assert row["query"] == "什么是多跳检索？"
    assert row["complexity"] == "multihop"
    assert row["retrieved_ids"] == ["d1", "d2"]
    assert row["capability"] == "rag"
    assert row["vote"] is None and row["reason"] is None
    assert row["version"], "样本必须带系统版本号"
    assert row["answer"] == "多跳检索是指……"


def test_append_sample_agent_with_tools_lands_row(sample_dir):
    """无检索命中但有工具调用（纯 agent）→ 落盘 capability=agent，记录工具轨迹。"""
    events = [
        {"type": "tool_end", "tool": "calculator", "success": True, "result": "5"},
        {"type": "tool_end", "tool": "calculator", "success": True, "result": "6"},  # 重复调用去重
        {"type": "tool_end", "tool": "search_web", "success": False, "result": "err"},  # 失败不计入
        {"type": "message", "delta": "结果是 6"},
        {"type": "done"},
    ]
    sample_mod.append_online_sample("s1", "cid:a", _meta(message="2+3?"), events, _rag_signals(retrieved_ids=[]))

    files = list(sample_dir.glob("online_*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text(encoding="utf-8"))
    assert row["capability"] == "agent"
    assert row["retrieved_ids"] == []
    assert row["tool_sequence"] == ["calculator"]  # 保序去重、失败不计入


def test_append_sample_skips_without_hits_and_tools(sample_dir):
    """无检索命中且无工具调用（寒暄 / 未配 Key / rag 未启用且未走工具）→ 不落样本。"""
    sample_mod.append_online_sample("s1", "cid:a", _meta(message="你好"), [], _rag_signals(retrieved_ids=[]))
    assert list(sample_dir.glob("online_*.jsonl")) == []


def test_append_sample_silent_on_error(sample_dir, monkeypatch):
    """采样失败必须静默：损坏的落盘路径绝不影响主流程。"""
    monkeypatch.setattr(sample_mod, "_sample_path", lambda: Path("z:/invalid/dir/online.jsonl"))
    sample_mod.append_online_sample("s1", "cid:a", _meta(), [], _rag_signals())  # 不应抛异常


# ---------------------------------------------------------------------------
# 回填：apply_feedback / append_orphan_feedback
# ---------------------------------------------------------------------------
def _seed_sample(sample_dir, session_id="s1", query="q1"):
    sample_mod.append_online_sample(session_id, "cid:a", _meta(message=query), [], _rag_signals())


def test_feedback_matches_and_updates(sample_dir):
    _seed_sample(sample_dir)
    matched, updated = sample_mod.apply_feedback("s1", "q1", "down", "回答有误")
    assert (matched, updated) == (True, True)

    rows = sample_mod._load_today_rows()
    assert rows[0]["vote"] == "down"
    assert rows[0]["reason"] == "回答有误"


def test_feedback_idempotent_repeat(sample_dir):
    """幂等：相同内容重复提交不触发二次写入。"""
    _seed_sample(sample_dir)
    sample_mod.apply_feedback("s1", "q1", "up", "")
    matched, updated = sample_mod.apply_feedback("s1", "q1", "up", "")
    assert (matched, updated) == (True, False)


def test_feedback_unmatched_lands_orphan(sample_dir):
    """未匹配（样本清理 / 未采集 / 篡改）→ apply_feedback 返回 False，端点转写 orphan 供排查。"""
    matched, updated = sample_mod.apply_feedback("ghost", "不存在的提问", "down", "")
    assert (matched, updated) == (False, False)

    # 端点逻辑：未匹配时调用 append_orphan_feedback 落盘
    sample_mod.append_orphan_feedback("ghost", "不存在的提问", "down", "")
    orphan = sample_dir / "orphan_feedback.jsonl"
    assert orphan.exists()
    row = json.loads(orphan.read_text(encoding="utf-8"))
    assert row["session_id"] == "ghost"
    assert row["vote"] == "down"


# ---------------------------------------------------------------------------
# 回流：scripts/eval_online.py
# ---------------------------------------------------------------------------
def test_branch_of_classification():
    mod = _load_eval_online()
    assert mod._branch_of({"insufficient": True, "retrieved_ids": ["d1"]}) == "out_of_kb"
    assert mod._branch_of({"insufficient": False, "retrieved_ids": []}) == "out_of_kb"
    assert mod._branch_of({"insufficient": False, "retrieved_ids": ["d1"], "complexity": "multihop"}) == "multihop"
    assert mod._branch_of({"insufficient": False, "retrieved_ids": ["d1"], "complexity": "decompose"}) == "decompose"
    assert mod._branch_of({"insufficient": False, "retrieved_ids": ["d1"], "complexity": "simple"}) == "simple"


def test_build_case_structure():
    mod = _load_eval_online()
    row = {
        "sample_id": "abc",
        "query": "公司今年营收？",
        "retrieval_mode": "hybrid",
        "complexity": "multihop",
        "generation_mode": "citation",
        "retrieved_ids": ["d9"],
        "vote": "down",
        "reason": "不全",
        "ts": "2026-09-06T10:00:00+00:00",
        "version": "dev",
    }
    case = mod._build_case(7, row)
    assert case["id"] == "on_0007"
    assert case["branch"] == "multihop"
    assert case["query"] == "公司今年营收？"
    assert case["expected"] == {
        "retrieval_need": True,
        "retrieval_mode": "hybrid",
        "complexity": "multihop",
        "generation_mode": "citation",
    }
    assert case["relevant"] == ["d9"]
    assert case["answer_keywords"] == [] and case["reference"] == ""
    assert case["origin"] == {
        "sample_id": "abc",
        "vote": "down",
        "reason": "不全",
        "ts": "2026-09-06T10:00:00+00:00",
        "version": "dev",
    }


def test_build_agent_case_structure():
    """Agent 样本 → 行为回归用例：mode + must_call（线上真实轨迹），金标留空待人工补。"""
    mod = _load_eval_online()
    row = {
        "sample_id": "abc",
        "query": "帮我算 23*17",
        "mode": "react",
        "tool_sequence": ["calculator"],
        "vote": "up",
        "reason": None,
        "ts": "2026-09-06T10:00:00+00:00",
        "version": "dev",
    }
    case = mod._build_agent_case(3, row)
    assert case["id"] == "agent_on_0003"
    assert case["capability"] == "agent"
    assert case["mode"] == "react"
    assert case["branch"] == "normal"
    assert case["query"] == "帮我算 23*17"
    assert case["must_call"] == ["calculator"]
    assert case["forbidden_tools"] == []
    assert case["answer_keywords"] == [] and case["reference"] == ""
    assert case["origin"]["sample_id"] == "abc"


def test_build_fix_case_structure():
    """点踩样本 → 修复池用例：金标不沿用线上行为（置空待人工修正），observed 记录观测。"""
    mod = _load_eval_online()
    rag_row = {
        "sample_id": "f1",
        "query": "公司去年营收？",
        "capability": "rag",
        "retrieved_ids": ["d9"],
        "complexity": "simple",
        "retrieval_mode": "hybrid",
        "generation_mode": "direct",
        "answer": "营收 1 亿",
        "vote": "down",
        "reason": "答非所问",
        "ts": "2026-09-06T10:00:00+00:00",
        "version": "dev",
    }
    case = mod._build_fix_case(2, rag_row)
    assert case["id"] == "fix_0002"
    assert case["capability"] == "rag" and case["label"] == "needs_fix"
    assert case["query"] == "公司去年营收？" and case["reason"] == "答非所问"
    assert case["expected"] is None, "点踩样本的 expected 金标不沿用线上路由（可能本身判错）"
    assert case["observed"]["retrieved_ids"] == ["d9"], "线上观测行为保留供人工参照"
    assert case["observed"]["answer"] == "营收 1 亿"
    assert case["answer_keywords"] == [] and case["reference"] == ""
    assert case["origin"]["sample_id"] == "f1"


def _write_sample(sample_dir, name, query, **kw):
    row = {
        "sample_id": name,
        "session_id": "s",
        "client_key": "cid",
        "ts": "2026-09-06T10:00:00+00:00",
        "version": "dev",
        "query": query,
        "effective_query": query,
        "capability": kw.get("capability"),
        "tool_sequence": kw.get("tool_sequence", []),
        "mode": kw.get("mode", "react"),
        "rag_scheme": "naive",
        "rag_enabled": True,
        "complexity": kw.get("complexity", "simple"),
        "retrieval_mode": "hybrid",
        "generation_mode": "citation",
        "insufficient": kw.get("insufficient", False),
        "retrieved_ids": kw.get("retrieved_ids", ["d1"]),
        "answer": "",
        "elapsed_ms": 100,
        "status": "done",
        "stats": {},
        "vote": kw.get("vote"),
        "reason": kw.get("reason"),
    }
    path = sample_dir / f"online_{kw.get('date', '20260906')}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_main(monkeypatch, args):
    mod = _load_eval_online()
    monkeypatch.setattr(sys, "argv", ["eval_online", *args])
    mod.main()


def test_online_flow_down_and_critical_full(sample_dir, monkeypatch):
    """点踩进修复池 + 关键分支全量回流：不被 ratio 抽样削减；点踩不污染回归池。"""
    _write_sample(sample_dir, "a", "普通问题")                        # fallback
    _write_sample(sample_dir, "b", "关键多跳", complexity="multihop")  # critical
    _write_sample(sample_dir, "c", "点踩问题", vote="down")            # down → 修复池
    rag_out = sample_dir.parent / "set.jsonl"
    fix_out = sample_dir.parent / "fix.jsonl"
    _run_main(monkeypatch, ["--ratio", "0.0", "--out", str(rag_out), "--out-fix", str(fix_out), "--seed", "42"])

    cases = [json.loads(l) for l in rag_out.read_text(encoding="utf-8").splitlines()]
    ids = {c["query"] for c in cases}
    assert "关键多跳" in ids, "关键分支必须全量回流（ratio=0 也不应被裁剪）"
    assert "点踩问题" not in ids, "点踩样本金标不可信，不得进入回归池"

    fix_cases = [json.loads(l) for l in fix_out.read_text(encoding="utf-8").splitlines()]
    assert [c["query"] for c in fix_cases] == ["点踩问题"], "点踩样本全部移交修复池"


def test_online_flow_dedup_prefers_down_to_fix(sample_dir, monkeypatch):
    """同 query 去重保留点踩 → 点踩移交修复池，回归池无该 query（金标不可信不沿用）。"""
    _write_sample(sample_dir, "x1", "同一问题")
    _write_sample(sample_dir, "x2", "同一问题", vote="down")
    rag_out = sample_dir.parent / "set.jsonl"
    fix_out = sample_dir.parent / "fix.jsonl"
    _run_main(monkeypatch, ["--ratio", "1.0", "--out", str(rag_out), "--out-fix", str(fix_out), "--seed", "42"])

    assert not rag_out.exists(), "唯一样本被点踩 → 回归池无可信金标，不创建文件"
    fix_cases = [json.loads(l) for l in fix_out.read_text(encoding="utf-8").splitlines()]
    assert len(fix_cases) == 1
    assert fix_cases[0]["origin"]["sample_id"] == "x2"
    assert fix_cases[0]["label"] == "needs_fix"


def test_online_flow_limit_keeps_priority(sample_dir, monkeypatch):
    """--limit 总量上限：回归池优先保留关键分支，点踩独立修复池不受抽样影响。"""
    for i in range(5):
        _write_sample(sample_dir, f"f{i}", f"兜底{i}")
    _write_sample(sample_dir, "c", "点踩问题", vote="down")
    rag_out = sample_dir.parent / "set.jsonl"
    fix_out = sample_dir.parent / "fix.jsonl"
    _run_main(monkeypatch, ["--ratio", "1.0", "--limit", "2", "--out", str(rag_out), "--out-fix", str(fix_out), "--seed", "42"])

    cases = [json.loads(l) for l in rag_out.read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 2, "回归池受 limit 封顶（5 条兜底裁剪到 2）"

    fix_cases = [json.loads(l) for l in fix_out.read_text(encoding="utf-8").splitlines()]
    assert [c["origin"]["sample_id"] for c in fix_cases] == ["c"], "点踩样本始终全量进修复池"


def test_online_flow_dry_run_writes_nothing(sample_dir, monkeypatch):
    _write_sample(sample_dir, "a", "普通问题")
    out = sample_dir.parent / "set.jsonl"
    _run_main(monkeypatch, ["--ratio", "1.0", "--dry-run", "--out", str(out), "--seed", "42"])
    assert not out.exists(), "--dry-run 不应写入评测集"


def test_online_flow_dual_domain_split(sample_dir, monkeypatch):
    """RAG 与 Agent 两域分流 + 双池：回归池各自文件；点踩进修复池，金标不沿用线上行为。"""
    _write_sample(sample_dir, "r1", "知识库问题", complexity="multihop")  # rag 域
    _write_sample(
        sample_dir,
        "a1",
        "帮我算 23*17",
        capability="agent",
        retrieved_ids=[],
        tool_sequence=["calculator"],
        vote="down",  # 点踩 agent 样本 → 修复池
    )
    _write_sample(
        sample_dir,
        "a2",
        "帮我查天气",
        capability="agent",
        retrieved_ids=[],
        tool_sequence=["search_weather"],
    )
    rag_out = sample_dir.parent / "rag_set.jsonl"
    agent_out = sample_dir.parent / "agent_set.jsonl"
    fix_out = sample_dir.parent / "fix_set.jsonl"
    _run_main(
        monkeypatch,
        ["--ratio", "1.0", "--out", str(rag_out), "--out-agent", str(agent_out), "--out-fix", str(fix_out), "--seed", "42"],
    )

    rag_cases = [json.loads(l) for l in rag_out.read_text(encoding="utf-8").splitlines()]
    agent_cases = [json.loads(l) for l in agent_out.read_text(encoding="utf-8").splitlines()]
    fix_cases = [json.loads(l) for l in fix_out.read_text(encoding="utf-8").splitlines()]

    assert [c["id"] for c in rag_cases] == ["on_0001"], "RAG 回归池只含 rag 样本"
    assert rag_cases[0]["branch"] == "multihop" and "expected" in rag_cases[0]

    assert len(agent_cases) == 1, "点踩样本不进 Agent 回归池，仅 a2 回流"
    assert agent_cases[0]["query"] == "帮我查天气"
    assert agent_cases[0]["must_call"] == ["search_weather"]
    assert agent_cases[0]["mode"] == "react" and agent_cases[0]["branch"] == "normal"

    assert len(fix_cases) == 1, "点踩 agent 样本进修复池"
    fix = fix_cases[0]
    assert fix["capability"] == "agent" and fix["label"] == "needs_fix"
    assert fix["origin"]["sample_id"] == "a1" and fix["reason"] is None
    assert fix["must_call"] is None, "点踩样本金标不沿用线上行为（待人工修正）"
    assert fix["observed"]["tool_sequence"] == ["calculator"], "线上观测行为记入 observed 供人工参照"


# ---------------------------------------------------------------------------
# 端点：POST /api/feedback
# ---------------------------------------------------------------------------
def test_feedback_api_disabled(monkeypatch):
    """eval_online_enabled 关闭 → 返回 disabled，不落 orphan（避免噪音）。"""
    from app.config import settings

    monkeypatch.setattr(settings, "eval_online_enabled", False)
    client = TestClient(app)
    resp = client.post(
        "/api/feedback",
        json={"session_id": "s1", "query": "q1", "vote": "down", "reason": ""},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"ok": True, "matched": False, "updated": False, "disabled": True}


def test_feedback_api_matches_sample(sample_dir, monkeypatch):
    """命中样本 → 回填 vote，正常响应 matched/updated。"""
    _seed_sample(sample_dir, "s1", "q1")
    client = TestClient(app)
    resp = client.post(
        "/api/feedback",
        json={"session_id": "s1", "query": "q1", "vote": "down", "reason": "不全"},
    )
    assert resp.json() == {"ok": True, "matched": True, "updated": True}
    rows = sample_mod._load_today_rows()
    assert rows[0]["vote"] == "down"


def test_feedback_api_unmatched_orphan(sample_dir):
    """未命中样本 → 匹配失败并落 orphan，供排查未采集/篡改。"""
    client = TestClient(app)
    resp = client.post(
        "/api/feedback",
        json={"session_id": "ghost", "query": "不存在", "vote": "up", "reason": ""},
    )
    assert resp.json() == {"ok": True, "matched": False, "updated": False}
    assert (sample_dir / "orphan_feedback.jsonl").exists()
