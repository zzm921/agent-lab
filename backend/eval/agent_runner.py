"""Agent 全量评测 · P1 确定性断言层：脚本化决策驱动真实 Harness，断言工具轨迹。

评测哲学与 eval/runner.py（modular RAG）一致——给定「决策脚本」下衡量编排链
（harness：工具循环 / 事件流 / 审批闸门 / 轮数上限 / 熔断）的执行质量，
隔离 LLM 语义质量（由后续 P2 LLM-as-a-Judge 层负责）：

- 脚本化决策：FakeChatModel 按 agent_case_set.jsonl 的 script 依次产出 tool_calls /
  最终回答，真实驱动 create_agent 的「思考-行动-观察」循环与中间件事件流；
- 轨迹断言：实际执行的工具序列 vs gold_tool_plan（顺序精确匹配 + 关键参数包含）、
  禁用工具零调用、高危工具（run_command）必须触发审批且不被自动执行、轮数上限收口；
- 全部离线确定性：不依赖 Key / 网络 / 真实 LLM，可直接接入 CI 回归门禁。

指标口径：
- sequence_match：执行序列 == 金标序列（顺序、工具名逐一相等）
- args_ok：每个金标步骤的 args_contains（键存在 + 值子串包含）全部命中；
  序列不匹配时参数校验无意义（记 None，不计入通过率）
- forbidden_hits：禁调工具被执行次数（必须为 0）
- 审批闸门：强制 HITL 工具被调用时产出 approval_request 且不执行
- 轮数上限：脚本超出 max_steps 时由 ModelCallLimit 收口，done 提示「已达到最大轮数上限」
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from app.agents.runner import AgentRunner
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.config import Settings
from app.llm.fake_model import FakeChatModel, FakeEmbeddings
from app.memory.session_store import SessionStore

# 评测集路径（相对本包）
_EVAL_SET_PATH = Path(__file__).resolve().parent / "agent_case_set.jsonl"


# ---------------------------------------------------------------------------
# 用例装载 / 脚本构造
# ---------------------------------------------------------------------------
def _load_cases() -> list[dict[str, Any]]:
    cases = []
    for line in _EVAL_SET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _script_messages(case: dict[str, Any]) -> list[BaseMessage]:
    """把用例 script（JSON 步骤）转成 FakeChatModel 的按次出牌消息队列。"""
    msgs = []
    for i, step in enumerate(case.get("script") or []):
        if step.get("type") == "tool_call":
            msgs.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": step["tool"],
                            "args": dict(step.get("args") or {}),
                            "id": f"eval_call_{i}",
                            "type": "tool_call",
                        }
                    ],
                )
            )
        else:  # final：无工具调用的收尾回答
            msgs.append(AIMessage(content=step.get("content", "（完成）")))
    return msgs


# ---------------------------------------------------------------------------
# 用例执行：真实 Harness 驱动
# ---------------------------------------------------------------------------
def _make_settings(case: dict[str, Any]) -> Settings:
    """离线配置：显式清空外部依赖（Qdrant/ES/MCP/Key），按用例覆盖 max_steps。"""
    return Settings(
        llm_api_key="eval",
        embedding_api_key="eval",
        mcp_servers="{}",
        qdrant_url="",
        qdrant_api_key="",
        es_url="",
        es_api_key="",
        es_username="",
        es_password="",
        rag_schemes=["naive"],
        rag_enabled=False,
        telemetry_enabled=False,
        memory_enabled=False,
        max_steps=int(case.get("max_steps", 8)),
    )


async def _execute_case(case: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    """跑单个用例：构建真实 runner（脚本化模型），收集全部 SSE 事件，返回 (事件, 耗时ms)。"""
    settings = _make_settings(case)
    sessions = SessionStore()
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), None, FakeEmbeddings())
    model = FakeChatModel(script=_script_messages(case))
    runner = AgentRunner(settings, model, registry, sessions)

    events: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    async for ev in runner.stream(
        session_id=f"eval-{case['id']}",
        message=case["query"],
        mode=case.get("mode", "react"),
        enabled=case["enabled"],
        prompt_strategy="standard",
        approval_policy=case.get("approval_policy", "never"),
        rag_enabled=False,
        memory_enabled=False,
        client_key="eval",
    ):
        events.append(ev)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return events, elapsed_ms


# ---------------------------------------------------------------------------
# 断言 / 汇总
# ---------------------------------------------------------------------------
def _evaluate(case: dict[str, Any], events: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
    """单用例指标：轨迹序列 / 参数包含 / 禁调 / 审批闸门 / 轮数收口。"""
    gold = case.get("gold_tool_plan") or []
    gold_tools = [g["tool"] for g in gold]

    executed = []
    for ev in events:
        if ev.get("type") == "tool_start":
            executed.append({"tool": ev.get("tool"), "args": ev.get("args") or {}})
    exec_tools = [e["tool"] for e in executed]

    # 序列精确匹配：工具名逐一相等且顺序一致
    sequence_match = exec_tools == gold_tools

    # 参数包含校验：每个金标步骤的 args_contains（键存在 + 值子串包含）全部命中
    args_ok: bool | None = True
    args_failures: list[dict[str, Any]] = []
    if sequence_match:
        for i, g in enumerate(gold):
            for k, v in (g.get("args_contains") or {}).items():
                got = executed[i]["args"].get(k)
                if got is None or (v and str(v) not in str(got)):
                    args_ok = False
                    args_failures.append({"step": i, "key": k, "expect": v, "got": got})
    else:
        args_ok = None  # 序列已失配，参数校验无意义

    forbidden = case.get("forbidden_tools") or []
    forbidden_hits = [t for t in exec_tools if t in forbidden]
    types = [ev.get("type") for ev in events]
    done = "done" in types
    done_limit = any(ev.get("type") == "done" and "上限" in str(ev.get("summary", "")) for ev in events)
    approval = "approval_request" in types
    error = "error" in types

    expect = case.get("expect") or {}
    if expect.get("approval_request"):
        # 高危工具：必须触发审批、不得被自动执行、不得报错
        passed = bool(approval) and not exec_tools and not error
    elif expect.get("done_limit"):
        passed = bool(done_limit) and sequence_match
    else:
        passed = bool(done) and not done_limit and not error and sequence_match and args_ok is True

    return {
        "id": case["id"],
        "branch": case["branch"],
        "query": case["query"],
        "mode": case.get("mode", "react"),
        "executed": exec_tools,
        "gold": gold_tools,
        "sequence_match": sequence_match,
        "args_ok": args_ok,
        "args_failures": args_failures,
        "forbidden_hits": forbidden_hits,
        "approval_requested": approval,
        "done": done,
        "done_limit": done_limit,
        "error": error,
        "elapsed_ms": round(elapsed_ms, 1),
        "passed": passed,
    }


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in records if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    branches: dict[str, dict[str, Any]] = {}
    for r in records:
        agg = branches.setdefault(r["branch"], {"cases": 0, "passed": 0})
        agg["cases"] += 1
        if r["passed"]:
            agg["passed"] += 1
    return {
        "cases": n,
        "passed": sum(1 for r in records if r["passed"]),
        "pass_rate": round(sum(1 for r in records if r["passed"]) / n, 3) if n else 0.0,
        "sequence_match_rate": _mean(records, "sequence_match"),
        "args_ok_rate": _mean(records, "args_ok"),
        "forbidden_hits": sum(len(r["forbidden_hits"]) for r in records),
        "approval_cases": sum(1 for r in records if r["approval_requested"]),
        "loop_guard_cases": sum(1 for r in records if r["done_limit"]),
        "error_cases": sum(1 for r in records if r["error"]),
        "avg_elapsed_ms": round(sum(r["elapsed_ms"] for r in records) / n, 1) if n else 0.0,
        "branches": branches,
    }


def run() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑全量 Agent 评测集，返回 (逐用例记录, 汇总报告)。全部离线确定性。"""
    records: list[dict[str, Any]] = []
    for case in _load_cases():
        events, elapsed_ms = asyncio.run(_execute_case(case))
        records.append(_evaluate(case, events, elapsed_ms))
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "P1 确定性断言（脚本化决策 + 真实 Harness 执行，无真实 LLM）",
            "modules": "create_agent + StreamEventsMiddleware + AgentHarness",
            "eval_cases": len(records),
        },
        "overall": _aggregate(records),
        "cases": records,
    }
    return records, report


def save_report(report: dict[str, Any], path: str) -> None:
    """把报告写入 JSON（含逐用例轨迹明细，供失败样本回流分析）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
