"""Agent 全量评测 · 任务/约束分离：脚本化决策驱动真实 Harness，按任务层 + 架构层双断言。

评测哲学与 eval/rag/runner.py（modular RAG）一致——给定「决策脚本」下衡量编排链
（harness：工具循环 / 事件流 / 审批闸门 / 轮数上限 / 熔断）的执行质量，
隔离 LLM 语义质量（由后续 P2 LLM-as-a-Judge 层负责）。

金标结构采用企业级「分层」：
- 任务层（task_set.jsonl，agent 无关）：
    must_call / must_not_call / forbidden_tools / expect —— 所有架构统一校验；
    sequence + args —— react 可选精确轨迹约束（线性循环下轨迹即契约）；
- 架构层（arch_specs.py，agent 特定）：
    plan_execute 计划协议、reflection 评审协议、multi_agent 委派协议 —— 架构不变量；
- 驱动：FakeChatModel 按 script 依次出牌（bind_tools 返回自身、全链共享队列），
  真实驱动 create_agent / StateGraph 的循环与中间件事件流；
- 全部离线确定性：不依赖 Key / 网络 / 真实 LLM，可直接接入 CI 回归门禁。
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from app.agents.runner import AgentRunner
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry
from app.config import Settings
from app.core.errors import ConfigError
from app.core.version import get_app_version
from app.llm.client import get_chat_model
from app.llm.fake_model import FakeChatModel, FakeEmbeddings
from app.memory.session_store import SessionStore

from eval.agent.arch_specs import check_arch_spec

# 评测集路径（相对本包）
_EVAL_SET_PATH = Path(__file__).resolve().parent / "task_set.jsonl"


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
# 成本计数包装：统计模型调用次数与 token 用量（真实/假模型共用）
# ---------------------------------------------------------------------------
class _CountingModel(BaseChatModel):
    """评测计数包装：委托内层模型，同时累计模型调用次数与 token 用量。

    - bind_tools/bind：委托内层绑定并返回计数 bound（BaseChatModel 默认实现会抛
      NotImplementedError，必须显式委托）；绑定后的 astream/ainvoke 等走 _CountingBound 计数；
    - 未绑定的 invoke/ainvoke 走 _generate/_agenerate 计数，stream/astream 直接覆盖计数；
    - token 从响应 usage_metadata 记账（DashScope 在消息/末块上写入 input/output_tokens；
      fake 无 usage → 0）；
    - 委托内层，不改变模型行为，仅作观测。
    """

    model_name: str = "eval-counting"

    def __init__(self, inner: BaseChatModel, stats: dict[str, int]) -> None:
        super().__init__()
        self._inner = inner
        self._stats = stats

    @property
    def _llm_type(self) -> str:
        return "eval-counting"

    @staticmethod
    def _record_usage(stats: dict[str, int], msg: Any) -> None:
        usage = getattr(msg, "usage_metadata", None) or {}
        stats["prompt_tokens"] += int(usage.get("input_tokens") or 0)
        stats["completion_tokens"] += int(usage.get("output_tokens") or 0)

    def bind_tools(self, tools, **kwargs):
        return _CountingBound(self._inner.bind_tools(tools, **kwargs), self._stats)

    def bind(self, **kwargs):
        return _CountingBound(self._inner.bind(**kwargs), self._stats)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._stats["calls"] += 1
        result = self._inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        self._record_usage(self._stats, result.generations[0].message)
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._stats["calls"] += 1
        result = await self._inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        self._record_usage(self._stats, result.generations[0].message)
        return result

    def stream(self, *args, **kwargs):
        self._stats["calls"] += 1
        return self._inner.stream(*args, **kwargs)

    async def astream(self, *args, **kwargs):
        self._stats["calls"] += 1
        last = None
        async for chunk in self._inner.astream(*args, **kwargs):
            last = chunk
            yield chunk
        if last is not None:
            self._record_usage(self._stats, last)


class _CountingBound:
    """绑定后的计数包装：转发内层 bound 的调用并计数（stream_model_call 主路径）。

    其余属性/方法（tool_calls 等）经 __getattr__ 透传内层。
    """

    def __init__(self, inner: Any, stats: dict[str, int]) -> None:
        self._inner = inner
        self._stats = stats

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def astream(self, *args, **kwargs):
        self._stats["calls"] += 1
        last = None
        async for chunk in self._inner.astream(*args, **kwargs):
            last = chunk
            yield chunk
        if last is not None:
            _CountingModel._record_usage(self._stats, last)

    def stream(self, *args, **kwargs):
        self._stats["calls"] += 1
        return self._inner.stream(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        self._stats["calls"] += 1
        result = await self._inner.ainvoke(*args, **kwargs)
        _CountingModel._record_usage(self._stats, result)
        return result

    def invoke(self, *args, **kwargs):
        self._stats["calls"] += 1
        result = self._inner.invoke(*args, **kwargs)
        _CountingModel._record_usage(self._stats, result)
        return result


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


async def _execute_case(case: dict[str, Any], real: bool = False) -> tuple[list[dict[str, Any]], float, dict[str, int]]:
    """跑单个用例：构建真实 runner（脚本化模型），收集全部 SSE 事件。

    返回 (事件流, 耗时ms, 成本统计{calls, prompt_tokens, completion_tokens})。
    real=True：用真实 LLM（chat 场景）替换剧本模型，用于鲁棒性/成功率评测（需 Key）。
    """
    settings = _make_settings(case)
    sessions = SessionStore()
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), None, FakeEmbeddings())
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if real:
        inner = get_chat_model("chat")
        if inner is None:
            raise ConfigError("真实 LLM 驱动评测需要配置 LLM_API_KEY（--real）。离线评测用默认 fake 模式。")
        model = _CountingModel(inner, stats)
    else:
        model = _CountingModel(FakeChatModel(script=_script_messages(case)), stats)
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
    return events, elapsed_ms, stats


# ---------------------------------------------------------------------------
# 断言 / 汇总
# ---------------------------------------------------------------------------
def _evaluate(case: dict[str, Any], events: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
    """单用例指标：任务层（工具约束/轨迹/审批/轮数）+ 架构层（L2 不变量）。"""
    executed = []
    for ev in events:
        if ev.get("type") == "tool_start":
            executed.append({"tool": ev.get("tool"), "args": ev.get("args") or {}})
    exec_tools = [e["tool"] for e in executed]

    # —— 任务层：通用工具约束（agent 无关）——
    must_call = case.get("must_call") or []
    must_not_call = case.get("must_not_call") or []
    forbidden = case.get("forbidden_tools") or []
    must_ok = set(must_call) <= set(exec_tools)
    must_not_hits = [t for t in exec_tools if t in must_not_call]
    forbidden_hits = [t for t in exec_tools if t in forbidden]

    # —— 任务层：react 可选精确轨迹（sequence + args 位置对齐）——
    sequence = case.get("sequence")
    seq_ok = True
    args_ok: bool | None = None
    args_failures: list[dict[str, Any]] = []
    if sequence is not None:
        seq_ok = exec_tools == list(sequence)
        args = case.get("args") or []
        args_ok = True
        if seq_ok:
            for i, step in enumerate(args):
                for k, v in (step.get("args_contains") or {}).items():
                    got = executed[i]["args"].get(k)
                    if got is None or (v and str(v) not in str(got)):
                        args_ok = False
                        args_failures.append({"step": i, "key": k, "expect": v, "got": got})
        else:
            args_ok = None  # 序列失配时参数校验无意义

    # —— 架构层：L2 不变量 ——
    arch_failures = check_arch_spec(case.get("mode", "react"), events)

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
        passed = bool(done_limit) and must_ok and not must_not_hits
    else:
        passed = (
            bool(done)
            and not done_limit
            and not error
            and must_ok
            and not must_not_hits
            and seq_ok
            and (args_ok is not False)
            and not arch_failures
        )

    return {
        "id": case["id"],
        "branch": case["branch"],
        "mode": case.get("mode", "react"),
        "query": case["query"],
        "executed": exec_tools,
        "must_call": must_call,
        "sequence": list(sequence) if sequence is not None else None,
        "must_ok": must_ok,
        "sequence_match": seq_ok if sequence is not None else None,
        "args_ok": args_ok,
        "args_failures": args_failures,
        "forbidden_hits": forbidden_hits,
        "arch_failures": arch_failures,
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


def _merge_runs(subs: list[dict[str, Any]], runs: int) -> dict[str, Any]:
    """合并同一用例的多次运行：成功率 + 均值耗时/成本（runs=1 时与原单次语义一致）。

    - passed：全部运行通过才算通过（runs=1 时即原语义）；
    - success_rate：通过次数 / runs（真实 LLM 下才有统计意义）；
    - 耗时/模型调用次数取均值；token 累加（整个用例多轮的累计消耗）。
    """
    rec = dict(subs[0])
    rec["runs"] = runs
    rec["pass_count"] = sum(1 for r in subs if r["passed"])
    rec["success_rate"] = round(rec["pass_count"] / runs, 3) if runs else 0.0
    rec["passed"] = rec["pass_count"] == runs
    rec["elapsed_ms"] = round(sum(r["elapsed_ms"] for r in subs) / runs, 1) if runs else 0.0
    rec["model_calls"] = round(sum(r["model_calls"] for r in subs) / runs, 1) if runs else 0
    rec["prompt_tokens"] = sum(r["prompt_tokens"] for r in subs)
    rec["completion_tokens"] = sum(r["completion_tokens"] for r in subs)
    return rec


def _latency_stats(times: list[float]) -> dict[str, float]:
    """耗时分布（min / p50 / p95 / max / avg），基于全部运行样本。"""
    if not times:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "avg": 0.0}
    ordered = sorted(times)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, int(len(ordered) * p))
        return round(ordered[idx], 1)

    return {
        "min": round(ordered[0], 1),
        "p50": pct(0.5),
        "p95": pct(0.95),
        "max": round(ordered[-1], 1),
        "avg": round(sum(times) / len(times), 1),
    }


def _aggregate(records: list[dict[str, Any]], latencies: list[float] | None = None) -> dict[str, Any]:
    n = len(records)
    modes: dict[str, dict[str, Any]] = {}
    branches: dict[str, dict[str, Any]] = {}
    for r in records:
        m = modes.setdefault(r["mode"], {"cases": 0, "passed": 0, "arch_failures": 0})
        m["cases"] += 1
        if r["passed"]:
            m["passed"] += 1
        m["arch_failures"] += len(r["arch_failures"])
        b = branches.setdefault(r["branch"], {"cases": 0, "passed": 0})
        b["cases"] += 1
        if r["passed"]:
            b["passed"] += 1
    total_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in records)
    return {
        "cases": n,
        "passed": sum(1 for r in records if r["passed"]),
        "pass_rate": round(sum(1 for r in records if r["passed"]) / n, 3) if n else 0.0,
        "must_call_rate": _mean(records, "must_ok"),
        "sequence_match_rate": _mean(records, "sequence_match"),
        "args_ok_rate": _mean(records, "args_ok"),
        "forbidden_hits": sum(len(r["forbidden_hits"]) for r in records),
        "arch_failures": sum(len(r["arch_failures"]) for r in records),
        "approval_cases": sum(1 for r in records if r["approval_requested"]),
        "loop_guard_cases": sum(1 for r in records if r["done_limit"]),
        "error_cases": sum(1 for r in records if r["error"]),
        "avg_elapsed_ms": round(sum(r["elapsed_ms"] for r in records) / n, 1) if n else 0.0,
        # 企业级补充维度：成功率（多跑）/ 成本 / 延迟分布
        "success_rate": _mean(records, "success_rate"),
        "cost": {
            "avg_model_calls": _mean(records, "model_calls"),
            "total_tokens": total_tokens,
            "avg_tokens": round(total_tokens / n, 1) if n else 0.0,
        },
        "latency_ms": _latency_stats(latencies or []),
        "modes": modes,
        "branches": branches,
    }


def run(runs: int = 1, real: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑全量 Agent 评测集（4 种架构），返回 (逐用例记录, 汇总报告)。

    - runs：每用例执行次数（默认 1）。runs>1 时逐用例给出 success_rate（通过次数/runs）；
      fake（确定性）下 N 次结果必同，成功率退化为 0/100%，无统计意义；
      real=True 用真实 LLM 驱动（需 Key），success_rate 才有真实鲁棒性含义。
    - real：用真实 LLM（chat 场景）替换剧本模型；默认 fake（剧本模型）全离线确定性。
    """
    runs = max(1, runs)
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in _load_cases():
        subs: list[dict[str, Any]] = []
        for _ in range(runs):
            events, elapsed_ms, stats = asyncio.run(_execute_case(case, real=real))
            rec = _evaluate(case, events, elapsed_ms)
            rec["model_calls"] = stats["calls"]
            rec["prompt_tokens"] = stats["prompt_tokens"]
            rec["completion_tokens"] = stats["completion_tokens"]
            subs.append(rec)
            latencies.append(elapsed_ms)
        records.append(_merge_runs(subs, runs))
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "version": get_app_version(),
            "mode": (
                "鲁棒性评测（真实 LLM chat 场景驱动，每用例 %d 次）" % runs
                if real
                else "任务层约束 + 架构层不变量（脚本化决策驱动真实 Harness，无真实 LLM）"
            ),
            "modules": "create_agent / StateGraph + StreamEventsMiddleware + AgentHarness",
            "driver": "真实 LLM（chat 场景）" if real else "FakeChatModel 脚本化决策",
            "runs": runs,
            "robustness": "success_rate = 通过次数 / runs（真实 LLM 驱动下才有统计意义）",
            "cost_source": "usage_metadata（input/output tokens），fake 模式下恒为 0",
            "eval_cases": len(records),
        },
        "overall": _aggregate(records, latencies),
        "cases": records,
    }
    return records, report


def save_report(report: dict[str, Any], path: str) -> None:
    """把报告写入 JSON（含逐用例轨迹明细，供失败样本回流分析）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
