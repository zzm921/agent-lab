"""检索任务编排层：任务图状态机（LangGraph，与内层同构）+ 任务账本预算。

L2 外层任务闭环：任务拆解（decompose）→ 逐节点执行（run_next 回环，依赖满足才就绪）
→ 任务级校验（task_verify）→ 收尾。节点执行通过注入的 run_node 消费「内层契约」，
编排层不接触检索实现；任务黑板（证据池 seed / 缺口 gaps / token 账本 / 事件 outbox）
跨节点共享，各组件读写黑板而非直接相互调用。

任务账本预算（防级联超支，P3 与会话账本叠加）：
- 节点数上限（拆解器封顶）｜内层触发上限（含节点与重查）｜任务 token 预算｜墙钟超时；
- 预算耗尽 ≠ 失败：终止并如实上报（note 说明原因，可答部分先行）。
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from app.rag.task.decomposer import TaskDecomposer
from app.rag.task.gap_center import ACT_REWRITE, GapStrategyCenter
from app.rag.task.graph import (
    NS_RESOLVED,
    TC_CLARIFIED,
    TC_COMPLETE,
    TC_PARTIAL,
    NodeResult,
    SessionLedger,
    TaskBudgets,
    TaskGraphState,
    TaskNode,
    TaskResult,
)

logger = logging.getLogger(__name__)

# run_node 契约：async (node: TaskNode, seed: list[dict]|None) -> dict
#   {hits, verdict, missing_facts, confidence, cost, note}
RunNode = Callable[[TaskNode, list[dict[str, Any]] | None], Awaitable[dict[str, Any]]]


def _merge_evidence(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """黑板证据池：跨节点按文本去重合并保序（节点 n2 的检索可复用 n1 已确认证据）。"""
    seen = {h.get("text") for h in existing}
    merged = list(existing)
    for h in new:
        if h.get("text") not in seen:
            merged.append(h)
            seen.add(h.get("text"))
    return merged


def _run_sync_in_thread(coro):
    """当前线程已处运行中事件循环时，转交独立线程的新 loop 跑完协程（同步语义不变）。"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class TaskExecutor:
    """任务图状态机编排器：拆解 → 节点回环执行 → 任务级校验。

    run_node：内层闭环消费者（由接入方注入，读取内层契约，返回 NodeResult 载荷）。
    """

    def __init__(
        self,
        run_node: RunNode,
        budgets: TaskBudgets | None = None,
        decomposer: TaskDecomposer | None = None,
        gap_center: GapStrategyCenter | None = None,
        session_ledger: SessionLedger | None = None,
    ):
        self.run_node = run_node
        self.budgets = budgets or TaskBudgets()
        self.decomposer = decomposer or TaskDecomposer(max_nodes=self.budgets.max_nodes)
        self.gap_center = gap_center or GapStrategyCenter()
        self.session = session_ledger  # 会话账本（P3）：与任务账本叠加，跨任务累计
        self._recursion_limit = max(32, self.budgets.max_nodes * 3 + 8)
        self._graph = self._build_graph()

    # ---- 图节点 ----

    async def _decompose_node(self, state: TaskGraphState) -> dict:
        nodes, thought, note = await asyncio.to_thread(
            self.decomposer.decompose, state["query"], ledger=state["task_tokens"]
        )
        logger.info("[task] 拆解：%d 节点（%s）", len(nodes), thought)
        return {
            "nodes": nodes,
            "outbox": [{
                "type": "task_plan", "task_id": state["task_id"], "query": state["query"],
                "nodes": nodes, "thought": thought, "note": note,
            }]
        }

    def _next_ready(self, state: TaskGraphState) -> dict[str, Any] | None:
        """下一个就绪节点：未执行且全部依赖已执行（DAG 拓扑序，天然防环执行）。"""
        nodes = state.get("nodes") or []
        results = state.get("results") or {}
        for n in nodes:
            nid = n.get("id")
            if nid in results:
                continue
            deps = n.get("deps") or []
            if all(d in results for d in deps):
                return n
        return None

    async def _run_next_node(self, state: TaskGraphState) -> dict:
        """执行一个就绪节点：触发内层闭环 → 缺口策略中心决策（改写重查回环）→ 黑板累积。

        节点内循环（P2）：内层返回缺口 → 策略中心分类（query=改写重查，上限 max_retries；
        data/cross_domain=如实上报；low_value=部分接受）→ 改写则同一节点换查询重查。
        每次内层触发都记账 inner_calls（防级联超支）；失败尝试的证据并入黑板证据池，
        改写重查时作为 seed 复用（已确认部分不再重复检索）。

        LangGraph 输入 state 为通道值副本：节点对已存在通道的重绑定（state["x"]=...）
        不会写回；故 nodes/results/seed/task_tokens/inner_calls/resolved/gaps 整表返回
        （resolved/gaps 为普通 list 通道，返回即整体替换；outbox 走归约器追加）。
        """
        node = self._next_ready(state)
        if node is None:
            return {}
        b = self.budgets
        results = dict(state.get("results") or {})
        task_tokens = dict(state.get("task_tokens") or {})
        resolved = list(state.get("resolved") or [])
        gaps = list(state.get("gaps") or [])
        seed = list(state.get("seed") or [])
        outbox: list[dict] = []
        inner_calls = state.get("inner_calls", 0)

        task_node = TaskNode(
            id=node["id"], query=node["query"],
            deps=list(node.get("deps") or []), reason=node.get("reason") or "",
        )
        attempt_seed = list(seed)
        node_retries = 0
        nr: NodeResult | None = None
        while True:
            # 会话账本（P3）：先扣后走——任务每次触发内层前检查会话余量，触顶即止（防跨任务叠加超支）
            if self.session is not None and self.session.exhausted():
                gaps.append({
                    "node_id": node["id"], "query": task_node.query,
                    "missing_facts": [], "confidence": 0.0,
                    "gap_type": "data", "action": "report", "note": "会话预算耗尽（跨任务累计触顶）",
                })
                break
            if inner_calls >= b.max_inner_calls:
                gaps.append({
                    "node_id": node["id"], "query": task_node.query,
                    "missing_facts": [], "confidence": 0.0,
                    "gap_type": "data", "action": "report", "note": "任务内层触发已达上限",
                })
                break
            inner_calls += 1
            contract = await self.run_node(task_node, attempt_seed)
            nr = NodeResult(
                node_id=node["id"],
                query=task_node.query,
                hits=list(contract.get("hits") or []),
                verdict=dict(contract.get("verdict") or {}),
                missing_facts=list(contract.get("missing_facts") or []),
                confidence=float(contract.get("confidence") or 0.0),
                cost=dict(contract.get("cost") or {}),
                note=contract.get("note") or "",
            )
            # 任务账本：内层契约 cost.tokens 汇入任务 token 记账；会话账本同步并入（跨任务累计）
            nt = nr.cost.get("tokens") or {}
            task_tokens["prompt"] += nt.get("prompt", 0)
            task_tokens["completion"] += nt.get("completion", 0)
            if self.session is not None:
                self.session.merge(nt, inner_calls=1)
            # 黑板证据池：失败尝试的证据同样并入（改写重查时作为 seed 复用）
            attempt_seed = _merge_evidence(attempt_seed, nr.hits)
            if nr.state == NS_RESOLVED:
                resolved.append(node["id"])
                seed = _merge_evidence(seed, attempt_seed)
                break
            # 缺口 → 策略中心：分类 + 决策表收敛（改写/上报/接受）
            decision = await self.gap_center.decide(
                task_node.query, nr.missing_facts, attempt_seed, node_retries, b.max_retries,
                ledger=task_tokens,
            )
            if decision.action == ACT_REWRITE and decision.rewrite_query and node_retries < b.max_retries:
                node_retries += 1
                outbox.append({
                    "type": "task_retry", "task_id": state["task_id"], "query": state["query"],
                    "node_id": node["id"], "query_prev": task_node.query,
                    "rewrite_query": decision.rewrite_query, "retries": node_retries,
                    "gap_type": decision.gap_type, "reason": decision.reason,
                })
                task_node = TaskNode(
                    id=node["id"], query=decision.rewrite_query,
                    deps=list(node.get("deps") or []),
                    reason=f"缺口改写重查({node_retries})",
                )
                continue
            gaps.append({
                "node_id": node["id"], "query": node["query"],
                "missing_facts": nr.missing_facts, "confidence": nr.confidence,
                "gap_type": decision.gap_type, "action": decision.action,
                "note": decision.note or "",
            })
            break
        # 收尾：终态尝试契约入库（resolved 或 gap 均记录最后一次 NodeResult，
        # 供 TaskResult.results / resolved_count / gap_count 与置信度聚合使用）
        results[node["id"]] = nr if nr is not None else NodeResult(
            node_id=node["id"], query=node["query"],
            verdict={"answerable": False}, missing_facts=[], confidence=0.0,
        )
        is_resolved = node["id"] in resolved
        final_nr = results[node["id"]]
        # 缺口备注（P4 事件协议：task_node 携带该节点缺口 note，前端可直接展示）
        gap_note = ""
        if not is_resolved:
            for g in reversed(gaps):
                if g.get("node_id") == node["id"]:
                    gap_note = g.get("note") or ""
                    break
        logger.info(
            "[task] 节点 %s 收尾：resolved=%s retries=%d 命中 %d 条",
            node["id"], node["id"] in resolved, node_retries, len(attempt_seed),
        )
        return {
            "inner_calls": inner_calls,
            "results": results,
            "task_tokens": task_tokens,
            "resolved": resolved,
            "gaps": gaps,
            "seed": seed,
            "outbox": outbox + [{
                "type": "task_node", "task_id": state["task_id"], "query": state["query"],
                "node_id": node["id"], "node_query": node["query"], "state": "resolved" if is_resolved else "gap",
                "verdict": {"answerable": is_resolved},
                "missing_facts": [] if is_resolved else list(final_nr.missing_facts),
                "confidence": final_nr.confidence or 0.0,
                "cost": final_nr.cost,
                "hits_count": len(attempt_seed), "retries": node_retries, "note": gap_note,
            }],
        }

    def _after_run_next(self, state: TaskGraphState) -> str:
        """回环出口判定（含任务账本与会话账本护栏）：预算/超时耗尽或无可就绪节点 → task_verify。"""
        b = self.budgets
        if self.session is not None and self.session.exhausted():
            return "task_verify"
        if state.get("inner_calls", 0) >= b.max_inner_calls:
            return "task_verify"
        tokens = state.get("task_tokens") or {}
        if b.token_budget > 0 and (tokens.get("prompt", 0) + tokens.get("completion", 0)) >= b.token_budget:
            return "task_verify"
        if time.monotonic() - state["started"] >= b.timeout_s:
            return "task_verify"
        if self._next_ready(state) is not None:
            return "run_next"
        return "task_verify"

    async def _verify_node(self, state: TaskGraphState) -> dict:
        b = self.budgets
        nodes = state.get("nodes") or []
        results = state.get("results") or {}
        resolved = [i for i in (state.get("resolved") or []) if i in results]
        # 终止备注（与 _after_run_next 同一判定口径，避免路由函数改写不落库）
        tokens = state.get("task_tokens") or {}
        if self.session is not None and self.session.exhausted():
            note = "会话预算耗尽（跨任务累计触顶，建议开新会话或精简复合检索）"
        elif state.get("inner_calls", 0) >= b.max_inner_calls:
            note = "任务内层触发已达上限"
        elif b.token_budget > 0 and (tokens.get("prompt", 0) + tokens.get("completion", 0)) >= b.token_budget:
            note = "任务 token 预算耗尽"
        elif time.monotonic() - state["started"] >= b.timeout_s:
            note = "任务墙钟超时"
        elif len(results) < len(nodes):
            note = "任务图存在依赖死锁/不可达节点"
        else:
            note = ""
        if not nodes:
            completion = TC_CLARIFIED
        elif len(resolved) == len(nodes):
            completion = TC_COMPLETE
        elif resolved:
            completion = TC_PARTIAL
        else:
            completion = TC_CLARIFIED
        confs = [results[i].confidence for i in resolved]
        confidence = round(sum(confs) / len(confs), 2) if confs else 0.0
        cost = {
            "tokens": dict(state.get("task_tokens") or {}),
            "calls": state.get("inner_calls", 0),
            "latency_ms": round((time.monotonic() - state["started"]) * 1000, 1),
        }
        result = TaskResult(
            task_id=state["task_id"], query=state["query"], nodes=nodes, results=results,
            completion=completion, evidence=state.get("seed") or [],
            gaps=state.get("gaps") or [], confidence=confidence, cost=cost,
            trace={
                "note": note,
                "inner_calls": state.get("inner_calls", 0),
                "session_inner_calls": self.session.inner_calls if self.session is not None else None,
            },
        )
        logger.info("[task] 完成 %s：completion=%s resolved=%d gaps=%d conf=%s", result.task_id, completion, result.resolved_count, result.gap_count, confidence)
        return {
            "result": result,
            "note": note,
            "outbox": [{
                "type": "task_done", "task_id": state["task_id"], "query": state["query"],
                "result": result.to_dict(), "note": note,
            }]
        }

    # ---- 图构建 ----

    def _build_graph(self):
        builder = StateGraph(TaskGraphState)
        builder.add_node("decompose", self._decompose_node)
        builder.add_node("run_next", self._run_next_node)
        builder.add_node("task_verify", self._verify_node)
        builder.add_edge(START, "decompose")
        builder.add_conditional_edges("decompose", self._after_run_next, {"run_next": "run_next", "task_verify": "task_verify"})
        builder.add_conditional_edges("run_next", self._after_run_next, {"run_next": "run_next", "task_verify": "task_verify"})
        builder.add_edge("task_verify", END)
        return builder.compile()

    @staticmethod
    def _initial_state(query: str, task_id: str | None) -> dict:
        return {
            "task_id": task_id or uuid.uuid4().hex[:8],
            "query": query,
            "nodes": [],
            "results": {},
            "resolved": [],
            "gaps": [],
            "seed": [],
            "task_tokens": {"prompt": 0, "completion": 0},
            "inner_calls": 0,
            "outbox": [],
            "started": time.monotonic(),
            "note": "",
            "result": None,
        }

    # ---- 主入口（同步） ----

    def run(self, query: str, task_id: str | None = None) -> TaskResult:
        """同步编排：编译图 ainvoke 一次跑完（评测脚本/同步上下文用）。"""
        initial = self._initial_state(query, task_id)
        coro = self._graph.ainvoke(initial, config={"recursion_limit": self._recursion_limit})
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            final = asyncio.run(coro)
        else:
            final = _run_sync_in_thread(coro)
        return final.get("result")

    # ---- 主入口（异步流式：逐事件下发） ----

    async def astream(self, query: str, task_id: str | None = None):
        """异步流式：逐 super-step 排空 outbox（task_plan → task_node* → task_done）。"""
        initial = self._initial_state(query, task_id)
        sent = 0
        async for snap in self._graph.astream(initial, stream_mode="values", config={"recursion_limit": self._recursion_limit}):
            outbox = snap.get("outbox") or []
            if len(outbox) > sent:
                for ev in outbox[sent:]:
                    yield ev
                sent = len(outbox)
