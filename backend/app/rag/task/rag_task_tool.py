"""检索任务编排工具：把「检索任务编排层」封装为主循环内的单一工具（L2 外层任务闭环）。

与 knowledge_retrieve（单查询内层闭环）互补：
- knowledge_retrieve：单查询充分性闭环（route→plan→retrieve→grade→verify⇄correct）；
- knowledge_task：复合任务的整棵任务图编排——拆解为子查询节点 DAG，逐节点触发内层闭环，
  黑板证据池跨节点累积复用，任务级校验汇总 completion，合并证据返回主 Agent。

层间契约：本工具只读内层契约（verdict/missing_facts/confidence/cost）驱动任务图，
不替内层判断证据够不够；结构化结果经 holder["task_state"] 旁路透传（程序化消费），
文本返回附任务状态行（模型直接理解）。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langchain_core.tools import tool

from app.rag.task.executor import TaskExecutor
from app.rag.task.graph import SessionLedger, TaskBudgets, TaskNode
from app.tools.rag_tool import _MAX_QUERY_CHARS, rag_block_payload

logger = logging.getLogger(__name__)


def task_status_line(task: dict[str, Any]) -> str:
    """任务状态摘要（结构化一行）：主 Agent 直接读到完成度/节点/置信度/成本/缺口类型。"""
    parts = [
        f"任务完成度={task.get('completion', 'clarified')}",
        f"节点={task.get('resolved', 0)}可答/{task.get('gaps', 0)}缺口",
    ]
    gap_types: dict[str, int] = {}
    for g in task.get("gap_list") or []:
        t = g.get("gap_type") or "data"
        gap_types[t] = gap_types.get(t, 0) + 1
    if gap_types:
        parts.append("缺口类型=" + ",".join(f"{k}×{v}" for k, v in gap_types.items()))
    if task.get("confidence"):
        parts.append(f"置信度={task['confidence']:.2f}")
    cost = task.get("cost") or {}
    parts.append(f"内层调用={cost.get('calls', 0)}")
    ms = cost.get("latency_ms") or 0
    if ms:
        parts.append(f"耗时={ms / 1000:.1f}s")
    return "【检索状态】" + " | ".join(parts)


def make_knowledge_task_tool(scheme, settings, emit, session_id, last_hits, context_holder, session_ledger: SessionLedger | None = None):
    """构建 knowledge_task 工具（agentic 单一检索入口）：检索任务编排层封装。

    入口职责（合并后，不再并列 knowledge_retrieve）：
    - 拆解器先规则粗筛：简单问题单节点直通内层（零额外 LLM 成本）；
    - 复合/链式问题 → 任务图状态机（节点 DAG + 独立缺口处理 + 任务级完成度）。

    节点执行复用方案内层闭环：逐节点 scheme.astream（读契约、转发内层事件）、
    黑板证据池作 seed 跨节点复用、前置路由决策（pre_route）与指代上下文共用。
    session_ledger（P3）：会话账本实例，与任务账本叠加——跨任务累计内层触发与
    token，任务执行中先扣后走，触顶如实上报。

    事件协议（P4）：任务图事件（task_plan/task_retry/task_node/task_done）自带
    task_id/node_id；节点执行中转发给主循环的内层事件（retrieve/answerability…）
    统一附加 task_id/node_id，前端可按任务图节点聚合检索轨迹。
    """
    task_ctx: dict[str, str] = {"task_id": ""}  # P4：当前任务 id（供 run_node 转发内层事件标注）

    async def run_node(node: TaskNode, seed: list[dict[str, Any]] | None) -> dict[str, Any]:
        """单节点执行：触发内层闭环 → 读契约（外层只读 verdict，不替内层判断）。

        转发内层事件时附加 task_id/node_id（P4 统一事件协议），使内层轨迹可关联到
        具体任务图节点。
        """
        holder = context_holder if context_holder is not None else {}
        stream_kwargs = {"context": holder.get("recent")}
        if seed:
            stream_kwargs["seed_hits"] = seed
        pre_route = holder.get("route")
        if pre_route:
            stream_kwargs["pre_route"] = pre_route
        hits: list[dict[str, Any]] = []
        verdict: dict[str, Any] = {"answerable": False}
        missing: list[str] = []
        confidence: float | None = None
        cost: dict[str, Any] = {}
        task_id = task_ctx["task_id"]
        try:
            async for ev in scheme.astream(node.query, settings.rag_top_k, **stream_kwargs):
                if emit is not None:
                    emit({**ev, "task_id": task_id, "node_id": node.id})
                if ev["type"] == "retrieve" and ev.get("hits"):
                    hits = ev["hits"]
                elif ev["type"] == "answerability":
                    verdict = ev.get("verdict") or {}
                    missing = list(verdict.get("missing_facts") or [])
                    confidence = ev.get("confidence")
                    cost = ev.get("cost") or {}
        except Exception as exc:  # noqa: BLE001 — 单节点检索故障降级为缺口，不中断任务
            logger.exception("[rag_task] 节点 %s 检索执行失败: %s", node.id, exc)
            return {
                "hits": [], "verdict": {"answerable": False},
                "missing_facts": [node.query], "confidence": 0.0,
                "cost": {"tokens": {"prompt": 0, "completion": 0}, "calls": 0, "latency_ms": 0.0},
                "note": "内层检索故障",
            }
        return {
            "hits": hits,
            "verdict": verdict,
            "missing_facts": missing,
            "confidence": confidence or 0.0,
            "cost": cost,
            "note": "",
        }

    @tool
    async def knowledge_task(query: str) -> str:
        """对需要拆解为多个子问题的复合检索任务执行一次完整编排（自动拆解子问题 → 逐节点检索 → 合并证据）。

        当问题涉及多个事实、规则与数据对比、或存在链式依赖（先查 A 再查 A 的属性）时调用本工具，
        一次调用返回合并后的依据文本与检索状态；简单单问题请用 knowledge_retrieve。
        若返回提示「检索结果不足」，请向用户追问澄清关键信息，不要编造内部数据。
        """
        if not query or not query.strip():
            return "检索任务查询为空，请向用户补充要查询的具体内容后再检索。"
        query = query.strip()[:_MAX_QUERY_CHARS]
        started = time.monotonic()
        holder = context_holder if context_holder is not None else {}
        budgets = TaskBudgets(
            max_nodes=settings.rag_agent_task_max_nodes,
            max_retries=settings.rag_agent_task_max_retries,
            max_inner_calls=settings.rag_agent_task_max_inner_calls,
            token_budget=settings.rag_agent_task_token_budget,
        )
        executor = TaskExecutor(run_node, budgets=budgets, session_ledger=session_ledger)
        task: dict[str, Any] | None = None
        # P4 统一事件协议：任务级事件统一 task_id（任务图节点内层事件经 task_ctx 标注）
        task_id = uuid.uuid4().hex[:8]
        task_ctx["task_id"] = task_id
        try:
            async for ev in executor.astream(query, task_id=task_id):
                if emit is not None:
                    emit(ev)
                if ev["type"] == "task_done":
                    task = ev["result"]
        except Exception as exc:  # noqa: BLE001 — 任务编排故障降级为未命中，不向 Agent 抛工具异常
            logger.exception("[rag_task] knowledge_task 执行失败: %s", exc)
            return (
                f"检索任务编排执行失败（{type(exc).__name__}）。请如实告知用户当前检索遇到故障，"
                "建议稍后重试或更换关键词；不要编造内部数据。"
            )
        finally:
            logger.info(
                "[rag_task] knowledge_task 完成 session=%s nodes=%d 耗时=%.3fs",
                session_id, (task or {}).get("resolved", 0), time.monotonic() - started,
            )
        if task is None:
            return (
                f"检索任务编排未完成（“{query}”）。请如实告知用户当前检索未能获取足够信息，"
                "并礼貌地向用户追问补充依据；不要编造内部数据。"
            )
        # 跨轮 seed 缓存更新：本轮任务合并证据（供下一轮过滤复用），否则清空
        if task.get("evidence"):
            last_hits[session_id] = task["evidence"]
        else:
            last_hits.pop(session_id, None)
        # 任务黑板结果旁路透传（程序化消费，零文本解析）
        holder["task_state"] = task
        evidence = task.get("evidence") or []
        if not evidence:
            return (
                f"检索任务编排未命中与“{query}”相关的内容。请如实告知用户当前检索未能获取足够信息，"
                "说明缺失的关键信息，并礼貌地向用户追问补充依据（如具体文件、部门名称等）；"
                "不要编造、不要依赖自身知识臆测内部数据。"
            )
        completion = task.get("completion", "clarified")
        generation_mode = (holder.get("route") or {}).get("generation_mode") or "citation"
        block = rag_block_payload(
            {"name": getattr(scheme, "name", "知识库"), "hits": evidence},
            insufficient=completion != "complete",
            generation_mode=generation_mode,
            query=query,
        )
        status = task_status_line(task)
        # 会话账本（P3）：状态行附跨任务余量，主 Agent/前端可见治理状态
        if session_ledger is not None and session_ledger.max_inner_calls > 0:
            status += f" | 会话内层余量={session_ledger.remaining_inner()}/{session_ledger.max_inner_calls}"
        return f"{block}\n{status}"

    return knowledge_task
