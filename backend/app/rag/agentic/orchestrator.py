"""Agentic 编排器（LangGraph StateGraph）：ROUTE → PLAN → RETRIEVE → GRADE ⇄ CORRECT → VERIFY。

与 agents/modes（plan_execute / reflection）同款 LangGraph 原生编排：
- 状态机表达为 StateGraph（节点 + 条件边回环），替代原手写 while 循环；
- 共享状态挂在 _GraphState.agent（AgentState 就地变更），SSE 事件经 outbox
  通道逐 super-step 排空下发（保持 classify/plan/agent_step/grade/verify/correct/
  retrieve/compress/answerability 协议不变）；
- 全部同步阻塞调用统一 asyncio.to_thread（项目硬约束，不阻塞事件循环）。

企业级治理（全部在编排层统一施加，角色/工具层不管预算）：
- 步数预算：全轮工具调用总上限（含被护栏拦截的调用）；
- 纠错回环上限：CRAG 纠错最多 N 轮（默认 2），超出仍不足 → 如实上报 clarify；
- token 预算：角色 LLM 调用累计 token 达阈值 → 后续角色全部规则回退；
- 墙钟超时：单查询 deadline，超时后不再发起新的 LLM 决策与工具波次；
- 熔断：同一角色连续 2 次 LLM 决策失败（解析失败/调用异常）→ 该角色锁定规则回退。

证据管线（每波）：工具注册表并行执行 → RRF 融合（定向卷先多样性截断、跨轮 seed
作额外一路）→ 重排 → 证据评审（CRAG，无关剔除）→ 压缩 → 父块回填 → 事实校验
（Self-RAG，支持度矩阵）。检索执行不设 LLM 角色：决策在 Planner/Corrector，
执行治理在 ToolRegistry（企业级决策/执行分离）。
"""
from __future__ import annotations

import asyncio
import logging
import operator
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from app.rag.agentic.roles import (
    ROLE_CORRECT,
    ROLE_GRADE,
    ROLE_PLAN,
    ROLE_ROUTE,
    ROLE_VERIFY,
    CorrectorAgent,
    GraderAgent,
    PlannerAgent,
    RoleOutcome,
    RouterAgent,
    RouteOutcome,
    VerifierAgent,
)
from app.rag.agentic.state import (
    REC_ANSWER,
    REC_CLARIFY,
    AgentState,
    ToolCallSpec,
    TraceEvent,
)
from app.rag.agentic.tools import ACTION_VOLUME, ToolRegistry, diversify
from app.rag.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

_FAIL_STREAK_LIMIT = 2  # 角色连续失败熔断阈值


@dataclass
class OrchestratorBudgets:
    """预算治理配置（企业级资源治理口径，源自 settings）。"""

    max_steps: int = 8  # 全轮工具调用总上限（含被拦截）
    correction_rounds: int = 2  # 纠错回环上限
    timeout_s: float = 90.0  # 单查询墙钟超时
    token_budget: int = 8000  # 角色 LLM 累计 token 预算（0=不限）
    call_cap: int = 3  # 单工具调用上限
    parallel: int = 4  # 一波内并行工具调用数


@dataclass
class OrchResult:
    """编排结果：最终证据 + 闸门结论 + 契约字段 + 轨迹（方案层包装为 RetrieveResult）。

    层间结构化契约（L1 内层 → L2 外层）：verdict/missing_facts 供上层做决策，
    confidence 供上层评估可信度，cost 供上层记账/审计——外层只读契约、不替内层判断证据够不够。
    """

    hits: list[dict[str, Any]] = field(default_factory=list)
    reranked: bool = False
    compressed: dict[str, int] | None = None
    answerable: bool = False
    missing_facts: list[str] = field(default_factory=list)
    generation_mode: str = "citation"
    retrieval_need: bool = True
    facts: list[str] = field(default_factory=list)
    corrections: int = 0
    confidence: float = 0.0  # 充分性置信度（0~1，确定性口径）
    cost: dict[str, Any] = field(default_factory=dict)  # 检索成本 {"tokens","calls","latency_ms"}
    trace: dict[str, Any] = field(default_factory=dict)
    pipeline: dict[str, Any] = field(default_factory=dict)  # 检索链路完整明细（触发/变换/策略/筛选/排序）

    @property
    def verdict(self) -> dict[str, Any]:
        """answerability 事件 / RetrieveResult.answerability 口径（与 runner 约定对齐）。"""
        return {
            "answerable": self.answerable,
            "missing_facts": self.missing_facts,
            "recommendation": REC_ANSWER if self.answerable else REC_CLARIFY,
        }


def _volumes(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """命中卷分布（按次数降序，最多 4 组）：agent_step 事件摘要。"""
    counter: dict[str, int] = {}
    for h in hits:
        vol = (h.get("metadata") or {}).get("volume") or "未知卷"
        counter[vol] = counter.get(vol, 0) + 1
    return [{"volume": v, "count": c} for v, c in sorted(counter.items(), key=lambda kv: -kv[1])[:4]]


def _run_sync_in_thread(coro):
    """当前线程已处运行中事件循环时，转交独立线程的新 loop 跑完协程（同步语义不变）。

    用于 run()（同步入口）在异步测试/异步 handler 内被调用时兜底——
    同一线程内 asyncio.run 会与运行中的 loop 冲突。
    """
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class _GraphState(TypedDict, total=False):
    """LangGraph 图状态：AgentState 就地变更 + 跨节点通道（每查询独立实例）。"""

    agent: AgentState  # 共享 Agent 状态（含证据/预算/轨迹，节点内就地变更）
    outbox: Annotated[list[dict], operator.add]  # 节点产出的 SSE 事件（逐 super-step 排空）
    calls: list[ToolCallSpec]  # 当前波待执行调用
    executed_calls: list[ToolCallSpec]  # 全轮已执行调用（纠错去重/轨迹依据）
    k: int | None  # 本查询 top_k
    seed_hits: list[dict[str, Any]] | None  # 跨轮 seed（作额外一路融合）
    fused: list[dict[str, Any]]  # 当前波融合+重排后的候选证据
    reranked: bool  # 是否发生过重排
    compress_metrics: dict[str, int] | None  # 末轮压缩指标
    grade_missing: list[str]  # 本轮评审归纳缺失事实（纠错兜底）
    verdict: dict[str, Any] | None  # 末轮校验结论 {"answerable","missing_facts"}
    retrieve_ran: bool  # 本波是否实际执行检索（预算/超时短路）
    has_correct: bool  # 纠错波是否产出可用调用
    pre_route: dict[str, Any] | None  # 主循环外前置路由决策（复用则跳过 RouterAgent，避免二次路由）


class AgenticOrchestrator:
    """Agentic RAG LangGraph 状态机编排器：角色调度 + 预算治理 + 证据管线。

    状态机（与 run()/astream() 同一编译图，决策与执行分离）：
        START → route ──(检索无关)──→ END
                     └─→ plan → retrieve ──(预算/超时短路)──→ END
                                    └─→ grade → verify ──(可答/预算耗尽)──→ END
                                                       └─→ correct ──(无调用)──→ END
                                                                 └─→ retrieve（循环回波）
    """

    def __init__(
        self,
        store,
        embeddings,
        reranker,
        compressor,
        parent_resolver: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        multi_hop=None,
        max_hops: int = 3,
        budgets: OrchestratorBudgets | None = None,
        hyde=None,  # HyDE 展开器：透传给工具注册表（hybrid 工具内隐式附加）
    ):
        self.budgets = budgets or OrchestratorBudgets()
        self.registry = ToolRegistry(
            store,
            multi_hop=multi_hop,
            max_hops=max_hops,
            call_cap=self.budgets.call_cap,
            parallel=self.budgets.parallel,
            hyde=hyde,
        )
        self.catalog = self.registry.catalog
        self.reranker = reranker
        self.compressor = compressor
        self.parent_resolver = parent_resolver
        self.router = RouterAgent()
        self.planner = PlannerAgent()
        self.grader = GraderAgent()
        self.corrector = CorrectorAgent()
        self.verifier = VerifierAgent()
        # 递归上限宽松于预算：一轮纠错最多 retrieve+grade+verify+correct 四步（经 config 传入）
        self._recursion_limit = max(32, self.budgets.correction_rounds * 4 + 6)
        self._graph = self._build_graph()

    # ---- 治理：LLM 可用性（预算/超时/熔断） ----

    def _use_llm(self, state: AgentState, role: str) -> bool:
        if state.over_budget(self.budgets.token_budget):
            return False
        if state.timed_out():
            return False
        return state.fail_streak.get(role, 0) < _FAIL_STREAK_LIMIT

    def _stage(self, state: AgentState, role: str, action: str, fn) -> RoleOutcome:
        """执行一个角色决策阶段：计时 + token 差额记账 + 轨迹事件 + 熔断计数。"""
        tokens_before = dict(state.tokens)
        t0 = time.perf_counter()
        outcome = fn()
        latency = (time.perf_counter() - t0) * 1000
        delta = {
            "prompt": state.tokens["prompt"] - tokens_before["prompt"],
            "completion": state.tokens["completion"] - tokens_before["completion"],
        }
        if outcome.note:
            state.fail_streak[role] = state.fail_streak.get(role, 0) + 1
        else:
            state.fail_streak[role] = 0
        state.add_event(
            TraceEvent(
                seq=state.next_seq(), role=role, thought=outcome.thought, action=action,
                latency_ms=round(latency, 1), tokens=delta, note=outcome.note,
            )
        )
        return outcome

    # ---- 层间契约：置信度 / 成本（确定性口径，供外层消费与记账） ----

    @staticmethod
    def _confidence(agent: AgentState, answerable: bool, missing: list[str]) -> float:
        """充分性置信度（0~1，确定性）：证据覆盖事实比例打底，规则回退/无证据降权。

        口径：可答时 0.5+0.5×覆盖比；缺口时 0.2+0.4×覆盖比（覆盖部分仍可先答）；
        任一角色发生规则回退（fail_streak>0）减 0.1；无检索证据直接折半。
        """
        facts = agent.facts or [agent.query]
        cov = max(0.0, min(1.0, (len(facts) - len(missing)) / max(1, len(facts))))
        conf = (0.5 + 0.5 * cov) if answerable else (0.2 + 0.4 * cov)
        if not agent.evidence:
            conf *= 0.5
        if any(streak > 0 for streak in agent.fail_streak.values()):
            conf -= 0.1
        return round(max(0.0, min(1.0, conf)), 2)

    @staticmethod
    def _cost(agent: AgentState) -> dict[str, Any]:
        """检索成本口径：token 记账（已有）+ 工具调用数（含护栏拦截）+ 墙钟时延。"""
        return {
            "tokens": dict(agent.tokens),
            "calls": agent.total_tool_exec + sum(agent.role_llm_calls.values()),
            "latency_ms": round((time.monotonic() - agent.started) * 1000, 1) if agent.started else 0.0,
        }

    # ---- 证据管线（每波共享） ----

    @staticmethod
    def _merge_evidence(
        existing: list[dict[str, Any]], new: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """跨轮累积证据：按文本去重合并保序（CRAG 多轮纠错的证据记忆）。

        每波评审后的相关证据并入全量 evidence 而非覆盖——否则后续波次（含被
        工具护栏拦截的波）会把先前已确认的事实（如部门归属）覆盖丢失，导致
        Verifier/最终结果「忘记」前面查到的内容。
        """
        seen = {h.get("text") for h in existing}
        merged = list(existing)
        for h in new:
            if h.get("text") not in seen:
                merged.append(h)
                seen.add(h.get("text"))
        return merged

    def _fuse_wave(
        self,
        wave: list,
        seed: list[dict[str, Any]] | None,
        k: int,
        facts_len: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """一波结果 → RRF 融合候选：定向卷先多样性截断，seed 作额外一路。返回 (融合, 保留数)。

        保留数（keep）决定重排后进 Grader 评审的候选宽度：多事实查询按事实数放大
        （上限为宽召回候选数），避免把排名靠后但相关的实体档案块在评审前就被 top-k 截断——
        Grader 才是噪声过滤器，截断应发生在评审之后而非之前（企业级「先宽后严」）。
        """
        hit_lists: list[list[dict[str, Any]]] = [seed] if seed else []
        executed = 0
        for wr in wave:
            if wr.note or not wr.hits:
                continue
            hits = diversify(wr.hits) if wr.call.action == ACTION_VOLUME else wr.hits
            if hits:
                hit_lists.append(hits)
                executed += 1
        keep = k * executed if executed else k
        if seed:
            keep = max(keep, len(seed) + k)
        if facts_len > 1:
            keep = max(keep, min(self._recall_k(k), k * facts_len))
        return reciprocal_rank_fusion(hit_lists), keep

    def _recall_k(self, k: int) -> int:
        return max(k * 3, 9)

    # ---- 图节点（异步；阻塞调用一律 to_thread） ----

    async def _route_node(self, state: _GraphState) -> dict:
        agent = state["agent"]
        outbox: list[dict] = [
            {"type": "classify", "query": agent.query, "scheme": "agentic", "status": "running"},
        ]
        pre = state.get("pre_route")
        if pre:
            # 复用主循环外前置的轻量路由决策（route_only 产出）：跳过 RouterAgent，
            # 省一次 LLM 调用，且保证前后生成策略一致（同一 query 不重复路由）。
            route = RouteOutcome(
                retrieval_need=bool(pre.get("retrieval_need", True)),
                generation_mode=str(pre.get("generation_mode") or "citation"),
                thought=str(pre.get("reason") or "前置语义路由"),
            )
            logger.info("[orchestrator] 路由：复用前置决策 need=%s mode=%s", route.retrieval_need, route.generation_mode)
        else:
            route = await asyncio.to_thread(
                self._stage, agent, ROLE_ROUTE, "route",
                lambda: self.router.run(agent.query, agent, use_llm=self._use_llm(agent, ROLE_ROUTE)),
            )
        agent.retrieval_need = route.retrieval_need
        agent.generation_mode = route.generation_mode
        agent.route_thought = route.thought or ""  # pipeline.trigger.reason（完整链路日志）
        outbox.append({
            "type": "classify", "query": agent.query, "scheme": "agentic", "status": "done",
            "retrieval_need": route.retrieval_need,
            "generation_mode": route.generation_mode,
            "reason": route.thought,
        })
        logger.info(
            "[orchestrator] 路由：need=%s mode=%s（%s）",
            route.retrieval_need, route.generation_mode, route.thought or "规则回退",
        )
        return {"outbox": outbox}

    def _after_route(self, state: _GraphState) -> str:
        return "plan" if state["agent"].retrieval_need else END

    async def _plan_node(self, state: _GraphState) -> dict:
        agent = state["agent"]
        outbox: list[dict] = [
            {"type": "plan", "query": agent.query, "scheme": "agentic", "status": "running"},
        ]
        plan = await asyncio.to_thread(
            self._stage, agent, ROLE_PLAN, "plan",
            lambda: self.planner.run(agent.query, self.catalog, agent, use_llm=self._use_llm(agent, ROLE_PLAN)),
        )
        agent.facts = plan.facts
        outbox.append({
            "type": "plan", "query": agent.query, "scheme": "agentic", "status": "done",
            "facts": plan.facts,
            "calls": [
                {"action": c.action, "query": c.query, "volume": c.volume, "reason": c.reason}
                for c in plan.calls
            ],
            "thought": plan.thought,
        })
        logger.info(
            "[orchestrator] 规划：%d 项事实，首发 %d 路调用（%s）",
            len(plan.facts), len(plan.calls), plan.thought or "规则回退",
        )
        return {"outbox": outbox, "calls": list(plan.calls), "executed_calls": list(plan.calls)}

    async def _retrieve_node(self, state: _GraphState) -> dict:
        agent = state["agent"]
        b = self.budgets
        outbox: list[dict] = []
        remaining = b.max_steps - agent.total_tool_exec
        if remaining <= 0 or agent.timed_out():
            logger.info("[orchestrator] %s → 停止检索", "步数预算耗尽" if remaining <= 0 else "墙钟超时")
            return {"outbox": outbox, "retrieve_ran": False}
        calls = (state.get("calls") or [])[:remaining]
        wave = await asyncio.to_thread(self.registry.execute_wave, calls, state.get("k"), self._recall_k(state.get("k") or 3), agent)
        agent.total_tool_exec += len(calls)
        recall_k = self._recall_k(state.get("k") or 3)
        guarded = 0
        for wr in wave:
            params = {"query": wr.call.query, "reason": wr.call.reason}
            if wr.call.volume:
                params["volume"] = wr.call.volume
            if wr.note:
                guarded += 1  # 护栏拦截/降级计数（pipeline.filters）
            agent.add_event(
                TraceEvent(
                    seq=agent.next_seq(), role="retriever", action=wr.call.action,
                    params=params, hits=len(wr.hits), note=wr.note or "",
                )
            )
            # 每路检索明细（pipeline.strategy）：命中分数分布 + 查询变换方法
            scores = [round(float(h.get("score") or 0.0), 3) for h in wr.hits]
            scores.sort(reverse=True)
            agent.recall_meta.append({
                "tool": wr.call.action,
                "query": wr.call.query,
                "volume": wr.call.volume or None,
                "reason": wr.call.reason or "",
                "guarded": wr.note or None,
                "recall_k": recall_k,
                "hits": len(wr.hits),
                "scores": scores[:3],
                "query_pipeline": wr.query_pipeline,
            })
            outbox.append({
                "type": "agent_step", "query": agent.query, "scheme": "agentic",
                "step": {
                    "index": agent.next_seq() - 1,
                    "role": "retriever", "action": wr.call.action, "params": params,
                    "note": wr.note, "hits_count": len(wr.hits), "volumes": _volumes(wr.hits),
                },
            })
            logger.info(
                "[orchestrator] 检索 [%s] %r → %d 条%s",
                wr.call.action, wr.call.query, len(wr.hits), f"（{wr.note}）" if wr.note else "",
            )
        if guarded:
            agent.filters_meta.append({"name": "guard", "dropped": guarded})
        fused, keep = await asyncio.to_thread(
            self._fuse_wave, wave, state.get("seed_hits"), state.get("k") or 3, len(agent.facts)
        )
        agent.ranking_meta["fusion"] = {
            "method": "RRF(K=60)", "fused": len(fused), "keep": keep,
        }
        reranked = False
        if fused:
            before = len(fused)
            fused = await asyncio.to_thread(lambda: self.reranker.rerank(agent.query, fused)[:keep])
            reranked = True
            agent.ranking_meta["rerank"] = {
                "model": getattr(self.reranker, "model", "lexical"),
                "before": before, "after": len(fused),
            }
        return {"outbox": outbox, "fused": fused, "reranked": reranked, "retrieve_ran": True}

    def _after_retrieve(self, state: _GraphState) -> str:
        return "grade" if state.get("retrieve_ran") else END

    async def _grade_node(self, state: _GraphState) -> dict:
        agent = state["agent"]
        fused = state.get("fused") or []
        # GRADE（CRAG）：无关证据剔除（prior_hits=先前轮累积证据，缺失归纳不遗忘已确认事实）
        grade = await asyncio.to_thread(
            self._stage, agent, ROLE_GRADE, "grade",
            lambda g=fused, prior=list(agent.evidence): self.grader.run(
                agent.query, g, agent, use_llm=self._use_llm(agent, ROLE_GRADE), prior_hits=prior,
            ),
        )
        kept = [h for i, h in enumerate(fused) if i in set(grade.keep)] if grade.keep else []
        logger.info("[orchestrator] 评审：%d/%d 条相关（%s）", len(kept), len(fused), grade.thought or "规则回退")
        agent.filters_meta.append({"name": "grade", "kept": len(kept), "total": len(fused)})
        compress_metrics: dict[str, int] | None = None
        if kept:
            # 压缩只做去重 + 超长截断，不按 top_k 硬截断：Grader 确认的相关证据全部保留
            # （硬截断会丢掉先前确认的事实，导致 Verifier 误判缺口、Corrector 重复检索）
            kept, compress_metrics = await asyncio.to_thread(self.compressor.compress, agent.query, kept, len(kept))
            if compress_metrics:
                agent.filters_meta.append({"name": "compress", **compress_metrics})
        # 证据跨轮累积（子块，供 Grader/Verifier 精准判断；父块回填延后到最终结果）
        agent.evidence = self._merge_evidence(agent.evidence, kept)
        outbox: list[dict] = [{
            "type": "grade", "query": agent.query, "scheme": "agentic",
            "kept": len(kept), "total": len(fused),
            "missing_facts": grade.missing_facts, "thought": grade.thought,
        }]
        return {
            "outbox": outbox,
            "grade_missing": list(grade.missing_facts),
            "compress_metrics": compress_metrics,
        }

    async def _verify_node(self, state: _GraphState) -> dict:
        agent = state["agent"]
        # VERIFY（Self-RAG）：事实-证据支持度（基于全量累积证据 + 跨轮已确认事实）
        verify = await asyncio.to_thread(
            self._stage, agent, ROLE_VERIFY, "verify",
            lambda e=agent.evidence, cf=list(agent.confirmed_facts): self.verifier.run(
                agent.query, agent.facts, e, agent, use_llm=self._use_llm(agent, ROLE_VERIFY),
                confirmed_facts=cf,
            ),
        )
        # 跨轮记忆写回：本轮被判定「非缺失」的事实即已确认，后续轮不再重复报缺失
        if verify.missing_facts:
            agent.confirmed_facts = [
                f for f in agent.facts
                if f not in set(verify.missing_facts) and f not in agent.confirmed_facts
            ] + agent.confirmed_facts
        outbox: list[dict] = [{
            "type": "verify", "query": agent.query, "scheme": "agentic",
            "answerable": verify.answerable, "missing_facts": verify.missing_facts,
            "thought": verify.thought,
        }]
        logger.info(
            "[orchestrator] 校验：answerable=%s（%s）",
            verify.answerable, verify.thought or "规则回退",
        )
        return {
            "outbox": outbox,
            "verdict": {"answerable": verify.answerable, "missing_facts": list(verify.missing_facts)},
        }

    def _after_verify(self, state: _GraphState) -> str:
        agent = state["agent"]
        b = self.budgets
        verdict = state.get("verdict") or {}
        if verdict.get("answerable"):
            return END
        if agent.correction_rounds >= b.correction_rounds:
            logger.info("[orchestrator] 纠错轮数已达上限（%d）→ 如实上报缺口", b.correction_rounds)
            return END
        if agent.total_tool_exec >= b.max_steps or agent.timed_out() or agent.over_budget(b.token_budget):
            logger.info("[orchestrator] 预算耗尽 → 如实上报缺口")
            return END
        return "correct"

    async def _correct_node(self, state: _GraphState) -> dict:
        agent = state["agent"]
        # 纠错缺口用 Verifier 的全量证据归纳（已基于累积证据，不会重复搜索已确认事实）；
        # prior_hits=state.evidence：让 Corrector 基于已确认证据生成下一跳（如张三=研发部→查研发部人数）
        verdict = state.get("verdict") or {}
        missing = verdict.get("missing_facts") or state.get("grade_missing") or []
        correct = await asyncio.to_thread(
            self._stage, agent, ROLE_CORRECT, "correct",
            lambda m=missing, prior=list(agent.evidence): self.corrector.run(
                agent.query, m, state.get("executed_calls") or [], self.catalog, agent,
                use_llm=self._use_llm(agent, ROLE_CORRECT), prior_hits=prior,
            ),
        )
        outbox: list[dict] = [{
            "type": "correct", "query": agent.query, "scheme": "agentic",
            "round": agent.correction_rounds + 1, "thought": correct.thought,
            "calls": [
                {"action": c.action, "query": c.query, "volume": c.volume, "reason": c.reason}
                for c in correct.calls
            ],
        }]
        if not correct.calls:
            logger.info("[orchestrator] 纠错无可用调用 → 如实上报缺口")
            return {"outbox": outbox, "has_correct": False}
        executed_calls = list(state.get("executed_calls") or [])
        executed_calls.extend(correct.calls)
        agent.correction_rounds = agent.correction_rounds + 1
        logger.info(
            "[orchestrator] 纠错第 %d 轮：%d 路调用（%s）",
            agent.correction_rounds, len(correct.calls), correct.thought or "规则回退",
        )
        return {
            "outbox": outbox, "has_correct": True,
            "calls": list(correct.calls), "executed_calls": executed_calls,
        }

    def _after_correct(self, state: _GraphState) -> str:
        return "retrieve" if state.get("has_correct") else END

    # ---- 图构建 ----

    def _build_graph(self):
        """编译 LangGraph 状态机（与 agents/modes 同款 StateGraph 原生编排）。"""
        builder = StateGraph(_GraphState)
        builder.add_node("route", self._route_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade", self._grade_node)
        builder.add_node("verify", self._verify_node)
        builder.add_node("correct", self._correct_node)
        builder.add_edge(START, "route")
        builder.add_conditional_edges("route", self._after_route, {"plan": "plan", END: END})
        builder.add_edge("plan", "retrieve")
        builder.add_conditional_edges("retrieve", self._after_retrieve, {"grade": "grade", END: END})
        builder.add_edge("grade", "verify")
        builder.add_conditional_edges("verify", self._after_verify, {"correct": "correct", END: END})
        builder.add_conditional_edges("correct", self._after_correct, {"retrieve": "retrieve", END: END})
        return builder.compile()

    @staticmethod
    def _initial_state(query: str, k: int | None, seed_hits: list[dict[str, Any]] | None, budgets, pre_route: dict[str, Any] | None = None) -> dict:
        agent = AgentState(query=query, deadline=time.monotonic() + budgets.timeout_s, started=time.monotonic())
        return {
            "agent": agent,
            "outbox": [],
            "k": k,
            "seed_hits": seed_hits,
            "reranked": False,
            "compress_metrics": None,
            "verdict": None,
            "pre_route": pre_route,
        }

    def _build_result(self, agent: AgentState, final: dict) -> OrchResult:
        """终态 → OrchResult（父块回填：Verifier 基于精准子块判断，注入生成 LLM 用父块全文）。

        层间契约：confidence（充分性置信度）与 cost（token/调用/时延）随结果下发，
        供外层决策/记账；非检索路径（寒暄）只记成本，不评置信度。
        """
        if not agent.retrieval_need:
            result = OrchResult(retrieval_need=False, generation_mode=agent.generation_mode)
            result.cost = self._cost(agent)
            result.trace = agent.trace(0)
            result.pipeline = agent.build_pipeline()
            return result
        verdict = final.get("verdict") or {}
        answerable = bool(verdict.get("answerable"))
        missing = list(verdict.get("missing_facts")) if verdict else list(agent.facts)
        result = OrchResult(
            hits=self.parent_resolver(agent.evidence),
            reranked=bool(final.get("reranked")),
            compressed=final.get("compress_metrics"),
            answerable=answerable,
            missing_facts=missing,
            generation_mode=agent.generation_mode,
            facts=agent.facts,
            corrections=agent.correction_rounds,
            confidence=self._confidence(agent, answerable, missing),
            cost=self._cost(agent),
        )
        result.trace = agent.trace(agent.correction_rounds)
        result.pipeline = agent.build_pipeline()
        return result

    # ---- 主入口（同步） ----

    def route_only(self, query: str) -> RouteOutcome:
        """轻量路由（不进图）：只跑 Router 角色（LLM + 规则回退），供主循环外前置
        做生成策略决策，产出可复用于检索的 pre_route（避免工具内二次路由）。"""
        agent = AgentState(query=query, deadline=time.monotonic() + self.budgets.timeout_s, started=time.monotonic())
        return self._stage(
            agent, ROLE_ROUTE, "route",
            lambda: self.router.run(query, agent, use_llm=self._use_llm(agent, ROLE_ROUTE)),
        )

    def run(self, query: str, k: int | None = None, seed_hits: list[dict[str, Any]] | None = None, pre_route: dict[str, Any] | None = None) -> OrchResult:
        """同步编排：编译图 ainvoke 一次跑完状态机（评测脚本/同步上下文用）。

        当前线程已处运行中事件循环时（异步 handler 内同步调用 retrieve_full），
        转交独立线程执行，保持同步语义不变。
        """
        initial = self._initial_state(query, k, seed_hits, self.budgets, pre_route)
        config = {"recursion_limit": self._recursion_limit}
        coro = self._graph.ainvoke(initial, config=config)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            final = asyncio.run(coro)
        else:
            final = _run_sync_in_thread(coro)
        return self._build_result(initial["agent"], final)

    # ---- 主入口（异步流式：逐事件下发） ----

    async def astream(self, query: str, k: int | None = None, seed_hits: list[dict[str, Any]] | None = None, pre_route: dict[str, Any] | None = None):
        """异步流式编排：编译图 astream 逐 super-step 排空 outbox 事件。

        事件序列：classify(running/done) → plan(running/done) → [agent_step* → grade →
        verify →（不足则 correct → 下一波）] → retrieve(含 trace) → compress → answerability。
        决策与工具执行均为同步阻塞调用，统一放线程池不阻塞事件循环（项目硬约束）。

        pre_route：主循环外前置的路由决策（{retrieval_need, generation_mode, reason}），
        传入则 route 节点跳过 RouterAgent 直接复用（省一次 LLM、保证生成策略前后一致）。
        """
        initial = self._initial_state(query, k, seed_hits, self.budgets, pre_route)
        agent = initial["agent"]
        sent = 0
        final = initial
        async for snap in self._graph.astream(
            initial, stream_mode="values", config={"recursion_limit": self._recursion_limit}
        ):
            final = snap
            outbox = snap.get("outbox") or []
            if len(outbox) > sent:
                for ev in outbox[sent:]:
                    yield ev
                sent = len(outbox)
        result = self._build_result(agent, final)
        if not agent.retrieval_need:
            return  # 寒暄：仅 classify 两条，不再下发检索/闸门事件
        yield {
            "type": "retrieve", "query": agent.query, "scheme": "agentic",
            "hits": result.hits, "reranked": result.reranked, "trace": result.trace,
            "pipeline": result.pipeline,
            "confidence": result.confidence, "cost": result.cost,
        }
        if result.compressed and (
            result.compressed["kept"] < result.compressed["original"] or result.compressed["truncated"] > 0
        ):
            yield {"type": "compress", "query": agent.query, "scheme": "agentic", "metrics": result.compressed}
        yield {
            "type": "answerability", "query": agent.query, "scheme": "agentic",
            "verdict": result.verdict, "escalated": result.corrections > 0,
            "confidence": result.confidence, "cost": result.cost,
        }
