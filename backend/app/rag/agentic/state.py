"""Agentic RAG 状态机数据结构：AgentState（共享状态）+ TraceEvent（可观测轨迹）。

企业级可观测口径：每个角色/工具调用都记录一条 TraceEvent（角色、思想、动作、
参数、命中数、时延、token 消耗、备注），Orchestrator 汇总为 RetrieveResult.trace——
支撑前端按角色渲染时间线与评测侧成本/轨迹核算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 状态机阶段（企业级 Agentic RAG：CRAG 证据评审闭环 + Self-RAG 答案校验闭环）
ST_ROUTE = "route"          # 路由：要不要检索 / 生成策略 / 定向提示
ST_PLAN = "plan"            # 规划：事实清单 + 首发检索计划
ST_RETRIEVE = "retrieve"    # 检索：工具注册表内并行执行一波调用
ST_GRADE = "grade"          # 评审（CRAG）：逐条证据相关性评分 + 缺口归纳
ST_CORRECT = "correct"      # 纠错：改写/换卷/换工具策略 → 回到 retrieve
ST_VERIFY = "verify"        # 校验（Self-RAG）：事实-证据支持度矩阵
ST_DONE = "done"            # 可答收尾
ST_CLARIFY = "clarify"      # 预算耗尽仍不足 → 如实上报缺口

# 建议（answerability 事件 recommendation 语义，与 runner/前端约定对齐）
REC_ANSWER = "answer"
REC_CLARIFY = "clarify"


@dataclass
class ToolCallSpec:
    """一次工具调用请求（角色决策产出，工具注册表执行）。"""

    action: str  # search / hybrid / volume_search / multi_hop
    query: str
    volume: str = ""  # 仅 volume_search：卷名（须在目录内）
    reason: str = ""  # 决策理由（对应事实缺口）


@dataclass
class TraceEvent:
    """一条可观测轨迹：角色决策或工具执行（latency/tokens 企业级成本核算口径）。"""

    seq: int  # 全局序号（从 1 递增）
    role: str  # router / planner / retriever / grader / corrector / verifier
    thought: str = ""  # 决策理由
    action: str = ""  # 动作（工具名 / 角色结果动词，如 route / grade）
    params: dict[str, Any] = field(default_factory=dict)  # 动作参数
    hits: int = 0  # 本步命中数（工具调用步）
    latency_ms: float = 0.0  # 本步耗时
    tokens: dict[str, int] = field(default_factory=dict)  # {"prompt":x,"completion":y}（LLM 步）
    note: str = ""  # 护栏备注（被拦截/降级/熔断回退原因，空=正常）

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "role": self.role,
            "thought": self.thought,
            "action": self.action,
            "params": self.params,
            "hits": self.hits,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "note": self.note,
        }


@dataclass
class AgentState:
    """一次查询的 Agent 共享状态：贯穿状态机各阶段，最终汇总为 trace。"""

    query: str
    deadline: float = 0.0  # 墙钟截止（time.monotonic 基准；预算治理用）
    # 阶段产物
    retrieval_need: bool = True  # 路由结论
    generation_mode: str = "citation"  # 生成策略（direct/citation/comparison，注入主 LLM）
    facts: list[str] = field(default_factory=list)  # 规划产出的事实清单
    confirmed_facts: list[str] = field(default_factory=list)  # 跨轮已确认事实（Verifier 校验后写回，防多轮遗忘）
    evidence: list[dict[str, Any]] = field(default_factory=list)  # 评审后的证据池（最终上下文）
    # 预算口径
    tool_calls: dict[str, int] = field(default_factory=dict)  # 各工具调用次数（含被护栏拦截，guard 记账）
    executed_keys: set[tuple[str, str, str]] = field(default_factory=set)  # 已正常执行的 (action,query,volume)，跨轮去重
    total_tool_exec: int = 0  # 工具调用总次数（含被护栏拦截，上限 max_steps）
    correction_rounds: int = 0  # 已用纠错轮数
    role_llm_calls: dict[str, int] = field(default_factory=dict)  # 各角色 LLM 调用次数
    fail_streak: dict[str, int] = field(default_factory=dict)  # 各角色连续失败次数（熔断依据）
    tokens: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    # 可观测
    events: list[TraceEvent] = field(default_factory=list)

    def next_seq(self) -> int:
        return len(self.events) + 1

    def add_event(self, event: TraceEvent) -> None:
        self.events.append(event)

    def add_tokens(self, usage: dict[str, int]) -> None:
        self.tokens["prompt"] += usage.get("prompt", 0)
        self.tokens["completion"] += usage.get("completion", 0)

    def over_budget(self, token_budget: int) -> bool:
        """token 预算是否已耗尽（角色调用降级为规则回退的依据）。"""
        return token_budget > 0 and (
            self.tokens["prompt"] + self.tokens["completion"] >= token_budget
        )

    def timed_out(self) -> bool:
        import time

        return self.deadline > 0 and time.monotonic() >= self.deadline

    def trace(self, corrections: int) -> dict[str, Any]:
        """汇总为 RetrieveResult.trace / retrieve 事件 trace 字段（评测与前端展示口径）。"""
        roles = {r: c for r, c in sorted(self.role_llm_calls.items())}
        return {
            "total_events": len(self.events),
            "tool_calls": dict(sorted(self.tool_calls.items())),
            "total_tool_exec": self.total_tool_exec,
            "corrections": corrections,
            "role_llm_calls": roles,
            "tokens": dict(self.tokens),
            "steps": [e.to_dict() for e in self.events],
        }
