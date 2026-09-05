"""检索任务编排层数据结构：任务图（节点 DAG）+ 节点状态 + 任务结果。

L2 外层任务闭环的形式化载体：复合问题 → 子查询节点 DAG，每个节点是一个有状态
小循环（planned → queried → verified → resolved / gap），由任务图状态机驱动；
「任务黑板」跨节点共享（证据池/缺口/预算/事件轨迹），各组件读写黑板而非直接相互调用。

层间契约纪律：外层只读内层契约（verdict/missing_facts/confidence/cost）做决策，
不替内层判断证据够不够；内层不替外层判断任务做没做完。
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

# 节点状态（外层任务闭环）
NS_PLANNED = "planned"    # 已规划待执行
NS_QUERIED = "queried"    # 已触发内层检索
NS_VERIFIED = "verified"  # 内层校验返回
NS_RESOLVED = "resolved"  # 证据充分（可答）
NS_GAP = "gap"            # 证据不足（缺口，转策略中心）

# 任务完成度（外层契约 completion）
TC_COMPLETE = "complete"    # 全部节点 resolve
TC_PARTIAL = "partial"      # 部分节点缺口（可答部分先行）
TC_CLARIFIED = "clarified"  # 无节点可答 / 无证据（如实上报 + 追问）


@dataclass
class TaskBudgets:
    """任务账本预算（源自 settings.rag_agent_task_*；P3 与会话账本叠加）。"""

    max_nodes: int = 4  # 单任务子查询节点上限（拆解器输出封顶）
    max_retries: int = 1  # 单节点缺口「改写重查」次数上限（缺口策略中心使用）
    max_inner_calls: int = 6  # 全任务内层触发上限（含节点与重查，防级联超支）
    token_budget: int = 0  # 任务级 LLM 累计 token 预算（0=不限）
    timeout_s: float = 120.0  # 单任务墙钟超时


@dataclass
class SessionLedger:
    """会话账本（P3，与任务账本叠加）：跨任务的累计记账，防「每任务各自达标、跨任务叠加超支」。

    治理口径（企业级「先扣后走」）：任务每次触发内层闭环前先检查会话余量
    （exhausted），触顶即终止并如实上报；任务实际消耗（内层触发数 + token）
    实时并入本账本，同会话内后续任务继承累计余量。同一 ledger 实例跨任务共享。
    """

    max_inner_calls: int = 20  # 会话级内层触发总上限（跨任务累计）
    token_budget: int = 0  # 会话级 token 累计上限（0=不限）
    inner_calls: int = 0  # 已累计内层触发数
    tokens: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})

    def remaining_inner(self) -> int:
        return max(0, self.max_inner_calls - self.inner_calls)

    def over_inner(self) -> bool:
        return self.inner_calls >= self.max_inner_calls

    def over_token(self) -> bool:
        if self.token_budget <= 0:
            return False
        return (self.tokens.get("prompt", 0) + self.tokens.get("completion", 0)) >= self.token_budget

    def exhausted(self) -> bool:
        return self.over_inner() or self.over_token()

    def merge(self, tokens: dict[str, int] | None = None, inner_calls: int = 1) -> None:
        """任务内层触发消耗并入会话账本（跨任务累计，由执行器每次内层触发后调用）。"""
        self.inner_calls += inner_calls
        t = tokens or {}
        self.tokens["prompt"] += t.get("prompt", 0)
        self.tokens["completion"] += t.get("completion", 0)


@dataclass
class TaskNode:
    """任务图中的一个子查询节点（拆解器产出）。"""

    id: str
    query: str
    deps: list[str] = field(default_factory=list)  # 依赖节点 id（须先执行）
    reason: str = ""
    state: str = NS_PLANNED
    retries: int = 0


@dataclass
class NodeResult:
    """单节点执行契约（内层闭环输出）：外层只读内层 verdict，不替内层判断证据够不够。"""

    node_id: str
    query: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    verdict: dict[str, Any] = field(default_factory=dict)
    missing_facts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    cost: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def state(self) -> str:
        return NS_RESOLVED if self.verdict.get("answerable") else NS_GAP


@dataclass
class TaskResult:
    """任务级结果（外层契约）：任务图 + 各节点契约 + 汇总口径。"""

    task_id: str
    query: str
    nodes: list[dict[str, Any]] = field(default_factory=list)  # [{id, query, deps, reason}]
    results: dict[str, NodeResult] = field(default_factory=dict)
    completion: str = TC_CLARIFIED
    evidence: list[dict[str, Any]] = field(default_factory=list)  # 黑板证据池（跨节点合并去重）
    gaps: list[dict[str, Any]] = field(default_factory=list)  # [{node_id, query, missing_facts, confidence, gap_type, action, note}]
    confidence: float = 0.0
    cost: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_count(self) -> int:
        return sum(1 for r in self.results.values() if r.state == NS_RESOLVED)

    @property
    def gap_count(self) -> int:
        return sum(1 for r in self.results.values() if r.state == NS_GAP)

    def to_dict(self) -> dict[str, Any]:
        """任务结果摘要（task_done 事件载荷，可 JSON 序列化）。"""
        return {
            "task_id": self.task_id,
            "query": self.query,
            "nodes": self.nodes,
            "completion": self.completion,
            "resolved": self.resolved_count,
            "gaps": self.gap_count,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "cost": self.cost,
            "gap_list": self.gaps,
        }


class TaskGraphState(TypedDict, total=False):
    """任务图状态机共享状态：节点规划产物 + 执行结果 + 黑板 + 事件（每任务独立实例）。"""

    task_id: str
    query: str
    nodes: list[dict[str, Any]]  # 拆解产物 [{id, query, deps, reason}]
    results: dict[str, NodeResult]  # node_id → 执行契约
    resolved: list[str]  # 已 resolve 节点 id（黑板：已确认事实；节点整表返回，不用归约器）
    gaps: list[dict[str, Any]]  # 缺口记录 [{node_id, query, missing_facts, confidence, gap_type, action, note}]
    seed: list[dict[str, Any]]  # 黑板证据池（跨节点累积，供后续节点 seed 复用）
    task_tokens: dict[str, int]  # 任务账本 token 记账（拆解 + 各节点内层累计）
    inner_calls: int  # 任务账本：内层触发次数
    outbox: Annotated[list[dict], operator.add]  # 任务图事件（task_plan/task_node/task_done，追加合并）
    started: float  # 墙钟起点（任务级成本核算）
    note: str  # 终止备注（预算耗尽/超时/依赖死锁等）
    result: TaskResult | None  # 终态任务结果
