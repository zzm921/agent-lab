"""Agent 护栏层（Harness / 驾驭工程）：为 Agent 提供保障性额外功能。

行业公式：Agent = Model + Harness。模型之外的可靠性工程均属 Harness；
本模块收敛本项目已有的护栏能力，供 AgentRunner（运行器）调用：
- 审批策略：approval_policy=always 时工具调用需 HITL 审批；高危工具（如命令执行）
  无论策略如何都强制 HITL（should_approve）
- 资源上限：迭代/递归上限（max_iterations → recursion_limit）
- 止损：stop() 取消进行中的运行，避免继续消耗 token
- 错误兜底：统一失败事件（error_event）
- 可观测：审批号与会话映射、工具调用统计
"""
from __future__ import annotations

import asyncio


# 强制 HITL 的工具：无论 approval_policy 如何，调用前都必须经人工审批（高危操作，如命令执行）
ALWAYS_APPROVE_TOOLS = {"run_command"}


def should_approve(approval_policy: str, tool_name: str) -> bool:
    """审批策略判定：approval_policy=always，或工具本身被标记为强制 HITL 时需审批。"""
    return approval_policy == "always" or tool_name in ALWAYS_APPROVE_TOOLS


class AgentHarness:
    """护栏层：持有审批策略、资源上限、运行注册表与统计，为运行器提供保障。"""

    def __init__(self, settings):
        self.settings = settings
        # approval_id → {session, interrupt_ids}：审批会话映射（HITL 恢复用）
        # interrupt_ids 记录该次审批覆盖的 LangGraph interrupt id（同一 superstep 可能有多个）
        self._approvals: dict[str, dict] = {}
        # session_id → 正在运行的后台图任务：供 stop() 取消，及时停止执行以节省 token
        self._tasks: dict[str, asyncio.Task] = {}
        # session_id → [工具调用计数]：同一轮内跨 resume 累计，避免多次审批时少算
        self._tool_counts: dict[str, list] = {}

    # --- 资源上限 ---
    def recursion_limit(self) -> int:
        """按最大迭代数推导递归上限（超限由 LangGraph 兜底报错）。"""
        return self.settings.max_iterations * 10 + 50

    # --- 止损：取消运行中的后台任务 ---
    def register_run(self, session_id: str, task: asyncio.Task) -> None:
        self._tasks[session_id] = task

    def release_run(self, session_id: str) -> None:
        self._tasks.pop(session_id, None)

    def stop(self, session_id: str) -> None:
        """取消指定会话正在运行的后台图任务（配合前端停止按钮，及时停止执行节省 token）。"""
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()

    # --- 可观测：工具调用统计（每轮重置，跨 resume 累计） ---
    def new_tool_counter(self, session_id: str) -> list:
        counter = [0]
        self._tool_counts[session_id] = counter
        return counter

    def tool_counter(self, session_id: str) -> list:
        return self._tool_counts.get(session_id) or [0]

    # --- 审批会话映射 ---
    def register_approval(self, approval_id: str, session_id: str, interrupt_ids=()) -> None:
        self._approvals[approval_id] = {"session": session_id, "interrupt_ids": list(interrupt_ids)}

    def resolve_approval(self, approval_id: str) -> str | None:
        entry = self._approvals.get(approval_id)
        return entry["session"] if entry else None

    def approval_interrupt_ids(self, approval_id: str) -> list:
        """该次审批对应的 LangGraph interrupt id 列表（用于 resume 恢复）。"""
        entry = self._approvals.get(approval_id)
        return entry["interrupt_ids"] if entry else []

    # --- 错误兜底：统一失败事件 ---
    @staticmethod
    def error_event(message: str, detail: str = "") -> dict:
        return {"type": "error", "message": message, "detail": detail}
