"""Agent 护栏层（Harness / 驾驭工程）：为 Agent 提供保障性额外功能。

行业公式：Agent = Model + Harness。模型之外的可靠性工程均属 Harness；
本模块收敛本项目已有的护栏能力，供 AgentRunner（运行器）调用：
- 审批策略：approval_policy=always 时工具调用需 HITL 审批；高危工具（如命令执行）
  无论策略如何都强制 HITL（should_approve）
- 资源上限：迭代/递归上限（max_iterations → recursion_limit）、工具调用次数上限
- 熔断：同一会话内工具连续失败达到阈值后熔断（half-open 冷却后放行探测）
- 止损：stop() 取消进行中的运行，避免继续消耗 token
- 错误兜底：统一失败事件（error_event）
- 可观测：审批号与会话映射、工具调用统计
"""
from __future__ import annotations

import asyncio
import json
import time


# 强制 HITL 的工具：无论 approval_policy 如何，调用前都必须经人工审批（高危操作，如命令执行）
ALWAYS_APPROVE_TOOLS = {"run_command"}


def should_approve(approval_policy: str, tool_name: str) -> bool:
    """审批策略判定：approval_policy=always，或工具本身被标记为强制 HITL 时需审批。"""
    return approval_policy == "always" or tool_name in ALWAYS_APPROVE_TOOLS


class AgentHarness:
    """护栏层：持有审批策略、资源上限、运行注册表、统计与故障注入，为运行器提供保障。"""

    def __init__(self, settings):
        self.settings = settings
        # approval_id → {session, interrupt_ids}：审批会话映射（HITL 恢复用）
        # interrupt_ids 记录该次审批覆盖的 LangGraph interrupt id（同一 superstep 可能有多个）
        self._approvals: dict[str, dict] = {}
        # session_id → 正在运行的后台图任务：供 stop() 取消，及时停止执行以节省 token
        self._tasks: dict[str, asyncio.Task] = {}
        # session_id → [工具调用计数]：同一轮内跨 resume 累计，避免多次审批时少算
        self._tool_counts: dict[str, list] = {}
        # 熔断：session_id → tool_name → {state, failures, open_until}
        # state ∈ closed（正常）| open（熔断，拒绝调用）| half_open（冷却结束，放行一次探测）
        self._circuits: dict[str, dict[str, dict]] = {}
        # 故障注入：tool_name → mode（error | timeout）。仅供验证熔断机制用，
        # 由工具处理节点（而非工具类本身）作为钩子读取并模拟失败
        self._faults: dict[str, str] = {}

    # --- 故障注入：验证熔断机制的钩子（不侵入工具类） ---
    def set_fault(self, tool_name: str, mode: str) -> None:
        """设置工具故障注入：error=模拟报错，timeout=模拟超时，off/空=恢复正常。"""
        if mode in ("off", "", "none"):
            self._faults.pop(tool_name, None)
        elif mode in ("error", "timeout"):
            self._faults[tool_name] = mode

    def fault_mode(self, tool_name: str) -> str | None:
        return self._faults.get(tool_name)

    def faults(self) -> dict[str, str]:
        return dict(self._faults)

    def fault_message(self, tool_name: str) -> str | None:
        """故障注入钩子：命中注入模式时返回模拟失败信息，否则返回 None。

        工具处理节点（events_mw._execute_tool_call / tools.runner.make_tools_node）
        在真正执行前调用；返回非空即短路为失败，并计入熔断失败次数。
        """
        mode = self._faults.get(tool_name)
        if mode == "error":
            return f"[故障注入] 工具 {tool_name} 模拟报错"
        if mode == "timeout":
            return f"[故障注入] 工具 {tool_name} 模拟超时"
        return None

    # --- 资源上限 ---
    def recursion_limit(self) -> int:
        """按最大迭代数推导递归上限（超限由 LangGraph 兜底报错）。"""
        return self.settings.max_iterations * 10 + 50

    def tool_calls_limit(self) -> int:
        """单轮最多工具调用次数（达到后拒绝后续调用）。"""
        return int(getattr(self.settings, "tool_max_calls", 20))

    def tool_calls_exceeded(self, session_id: str) -> bool:
        """本轮工具调用是否已达上限。"""
        return self.tool_counter(session_id)[0] >= self.tool_calls_limit()

    # --- 熔断：同一会话内“同一工具+同一参数”连续失败达到阈值后，短路该次调用 ---
    # 熔断键为「工具名+参数签名」：模型换参数重试视为新的调用（key 不同）始终放行，
    # 仅对相同参数（键不变）的重复失败熔断，避免模型只能回「工具不可用」而无法换参重试。
    def _fault_key(self, tool_name: str, args) -> str:
        try:
            sig = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            sig = str(args)
        return f"{tool_name}:{sig}"

    def _circuit(self, session_id: str, key: str) -> dict:
        by_key = self._circuits.setdefault(session_id, {})
        return by_key.setdefault(key, {"state": "closed", "failures": 0, "open_until": 0.0})

    def circuit_allows(self, session_id: str, tool_name: str, args) -> bool:
        """熔断判定：closed 放行；open 冷却中拒绝“相同参数”的调用；冷却结束放行一次探测（half-open）。
        换参数调用（key 不同）始终放行，便于模型修正参数后重试。"""
        entry = self._circuit(session_id, self._fault_key(tool_name, args))
        if entry["state"] == "closed":
            return True
        if entry["state"] == "open":
            if time.time() >= entry["open_until"]:
                entry["state"] = "half_open"
                return True  # 冷却结束，放行一次探测
            return False
        # half_open：探测已放行（或进行中），不再重复放行，回到 open 重新冷却
        entry["state"] = "open"
        entry["open_until"] = time.time() + int(getattr(self.settings, "circuit_cooldown", 30))
        return False

    def record_tool_success(self, session_id: str, tool_name: str, args) -> None:
        """工具执行成功：关闭对应调用键的熔断并清零失败计数。"""
        entry = self._circuit(session_id, self._fault_key(tool_name, args))
        entry["state"] = "closed"
        entry["failures"] = 0

    def record_tool_failure(self, session_id: str, tool_name: str, args) -> None:
        """工具执行失败：累计失败次数，达到阈值（或探测失败）即熔断并进入冷却。"""
        entry = self._circuit(session_id, self._fault_key(tool_name, args))
        entry["failures"] += 1
        threshold = int(getattr(self.settings, "circuit_fail_threshold", 3))
        if entry["state"] == "half_open" or entry["failures"] >= threshold:
            entry["state"] = "open"
            entry["failures"] = 0
            entry["open_until"] = time.time() + int(getattr(self.settings, "circuit_cooldown", 30))

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
