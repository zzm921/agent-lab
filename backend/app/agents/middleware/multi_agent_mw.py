"""MultiAgentMiddleware：multi-agent 编排者的事件发射中间件（任务单协议）。

- awrap_tool_call：当编排者调用 worker 子代理工具时（入参为任务单数组
  {"tasks": [{id, role, task, context, deps}]}）：
  1. 任务数上限截断（防拆解爆炸）；
  2. 发射 plan 事件（todo 视图：id/desc/deps/status/assignee），前端顶部计划区展示；
  3. 逐个 task 发射 agent_event dispatch（携带 task_id），随后执行 handler；
  4. 结果按 task id 归位：解析 worker 返回的 JSON 结果数组，逐个发射 agent_event done；
  5. 全部完成后发射 plan done 事件。
- 同一轮多个 worker 工具调用由 create_agent 天然并行执行；批内/跨批依赖由 worker 端
  分派治理处理（拓扑排序 + 完成度校验 + 前置结果注入），中间件不做波次调度。
- 结果条目带 error 字段（依赖未满足/正在执行中/重复 id/依赖环）时，该任务标记 failed
  并携带错误文本，供编排者据错误内容修正后重新分派。
"""

from __future__ import annotations

import json

from langchain.agents.middleware import AgentMiddleware

from app.tools.retry import format_tool_error

_WORKER_NAMES = ("worker",)
MAX_TASKS = 8  # 单轮任务单数量上限：防拆解爆炸（护栏，后续可配置化）


class MultiAgentMiddleware(AgentMiddleware):
    """为编排者补充 agent_event 分派/完成事件（worker 名固定为 worker）。"""

    def __init__(self, emit, settings=None):
        self._emit = emit
        self._settings = settings  # 预留：任务数上限后续可配置化

    @staticmethod
    def _todo_items(tasks: list[dict], worker: str) -> list[dict]:
        """任务单 → todo 视图：items 带 assignee（执行角色），状态统一 dispatched。"""
        items = []
        for t in tasks:
            items.append(
                {
                    "id": t.get("id"),
                    "desc": t.get("task", ""),
                    "deps": t.get("deps") or [],
                    "status": "dispatched",
                    "assignee": t.get("role") or worker,
                }
            )
        return items

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if name not in _WORKER_NAMES:
            return await handler(request)
        args = request.tool_call.get("args", {})
        tasks = (args.get("tasks") or [])[:MAX_TASKS]  # 任务数上限：截断超限任务单
        if not tasks:
            # 协议未按任务单调用（tasks 缺失/为空）：退化为原裸执行，事件无 task 归属
            self._emit({"type": "agent_event", "worker": name, "status": "dispatch"})
            result = await handler(request)
            self._emit({"type": "agent_event", "worker": name, "status": "done", "result": str(getattr(result, "content", "") or "")})
            return result
        # todo 事件：items 为任务单视图（dispatched），供前端顶部计划区展示
        self._emit(
            {
                "type": "plan",
                "items": self._todo_items(tasks, name),
                "current_step": 0,
                "status": "created",
            }
        )
        # 逐个任务派发：agent_event 携带 task_id，事件与任务单按 id 对齐
        for t in tasks:
            self._emit(
                {
                    "type": "agent_event",
                    "worker": name,
                    "status": "dispatch",
                    "task_id": t.get("id"),
                    "task": t.get("task", ""),
                }
            )
        done_items = [dict(t) for t in tasks]
        try:
            result = await handler(request)
        except Exception as exc:
            # 工具执行异常（worker 失败/轮数超限等）：逐任务标记 failed 并携带错误文本，
            # 再向上抛由外层 StreamEventsMiddleware 的工具层错误处理兜底（tool_end 由其发射）。
            msg = format_tool_error(name, exc)
            for t in tasks:
                tid = t.get("id")
                self._emit(
                    {
                        "type": "agent_event",
                        "worker": name,
                        "status": "failed",
                        "task_id": tid,
                        "result": msg,
                    }
                )
                for item in done_items:
                    if item["id"] == tid:
                        item["status"] = "failed"
                        item["result"] = msg
                        break
            self._emit({"type": "plan", "items": done_items, "current_step": len(done_items), "status": "done"})
            raise
        text = str(getattr(result, "content", "") or "")
        # 按 id 归位：worker 返回 JSON 结果数组 [{"id", "role", "result"}]；
        # 解析失败（工具执行异常/协议外返回）时标记 failed 并带错误文本，
        # 前端子代理卡片显示失败状态与原因（工具卡片已隐藏，失败信号不能丢）。
        try:
            results = json.loads(text)
            parsed = True
        except (ValueError, TypeError):
            results = []
            parsed = False
        by_id = {r.get("id"): r for r in (results or [])}
        for t in tasks:
            tid = t.get("id")
            entry = by_id.get(tid) or {}
            # 任务级失败：解析失败（工具异常/协议外返回）或条目带 error 字段（分派治理失败：
            # 依赖未满足/正在执行中/重复 id/依赖环）→ 该任务标记 failed 并携带错误文本
            has_error = bool(entry.get("error"))
            status = "failed" if (not parsed or has_error) else "done"
            payload = text if not parsed else (entry.get("error") or entry.get("result") or "")
            self._emit(
                {
                    "type": "agent_event",
                    "worker": name,
                    "status": status,
                    "task_id": tid,
                    "result": payload,
                }
            )
            for item in done_items:
                if item["id"] == tid:
                    item["status"] = status
                    item["result"] = payload
                    break
        self._emit({"type": "plan", "items": done_items, "current_step": len(done_items), "status": "done"})
        return result
