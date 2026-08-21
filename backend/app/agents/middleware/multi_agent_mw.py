"""MultiAgentMiddleware：multi-agent 编排者的事件发射中间件。

- awrap_tool_call：当编排者调用 compute/analyze 子代理工具时，
  发射 orchestrator dispatch、worker dispatch / done 事件，随后透传给内层中间件执行。
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware

_WORKER_NAMES = ("compute", "analyze")


class MultiAgentMiddleware(AgentMiddleware):
    """为编排者补充 agent_event 分派/完成事件（worker 名固定为 compute/analyze）。"""

    def __init__(self, emit):
        self._emit = emit

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if name not in _WORKER_NAMES:
            return await handler(request)
        args = request.tool_call.get("args", {})
        task = args.get("task", "")
        self._emit({"type": "agent_event", "worker": "orchestrator", "status": "dispatch", "task": task})
        self._emit({"type": "agent_event", "worker": name, "status": "dispatch", "task": task})
        result = await handler(request)
        text = str(getattr(result, "content", "") or "")
        self._emit({"type": "agent_event", "worker": name, "status": "done", "result": text})
        return result
