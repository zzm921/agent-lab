"""ReflectionMiddleware：生成-反思-修订 的 create_agent 中间件实现。

在单次模型调用（awrap_model_call）内完成 草稿 → 批评 → 修订 → 再批评 循环，
直到批评为「无」或达到最大迭代轮次；工具循环不参与该模式。
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.events import emit_text

_CRITIC_PROMPT = "你是严格的质量评审员。评估下面的草稿，指出不足并给出改进建议；若已足够好只回复『无』。"
_REVISE_PROMPT = "根据评审意见修订答案，输出完整的修订版。"


class ReflectionMiddleware(AgentMiddleware):
    """一次模型调用内完成 reflection 全流程，返回最终（修订后）消息。"""

    def __init__(self, llm, emit, settings):
        self._llm = llm
        self._emit = emit
        self._max_iterations = settings.max_iterations

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        draft = response.result[0]
        emit_text(self._emit, "message", draft.content or "")
        self._emit({"type": "reflect", "stage": "draft"})
        current = draft
        for _ in range(self._max_iterations):
            critique = (await self._llm.ainvoke([SystemMessage(_CRITIC_PROMPT), HumanMessage(current.content or "")])).content or ""
            self._emit({"type": "reflect", "critique": critique})
            if not critique.strip() or critique.strip() == "无":
                break
            revised = await self._llm.ainvoke([SystemMessage(_REVISE_PROMPT), HumanMessage(f"草稿：{current.content}\n评审意见：{critique}")])
            emit_text(self._emit, "revise", revised.content or "")
            current = revised
        return ModelResponse(result=[current], structured_response=response.structured_response)
