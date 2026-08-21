"""ReAct 模式：思考(Thought)-行动(Action)-观察(Observation) 循环（create_agent 实现）。

工具循环与绑定由 create_agent 内建；事件流（thinking/message/tool_*）与 HITL 审批由
StreamEventsMiddleware 统一负责，取代旧手写 agent ⇄ tools 图。
"""
from langchain.agents import create_agent

from app.agents.middleware.events_mw import StreamEventsMiddleware


def build_react_agent(llm, tools, emit, settings, checkpointer=None, harness=None):
    """构建 ReAct 代理：模型 ⇄ 工具循环 + thinking/message 事件 + 工具 HITL。"""
    return create_agent(
        model=llm,
        tools=list(tools),
        middleware=[StreamEventsMiddleware(emit, harness=harness)],
        checkpointer=checkpointer,
    )
