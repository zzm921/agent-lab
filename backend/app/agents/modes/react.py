"""ReAct 模式：思考(Thought)-行动(Action)-观察(Observation) 循环（create_agent 实现）。

工具循环与绑定由 create_agent 内建；事件流（thinking/message/tool_*）与 HITL 审批由
StreamEventsMiddleware 统一负责，取代旧手写 agent ⇄ tools 图；
轮数上限由 AskFreeCallLimitMiddleware 强制：单轮模型调用（思考/工具回合）超过 max_steps
即抛 ModelCallLimitExceededError，运行器转为 done「已达到最大轮数上限」，防死循环。
其中模型只调用 ask_user（澄清提问）的轮次不计入上限——提问/回答是 HITL 交互，不消耗执行预算。
"""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

from app.agents.middleware.events_mw import StreamEventsMiddleware


def _only_ask_user(msg) -> bool:
    calls = getattr(msg, "tool_calls", None) or []
    return bool(calls) and all(c.get("name") == "ask_user" for c in calls)


class AskFreeCallLimitMiddleware(ModelCallLimitMiddleware):
    """ModelCallLimitMiddleware 变体：模型本轮只调用 ask_user 时不递增轮数计数。

    after_model 作为图中节点接收模型调用后的完整 state，最后一条消息即本次模型输出；
    若其工具调用全部是 ask_user，判定为澄清轮，跳过计数（返回 None 不产生状态更新）。
    """

    def after_model(self, state, runtime):
        msgs = state.get("messages") or []
        if msgs and _only_ask_user(msgs[-1]):
            return None
        return super().after_model(state, runtime)

    async def aafter_model(self, state, runtime):
        return self.after_model(state, runtime)


def build_react_agent(llm, tools, emit, settings, checkpointer=None, harness=None):
    """构建 ReAct 代理：模型 ⇄ 工具循环 + thinking/message 事件 + 工具 HITL + 轮数上限。"""
    return create_agent(
        model=llm,
        tools=list(tools),
        middleware=[
            StreamEventsMiddleware(emit, harness=harness),
            AskFreeCallLimitMiddleware(run_limit=max(1, settings.max_steps), exit_behavior="error"),
        ],
        checkpointer=checkpointer,
    )
