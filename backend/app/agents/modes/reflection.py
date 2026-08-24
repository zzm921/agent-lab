"""reflection 模式：LangGraph StateGraph 原生编排（generator ⇄ tools → critic → 条件循环）。

- generator：生成草稿（首稿下发 message 事件）或依据评审反馈修订（修订稿下发 revise 事件）；
  绑定用户启用的工具（如联网检索），模型请求工具时经 tools 节点执行后回到本节点继续生成，
  直到产出不含工具调用的完整答案；
- tools：复用 make_tools_node（工具事件 + HITL 审批 + 异常兜底）；
- critic：流式评审——思考走 thinking 事件、评审文本走 critique 增量事件（前端流式展示评审过程）；
  通过与否由评审文本解析（_judge_text：无 / 【PASS】开头即通过，显式 FAIL 不通过）；
- should_continue：评审通过（passed）或达到最大迭代（max_iter）即结束，
  否则回到 generator 继续修订（两个终止条件共同防止死循环）；
- 轮数上限（max_steps）：generator 累计模型调用/工具回合数（steps），超过即置 stopped=max_steps
  不再调用模型并结束，防「反复请求工具从不产出草稿」的死循环。

事件流：message / reflect（stage、critique 增量）/ revise / tool_start / tool_end。
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.middleware.events_mw import stream_model_call
from app.tools.runner import make_tools_node

_GENERATOR_SYSTEM = "你是专业的 AI 助手，请直接、准确地回答用户的问题。"
_GENERATOR_PROMPT = "请回答下面的问题：{query}"
_REVISE_PROMPT = (
    "基于之前回答的缺陷修改答案，输出优化后的完整回答。\n"
    "原始问题：{query}\n上一轮回答：{draft}\n评审反馈（需要改进点）：{critique}"
)
_CRITIC_SYSTEM = (
    "你是严格的质量评审专家。评估回答是否完整、准确、无遗漏。\n"
    "若回答的有缺陷和遗漏、输出可落地的改进建议（不要只说『回答不够好』）。"
    "若回答已足够好、无需修改，请直接以「【PASS】」开头输出通过结论；\n"
)


class ReflectionState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    draft: str       # 当前草稿/修订稿
    critique: str    # 最新评审意见
    iteration: int   # 已完成的生成轮次（草稿 + 各轮修订）
    max_iter: int    # 最大允许生成轮次（防死循环）
    steps: int       # 累计模型调用/工具回合数（轮数上限判定）
    stopped: str     # 结束原因："max_steps" 表示达到轮数上限
    passed: bool     # 评审是否通过
    step_failed: bool  # 工具执行是否失败（tools 节点返回，供后续决策使用）


def _latest_user_text(state) -> str:
    """取最近一条用户消息作为当前任务（多轮会话取最新输入）。"""
    for m in reversed(state.get("messages") or []):
        if getattr(m, "type", None) == "human":
            return m.content
    return ""


def _is_fresh_turn(state) -> bool:
    """最近一条消息为 user 即新回合：重置循环状态，避免多轮会话残留 iteration/passed。"""
    msgs = state.get("messages") or []
    return bool(msgs) and getattr(msgs[-1], "type", None) == "human"


def _judge_text(text: str) -> bool:
    """流式自由文本评审的通过判定：无 / 空 / 以【PASS】或 PASS 开头视为通过；显式 FAIL 视为不通过。"""
    t = (text or "").strip()
    if not t or t == "无":
        return True
    if "【FAIL】" in t or "【不通过】" in t:
        return False
    head = t.splitlines()[0]
    return head.startswith("【PASS】") or head.startswith("PASS")


def build_reflection_agent(generator_llm, critic_llm, tools, emit, settings, checkpointer=None, harness=None):
    """构建 reflection 代理：generator ⇄ tools → critic → 条件循环（评审通过/达上限/达轮数上限即结束）。

    generator_llm：生成草稿/修订稿（流式，产出 message/revise 事件）；
    critic_llm：质量评审（严格、低随机，产出 critique 事件）。
    """
    max_iter = max(1, settings.max_iterations)
    max_steps = max(1, settings.max_steps)
    tool_list = list(tools)
    tools_node = make_tools_node(tool_list, emit, harness=harness)

    async def generator(state):
        fresh = _is_fresh_turn(state)
        iteration = 0 if fresh else (state.get("iteration") or 0)
        draft = "" if fresh else (state.get("draft") or "")
        critique = "" if fresh else (state.get("critique") or "")
        query = _latest_user_text(state)
        # 已有评审反馈且已完成过草稿 → 修订阶段；否则为草稿阶段
        revising = iteration > 0 and bool(critique)

        # 轮数上限：累计模型调用/工具回合数，超过即不再调用模型、标记结束（防反复请求工具死循环）
        steps = (0 if fresh else (state.get("steps") or 0)) + 1
        if steps > max_steps:
            return {"steps": steps, "stopped": "max_steps"}

        msgs = list(state.get("messages") or [])
        base = ""
        if msgs and getattr(msgs[0], "type", None) == "system":
            base = str(msgs[0].content)
            msgs = msgs[1:]
        # 工具回合（上一步刚执行完工具）：携带完整历史继续生成，不再追加任务/修订提示
        if not (msgs and getattr(msgs[-1], "type", None) == "tool"):
            if revising:
                prompt = _REVISE_PROMPT.format(query=query, draft=draft, critique=critique)
            else:
                prompt = _GENERATOR_PROMPT.format(query=query)
            msgs = msgs + [HumanMessage(prompt)]

        # 首稿 → message 事件；修订稿 → revise 事件（前端按修订稿样式展示）
        msg = await stream_model_call(
            generator_llm, msgs, emit,
            tools=tool_list,
            system_prompt=base if base else _GENERATOR_SYSTEM,
            output_event="revise" if revising else "message",
        )
        if getattr(msg, "tool_calls", None):
            # 模型请求工具：路由到 tools 执行，完成后回到本节点继续生成
            return {"messages": [msg], "steps": steps}
        if not revising:
            emit({"type": "reflect", "stage": "draft"})
        return {
            "messages": [msg],
            "draft": msg.content or "",
            "iteration": iteration + 1,
            "passed": False,
            "steps": steps,
        }

    async def critic(state):
        query = _latest_user_text(state)
        draft = state.get("draft") or ""
        # 流式评审：思考 → thinking 事件，评审文本 → critique 增量事件（前端流式展示评审过程）
        msg = await stream_model_call(
            critic_llm,
            [HumanMessage(f"原始用户问题：{query}\n当前回答草稿：{draft}\n\n请输出评审结论。")],
            emit,
            system_prompt=_CRITIC_SYSTEM,
            output_event="critique",
        )
        text = msg.content or ""
        return {"critique": text, "passed": _judge_text(text)}

    def after_generator(state) -> str:
        # 达到轮数上限 → 直接结束（不进入工具/评审）
        if state.get("stopped") == "max_steps":
            return "end"
        # 最近一条模型消息带工具调用 → 去 tools 执行；否则草稿/修订完成 → 评审
        msgs = state.get("messages") or []
        if msgs and getattr(msgs[-1], "tool_calls", None):
            return "tools"
        return "critic"

    def should_continue(state) -> str:
        # 终止条件1：评审通过；终止条件2：达到最大迭代（防死循环）
        if state.get("passed") or (state.get("iteration") or 0) >= max_iter:
            return "end"
        return "generator"

    builder = StateGraph(ReflectionState)
    builder.add_node("generator", generator)
    builder.add_node("tools", tools_node)
    builder.add_node("critic", critic)
    builder.add_edge(START, "generator")
    builder.add_conditional_edges(
        "generator",
        after_generator,
        {"tools": "tools", "critic": "critic", "end": END},
    )
    builder.add_edge("tools", "generator")
    builder.add_conditional_edges(
        "critic",
        should_continue,
        {"generator": "generator", "end": END},
    )
    return builder.compile(checkpointer=checkpointer)
