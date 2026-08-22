"""multi-agent 模式：Orchestrator 通过工具调用 compute/analyze 子代理 → 汇总（create_agent 实现）。

- compute/analyze 均为独立 create_agent 子代理，经 convert_runnable_to_tool 包装为编排者工具；
- 编排者 create_agent 自带工具路由与循环，MultiAgentMiddleware 发射分派/完成事件，
  StreamEventsMiddleware 负责 thinking/message 与工具 HITL；
- 编排者与子代理都挂 ModelCallLimitMiddleware（轮数上限）：单轮模型调用超过 max_steps
  即抛 ModelCallLimitExceededError，运行器转为 done，防死循环；
- 子代理不持有 checkpointer（WorkerEventsMiddleware 不做 HITL 中断），审批收敛到编排者层。
"""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import convert_runnable_to_tool

from app.agents.middleware.events_mw import StreamEventsMiddleware, WorkerEventsMiddleware
from app.agents.middleware.multi_agent_mw import MultiAgentMiddleware

_COMPUTE_PROMPT = (
    "你是计算 Worker，负责数值计算与带计算的子任务。需要时调用计算工具，最后给出结论。"
    "若工具调用失败或返回错误，先修正参数或换一种方式重试，不要直接说工具不可用。"
)
_ANALYZE_PROMPT = "你是分析 Worker，负责逻辑分析、归纳与总结子任务。不调用工具，直接给出分析结论。"
_ORCHESTRATOR_PROMPT = (
    "你是编排者。根据任务调用 compute（计算）与 analyze（分析）两个 Worker，"
    "最后整合它们的结果给出完整的最终答案。若某 Worker 返回错误，可调整任务措辞后重新分派。"
)


def _worker_tool(worker, name, description):
    """把子代理包装为 {task: str} 工具：调用时以该任务驱动 worker 图执行。"""

    async def _run(task: str) -> str:
        result = await worker.ainvoke({"messages": [HumanMessage(task)]})
        last = result["messages"][-1]
        return str(getattr(last, "content", "") or "")

    return convert_runnable_to_tool(
        RunnableLambda(_run),
        name=name,
        description=description,
        arg_types={"task": str},
    )


def build_multi_agent_agent(llm, tools, emit, settings, checkpointer=None, harness=None):
    """构建 multi-agent 代理：编排者路由调用 compute/analyze 子代理并汇总。"""
    compute_tools = [t for t in tools if t.name == "calculator"]
    step_limit = max(1, settings.max_steps)

    compute_agent = create_agent(
        model=llm,
        tools=compute_tools,
        system_prompt=_COMPUTE_PROMPT,
        middleware=[
            WorkerEventsMiddleware(emit, harness=harness),
            ModelCallLimitMiddleware(run_limit=step_limit, exit_behavior="error"),
        ],
        checkpointer=None,
    )
    analyze_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=_ANALYZE_PROMPT,
        middleware=[
            WorkerEventsMiddleware(emit, harness=harness),
            ModelCallLimitMiddleware(run_limit=step_limit, exit_behavior="error"),
        ],
        checkpointer=None,
    )

    compute_tool = _worker_tool(compute_agent, "compute", "负责数值计算与带计算的子任务，参数为 task。")
    analyze_tool = _worker_tool(analyze_agent, "analyze", "负责逻辑分析、归纳与总结子任务，参数为 task。")

    return create_agent(
        model=llm,
        tools=[compute_tool, analyze_tool],
        system_prompt=_ORCHESTRATOR_PROMPT,
        middleware=[
            MultiAgentMiddleware(emit),
            StreamEventsMiddleware(emit, harness=harness),
            ModelCallLimitMiddleware(run_limit=step_limit, exit_behavior="error"),
        ],
        checkpointer=checkpointer,
    )
