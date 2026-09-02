"""共享工具执行节点：统一执行工具、推送事件，并按审批策略触发 HITL 中断。"""
from __future__ import annotations

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from app.agents.harness import should_approve
from app.core.errors import RetryableToolError
from app.core.events import event
from app.security import is_untrusted_tool, wrap_untrusted
from app.tools.retry import format_tool_error, invoke_with_retry


def _wrap_untrusted_tool_output(name: str, output: str, settings) -> str:
    """不可信外部内容工具（网页/命令/记忆）返回经来源分级包装后再给模型，防间接注入。

    只包装「外部内容来源」工具的成功返回（web_search / run_command / memory_recall），
    内部工具（calculator 等）原样透传，避免污染正常工具结果。
    """
    mark = bool(getattr(settings, "security_enabled", True) and getattr(settings, "mark_untrusted", True))
    if not mark or not is_untrusted_tool(name):
        return str(output)
    return wrap_untrusted(str(output), name)


def make_tools_node(tools, emit, harness=None):
    """构建 LangGraph tools 节点。审批策略通过 config['configurable']['approval_policy'] 传入；
    任一步骤工具失败（异常/未注入/用户拒绝）时返回 step_failed=True，供 plan-execute 的 replan 决策使用。
    harness（可选）提供熔断、工具次数上限与 Agent 层重试上限等护栏。"""
    by_name = {t.name: t for t in tools}

    async def tools_node(state, config):
        policy = config["configurable"]["approval_policy"]
        session_id = config["configurable"]["thread_id"]
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        if not calls:
            return {"messages": []}

        # 存在故障注入的工具：短路审批（与 events_mw 一致），由 per-call 循环按注入类型处理
        has_fault = harness is not None and any(harness.fault_spec(c["name"]) for c in calls)
        # 任一本步工具需要审批（approval_policy=always 或强制 HITL 工具）即整批审批
        
        if any(should_approve(policy, c["name"]) for c in calls):
            payload = [{"name": c["name"], "args": c.get("args", {}), "id": c.get("id")} for c in calls]
            decision = interrupt({"tool_calls": payload})
            action = decision.get("action", "approve")
            if action == "reject":
                results = []
                for c in calls:
                    emit(event("tool_end", tool=c["name"], args=c.get("args"), result="用户拒绝了该工具调用", success=False))
                    results.append(
                        ToolMessage(
                            content="用户拒绝了该工具调用，请改用其它方式或询问用户。",
                            tool_call_id=c["id"],
                        )
                    )
                return {"messages": results, "step_failed": True}
            modified = decision.get("modified_args") or {}
            effective = []
            for c in calls:
                if c.get("id") in modified:
                    effective.append({**c, "args": modified[c.get("id")]})
                elif c["name"] in modified:
                    # 兜底：部分模型未回传工具调用 id 时前端按名称回填，按名称匹配保证修改生效
                    effective.append({**c, "args": modified[c["name"]]})
                else:
                    effective.append(c)
        elif has_fault:
            effective = calls
        else:
            effective = calls

        results = []
        any_failed = False
        settings = getattr(harness, "settings", None) if harness is not None else None
        for c in effective:
            name = c["name"]
            args = c.get("args", {})
            fault = harness.fault_spec(name) if harness is not None else None
            # 护栏：熔断 / 工具调用次数上限 / Agent 层重试上限。命中直接短路，不执行。
            # 熔断按「工具+参数」计：相同参数重复失败才拦截，换参数重试始终放行
            if harness is not None:
                if not harness.circuit_allows(session_id, name, args):
                    msg = f"工具 {name} 在当前参数下已连续失败触发熔断保护，请更换参数重试或改用其它工具。"
                    emit(event("tool_start", tool=name, args=args))
                    emit(event("tool_end", tool=name, args=args, result=msg, success=False))
                    results.append(ToolMessage(content=msg, tool_call_id=c["id"]))
                    continue
                if harness.tool_calls_exceeded(session_id):
                    msg = f"本轮工具调用已达上限（{harness.tool_calls_limit()} 次），已停止执行。"
                    emit(event("tool_start", tool=name, args=args))
                    emit(event("tool_end", tool=name, args=args, result=msg, success=False))
                    results.append(ToolMessage(content=msg, tool_call_id=c["id"]))
                    continue
                # 故障注入（验证两层重试与熔断机制用，短路审批）：永久错误（error/business/400/401/403/404）
                # → 不直接重试，错误直接返回给模型思考后重试；瞬时错误（timeout/conn_reset/dns/429/5xx）
                # → 下方执行时抛 RetryableToolError 进入工具层透明重试
                if fault is not None and not fault["retryable"]:
                    emit(event("tool_start", tool=name, args=args))
                    emit(event("tool_end", tool=name, args=args, result=fault["message"], success=False))
                    harness.record_tool_failure(session_id, name, args)
                    results.append(ToolMessage(content=fault["message"], tool_call_id=c["id"]))
                    continue
                # Agent 层重试上限：同一工具连续失败达到上限后，提示模型改用其它工具
                if harness.tool_exhausted(session_id, name):
                    msg = f"工具 {name} 已连续失败 {harness.agent_retry_limit()} 次，请停止使用该工具，改用其它工具或向用户说明。"
                    emit(event("tool_start", tool=name, args=args))
                    emit(event("tool_end", tool=name, args=args, result=msg, success=False))
                    results.append(ToolMessage(content=msg, tool_call_id=c["id"]))
                    continue
            emit(event("tool_start", tool=name, args=args))
            tool = by_name.get(name)
            if tool is None:
                msg = format_tool_error(name, LookupError(f"工具 {name} 未注入"))
                emit(event("tool_end", tool=name, args=args, result=msg, success=False))
                results.append(ToolMessage(content=msg, tool_call_id=c["id"]))
                continue

            async def _run():
                if fault is not None:  # 瞬时故障注入：每次尝试均模拟瞬时失败，直至直接重试耗尽
                    raise RetryableToolError(fault["message"])
                return await tool.ainvoke(args)

            # 工具层透明重试（仅瞬时错误直接重试，指数退避+抖动；参数/业务错误不重试直接返回）
            output, success, error, retries = await invoke_with_retry(_run, name, settings, emit)
            if success:
                if harness is not None:
                    harness.record_tool_success(session_id, name, args)
                emit(event("tool_end", tool=name, args=args, result=str(output), success=True))
                results.append(ToolMessage(content=_wrap_untrusted_tool_output(name, str(output), settings), tool_call_id=c["id"]))
            else:
                any_failed = True
                if harness is not None:
                    harness.record_tool_failure(session_id, name, args)
                msg = format_tool_error(name, error, retried=retries)
                emit(event("tool_end", tool=name, args=args, result=msg, success=False))
                results.append(ToolMessage(content=msg, tool_call_id=c["id"]))
        return {"messages": results, "step_failed": any_failed}

    return tools_node
