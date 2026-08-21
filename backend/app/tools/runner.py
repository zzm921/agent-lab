"""共享工具执行节点：统一执行工具、推送事件，并按审批策略触发 HITL 中断。"""
from __future__ import annotations

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from app.core.events import event


def make_tools_node(tools, emit):
    """构建 LangGraph tools 节点。审批策略通过 config['configurable']['approval_policy'] 传入；
    任一步骤工具失败（异常/未注入/用户拒绝）时返回 step_failed=True，供 plan-execute 的 replan 决策使用。"""
    by_name = {t.name: t for t in tools}

    async def tools_node(state, config):
        policy = config["configurable"]["approval_policy"]
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        if not calls:
            return {"messages": []}

        if policy == "always":
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
        else:
            effective = calls

        results = []
        any_failed = False
        for c in effective:
            emit(event("tool_start", tool=c["name"], args=c.get("args", {})))
            try:
                tool = by_name.get(c["name"])
                if tool is None:
                    raise LookupError(f"工具 {c['name']} 未注入")
                output = await tool.ainvoke(c.get("args", {}))
                success = True
            except Exception as exc:
                output = f"工具执行失败：{exc}"
                success = False
                any_failed = True
            emit(event("tool_end", tool=c["name"], args=c.get("args", {}), result=str(output), success=success))
            results.append(ToolMessage(content=str(output), tool_call_id=c["id"]))
        return {"messages": results, "step_failed": any_failed}

    return tools_node
