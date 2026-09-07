"""工具组装：按启用的能力 id 从注册表解析出 LangChain 工具集（能力热插拔核心）。"""
from app.tools.ask_user import ask_user


def build_tools(registry, enabled: list[str], session_id: str, emit=None) -> list:
    """根据 enabled 能力列表组装工具集；不可用/未注入的能力跳过，按名称去重。

    ask_user（HITL 澄清）不属于能力目录，始终注入：模型在缺少用户私有信息时可
    主动调用提问，任何模式/能力组合下都可用。
    """
    tools: list = []
    for cap_id in enabled:
        cap = registry.get(cap_id)
        if cap is None or cap.get("availability") != "available":
            continue
        tool = registry.tool_for(cap_id, session_id, emit)
        if tool is None:
            continue
        if isinstance(tool, list):
            tools.extend(tool)
        else:
            tools.append(tool)
    seen: set[str] = set()
    result: list = []
    for t in tools:
        if t.name not in seen:
            seen.add(t.name)
            result.append(t)
    if ask_user.name not in seen:
        seen.add(ask_user.name)
        result.append(ask_user)  # 始终注入，不依赖能力选配
    return result
