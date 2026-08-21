"""工具组装：按启用的能力 id 从注册表解析出 LangChain 工具集（能力热插拔核心）。"""


def build_tools(registry, enabled: list[str], session_id: str, emit=None) -> list:
    """根据 enabled 能力列表组装工具集；不可用/未注入的能力跳过，按名称去重。"""
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
    return result
