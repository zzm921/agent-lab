"""能力热插拔：按启用的能力列表组装工具集。"""
from app.agents.tools_builder import build_tools
from app.capabilities.mcp import McpManager
from app.capabilities.registry import CapabilityRegistry


def test_build_tools_by_enabled(registry):
    tools = build_tools(registry, ["calculator", "time_now", "not_exist"], "s1")
    names = {t.name for t in tools}
    assert names == {"calculator", "time_now"}


def test_build_tools_empty(registry):
    assert build_tools(registry, [], "s1") == []


def test_build_tools_memory_not_in_toolset(registry):
    """记忆是系统前置能力（常驻注入/主动召回/轮末巩固），不再暴露为工具。"""
    assert build_tools(registry, ["memory"], "s1") == []


def test_build_tools_skips_unavailable(settings, sessions):
    registry = CapabilityRegistry(settings, sessions, McpManager("{}"), None, None)
    registry.list()
    assert build_tools(registry, ["rag", "calculator"], "s1")[0].name == "calculator"


def test_build_tools_dedupe(registry):
    tools = build_tools(registry, ["calculator", "calculator", "time_now"], "s1")
    names = [t.name for t in tools]
    assert names.count("calculator") == 1
