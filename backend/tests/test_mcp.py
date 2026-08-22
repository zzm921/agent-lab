"""MCP 集成测试：工具发现成功与连接失败标记「不适配」；开关默认关闭、页面点选开启。"""
from app.capabilities.mcp import McpManager


class FakeTool:
    name = "tool_a"
    description = "一个假的 MCP 工具"


async def test_mcp_disabled_no_discover(monkeypatch):
    """默认关闭：开关未开启时 discover 不连接，能力目录为空。"""
    mcp = McpManager('{"srv": {"command": "x"}}', enabled=False)

    async def boom(name, conf):
        raise AssertionError("开关未开不应连接")

    monkeypatch.setattr(mcp, "_load_tools", boom)
    await mcp.discover()
    assert mcp.capabilities == []
    assert mcp.enabled is False


async def test_mcp_discover_success(monkeypatch):
    mcp = McpManager('{"srv": {"command": "x"}}', enabled=True)

    async def fake_load(name, conf):
        return [FakeTool()]

    monkeypatch.setattr(mcp, "_load_tools", fake_load)
    await mcp.discover()
    assert mcp.capabilities[0]["id"] == "srv:tool_a"
    assert mcp.capabilities[0]["source"] == "mcp"
    assert mcp.capabilities[0]["server"] == "srv"
    assert mcp.capabilities[0]["availability"] == "available"
    assert isinstance(mcp.tool("srv:tool_a"), FakeTool)


async def test_mcp_discover_failure_marks_unavailable(monkeypatch):
    mcp = McpManager('{"srv": {"command": "x"}}', enabled=True)

    async def boom(name, conf):
        raise ConnectionError("无法连接")

    monkeypatch.setattr(mcp, "_load_tools", boom)
    await mcp.discover()
    assert mcp.capabilities[0]["unavailable_reason"].startswith("不适配")


async def test_mcp_partial_failure(monkeypatch):
    mcp = McpManager('{"good": {"command": "a"}, "bad": {"command": "b"}}', enabled=True)

    async def partial(name, conf):
        if name == "bad":
            raise RuntimeError("启动失败")
        return [FakeTool()]

    monkeypatch.setattr(mcp, "_load_tools", partial)
    await mcp.discover()
    ids = {c["id"]: c for c in mcp.capabilities}
    assert ids["good:tool_a"]["source"] == "mcp"
    assert ids["bad:*"]["unavailable_reason"].startswith("不适配")


async def test_mcp_enable_then_disable(monkeypatch):
    """页面点选开启→发现工具；关闭→清空能力且可再次开启重新发现。"""
    mcp = McpManager('{"srv": {"command": "x"}}', enabled=False)

    async def fake_load(name, conf):
        return [FakeTool()]

    monkeypatch.setattr(mcp, "_load_tools", fake_load)
    await mcp.enable()
    assert mcp.enabled is True
    assert mcp.capabilities[0]["id"] == "srv:tool_a"

    mcp.disable()
    assert mcp.enabled is False
    assert mcp.capabilities == []
    assert mcp._discovered is False
    assert mcp.tool("srv:tool_a") is None

    await mcp.enable()
    assert mcp.enabled is True
    assert mcp.capabilities[0]["id"] == "srv:tool_a"
