"""MCP 集成测试：工具发现成功与连接失败标记「不适配」。"""
from app.capabilities.mcp import McpManager


class FakeTool:
    name = "tool_a"
    description = "一个假的 MCP 工具"


async def test_mcp_discover_success(monkeypatch):
    mcp = McpManager('{"srv": {"command": "x"}}')

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
    mcp = McpManager('{"srv": {"command": "x"}}')

    async def boom(name, conf):
        raise ConnectionError("无法连接")

    monkeypatch.setattr(mcp, "_load_tools", boom)
    await mcp.discover()
    assert mcp.capabilities[0]["unavailable_reason"].startswith("不适配")


async def test_mcp_partial_failure(monkeypatch):
    mcp = McpManager('{"good": {"command": "a"}, "bad": {"command": "b"}}')

    async def partial(name, conf):
        if name == "bad":
            raise RuntimeError("启动失败")
        return [FakeTool()]

    monkeypatch.setattr(mcp, "_load_tools", partial)
    await mcp.discover()
    ids = {c["id"]: c for c in mcp.capabilities}
    assert ids["good:tool_a"]["source"] == "mcp"
    assert ids["bad:*"]["unavailable_reason"].startswith("不适配")
