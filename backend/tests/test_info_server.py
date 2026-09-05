"""mcp-info 只读 MCP Server 测试：工具函数只读往返 + 环境变量白名单（不联网）。"""
import app.mcp_server.info_server as info


def test_now_returns_readable_time():
    out = info.now()
    assert "当前时间" in out
    assert "时区" in out


def test_system_info_covers_core_fields():
    out = info.system_info()
    assert "操作系统" in out
    assert "Python" in out
    assert "主机名" in out
    assert "CPU" in out


def test_env_get_whitelist_ok(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    out = info.env_get("TZ")
    assert "Asia/Shanghai" in out


def test_env_get_unset_env(monkeypatch):
    monkeypatch.delenv("TZ", raising=False)
    out = info.env_get("TZ")
    assert "未设置" in out


def test_env_get_rejects_outside_whitelist():
    out = info.env_get("LLM_API_KEY")
    assert "拒绝读取" in out
    assert "LLM_API_KEY" in out
