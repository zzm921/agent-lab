"""工具与向量检索测试。"""
import builtins
import os
from datetime import datetime

import pytest

from app.memory.vector_store import VectorStore
from app.tools.calculator import calculator
from app.tools.run_command import _sandbox_volumes, make_run_command_tool
from app.tools.time_now import time_now


def test_calculator_basic():
    assert calculator.invoke({"expression": "1+1"}) == "2"


def test_calculator_chinese_operators():
    assert calculator.invoke({"expression": "(137×0.85−20)÷3"}) == "32.15"


def test_calculator_unsafe_rejected():
    with pytest.raises(ValueError):
        calculator.invoke({"expression": "__import__('os')"})


def test_time_now_format():
    result = time_now.invoke({})
    datetime.strptime(result, "%Y-%m-%d %H:%M:%S")


def test_run_command_echo(settings):
    settings.sandbox_backend = "local"
    tool = make_run_command_tool(settings)
    result = tool.invoke({"command": "echo hello-sandbox"})
    assert "hello-sandbox" in result


def test_run_command_deny_dangerous(settings):
    tool = make_run_command_tool(settings)
    result = tool.invoke({"command": "rm -rf /"})
    assert "拒绝" in result
    assert "rm -rf /" in result


def test_run_command_deny_empty(settings):
    tool = make_run_command_tool(settings)
    assert "拒绝" in tool.invoke({"command": "  "})


def test_run_command_timeout(settings):
    settings.sandbox_backend = "local"
    settings.sandbox_timeout = 1
    tool = make_run_command_tool(settings)
    cmd = "ping -n 3 127.0.0.1" if os.name == "nt" else "sleep 3"
    result = tool.invoke({"command": cmd})
    assert "超时" in result


def test_run_command_output_truncated(settings):
    settings.sandbox_backend = "local"
    settings.sandbox_max_output = 10
    tool = make_run_command_tool(settings)
    result = tool.invoke({"command": "echo 0123456789ABCDEF"})
    assert "截断" in result


def test_run_command_opensandbox_backend_dispatch(settings, monkeypatch):
    """sandbox_backend=opensandbox 时命令交给 OpenSandbox 执行器，超时参数透传。"""
    settings.sandbox_backend = "opensandbox"
    tool = make_run_command_tool(settings)
    calls = []

    def fake_run(command, timeout, _settings, _work_dir):
        calls.append((command, timeout))
        return "sandbox-out"

    monkeypatch.setattr("app.tools.run_command._run_opensandbox", fake_run)
    assert tool.invoke({"command": "echo hi", "timeout": 5}) == "sandbox-out"
    assert calls == [("echo hi", 5)]


def test_run_command_opensandbox_sdk_missing(settings, monkeypatch):
    """OpenSandbox SDK 未安装时优雅降级，给出明确提示而非抛异常。"""
    settings.sandbox_backend = "opensandbox"
    tool = make_run_command_tool(settings)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("opensandbox"):
            raise ImportError("no opensandbox sdk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = tool.invoke({"command": "echo hi"})
    assert "未安装" in result


def test_run_command_local_persists_to_work_dir(settings, tmp_path):
    """local 后端：命令在工作目录执行，写入的文件持久化到工作目录（供下载）。"""
    settings.sandbox_backend = "local"
    settings.sandbox_work_dir = str(tmp_path / "work")
    tool = make_run_command_tool(settings)
    result = tool.invoke({"command": "echo hello > artifact.txt"})
    assert "退出码" in result
    assert (tmp_path / "work" / "artifact.txt").read_text(encoding="utf-8").strip() == "hello"


def test_run_command_opensandbox_mounts_work_volume(settings, tmp_path, monkeypatch):
    """opensandbox 后端：沙箱按配置指纹复用，创建时挂载工作目录，命令在挂载点执行且不逐条销毁。"""
    settings.sandbox_backend = "opensandbox"
    settings.sandbox_work_dir = str(tmp_path / "work")
    settings.sandbox_mount_target = "/work"
    captured = {}

    class FakeCommands:
        def run(self, command, opts=None):
            captured["command"] = command
            captured["opts"] = opts
            return type("E", (), {"logs": type("L", (), {"stdout": [], "stderr": []})(), "exit_code": 0})()

    class FakeSandbox:
        def __init__(self, image=None, connection_config=None, timeout=None, volumes=None, ready_timeout=None):
            captured["image"] = image
            captured["volumes"] = volumes

        @classmethod
        def create(cls, image, connection_config=None, timeout=None, ready_timeout=None, volumes=None):
            captured["create_count"] = captured.get("create_count", 0) + 1
            return cls(
                image=image,
                connection_config=connection_config,
                timeout=timeout,
                ready_timeout=ready_timeout,
                volumes=volumes,
            )

        @property
        def commands(self):
            return FakeCommands()

        def destroy(self):
            captured["destroyed"] = captured.get("destroyed", 0) + 1

    import opensandbox

    monkeypatch.setattr(opensandbox, "SandboxSync", FakeSandbox)
    tool = make_run_command_tool(settings)
    assert tool.invoke({"command": "echo hi"}) == "退出码：0"
    assert tool.invoke({"command": "echo hi2"}) == "退出码：0"
    # 复用池：多次调用只创建一次沙箱，命令不逐条销毁
    assert captured["create_count"] == 1
    assert captured.get("destroyed", 0) == 0
    assert captured["command"] == "echo hi2"
    assert captured["opts"].timeout.total_seconds() == settings.sandbox_timeout
    # 命令在挂载的工作目录执行，保证相对路径写入落盘到宿主机工作目录
    assert captured["opts"].working_directory == "/work"
    vol = captured["volumes"][0]
    assert vol.mount_path == "/work"
    assert vol.host.path == str((tmp_path / "work").resolve())


def test_run_command_opensandbox_reuse_pool_destroys_on_config_change(settings, tmp_path, monkeypatch):
    """opensandbox 复用池：配置指纹变化（如工作目录变更）时销毁旧沙箱并重建新沙箱。"""
    settings.sandbox_backend = "opensandbox"
    settings.sandbox_work_dir = str(tmp_path / "work")
    settings.sandbox_mount_target = "/work"
    destroyed = []
    created = []

    class FakeCommands:
        def run(self, command, opts=None):
            return type("E", (), {"logs": type("L", (), {"stdout": [], "stderr": []})(), "exit_code": 0})()

    class FakeSandbox:
        def __init__(self, image=None, connection_config=None, timeout=None, volumes=None, ready_timeout=None):
            created.append(volumes)

        @classmethod
        def create(cls, image, connection_config=None, timeout=None, ready_timeout=None, volumes=None):
            return cls(volumes=volumes)

        @property
        def commands(self):
            return FakeCommands()

        def destroy(self):
            destroyed.append(1)

    import opensandbox

    monkeypatch.setattr(opensandbox, "SandboxSync", FakeSandbox)
    from app.tools import run_command as rc

    monkeypatch.setattr(rc, "_sandbox_pool", {})
    monkeypatch.setattr(rc, "_sandbox_last_used", {})
    tool1 = make_run_command_tool(settings)
    tool1.invoke({"command": "echo a"})
    assert len(created) == 1

    # 工作目录变化 → 新指纹 → 旧沙箱被销毁并重建
    settings.sandbox_work_dir = str(tmp_path / "other")
    tool2 = make_run_command_tool(settings)
    tool2.invoke({"command": "echo b"})
    assert len(destroyed) == 1
    assert len(created) == 2


def test_run_command_opensandbox_volume_passes_sdk_converter(settings, tmp_path):
    """回归：挂载卷须能通过 opensandbox SDK 真实转换器（规避 0.1.15 的 Unset.claim_name bug）。"""
    from opensandbox.adapters.converter.sandbox_model_converter import SandboxModelConverter

    settings.sandbox_work_dir = str(tmp_path / "work")
    settings.sandbox_mount_target = "/work"
    volumes = _sandbox_volumes(settings, tmp_path / "work")
    api_volume = SandboxModelConverter.to_api_volume(volumes[0])
    d = api_volume.to_dict()
    assert d["mountPath"] == "/work"
    assert d["host"]["path"] == str((tmp_path / "work").resolve())


def test_vector_store_search_topk(embeddings):
    vs = VectorStore(embeddings)
    vs.add("关于苹果的水果信息")
    vs.add("关于汽车的技术信息")
    hits = vs.search("水果", top_k=1)
    assert len(hits) == 1
    assert hits[0]["score"] >= 0.0


def test_vector_store_empty(embeddings):
    vs = VectorStore(embeddings)
    assert vs.search("任意") == []
