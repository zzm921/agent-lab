"""命令执行工具：在隔离沙箱中执行命令，强制人工审批（HITL）。

沙箱后端由 Settings.sandbox_backend 决定（默认 opensandbox）：
- opensandbox（默认）：连接 Docker 部署的 OpenSandbox 服务端，命令在远端 Docker 容器沙箱内
  执行，由 OpenSandbox 提供强隔离（容器隔离、网络策略、生命周期管理）。服务端由用户自行部署，
  本工具仅通过官方 SDK（pip install opensandbox）接入；SDK 未安装或连接失败时给出明确提示。
- local（轻量 OS 级兜底）：在持久化工作目录 + 最小环境 + 超时硬杀 + 输出上限下于本机执行。

两个后端都把工作目录（Settings.sandbox_work_dir，沙箱内挂载到 sandbox_mount_target）作为命令
执行目录：命令默认在挂载点内执行，相对路径写入的文件会持久化到宿主机该目录，后端通过
/api/sandbox/files 提供列表与下载。opensandbox 后端会按配置指纹复用沙箱容器（创建较慢），
仅在配置变化或空闲超时后销毁重建，避免 Agent 循环内逐条新建拖慢流程。

两层后端共有的保障：
- 危险命令黑名单：拒绝格式化磁盘、递归删除、关机重启等破坏性命令；
- 强制 HITL：无论审批策略如何，该工具调用前都必须经人工审批
  （见 app.agents.harness.ALWAYS_APPROVE_TOOLS），由人来确认命令本身。
"""
from __future__ import annotations

import atexit
import os
import subprocess
import threading
import time
from datetime import timedelta
from pathlib import Path

from langchain_core.tools import tool

# 危险命令特征（小写子串匹配）；命中即拒绝，防止破坏宿主环境
_DANGEROUS_PATTERNS = (
    "rm -rf /",
    "rm -rf /*",
    "rm -fr /",
    "rm -fr /*",
    "sudo rm -rf",
    "del /s /q",
    "rd /s /q",
    "rmdir /s",
    "deltree",
    "format c:",
    "format /q",
    "mkfs",
    "dd if=/dev/zero",
    "dd if=/dev/urandom",
    "shutdown",
    "reboot",
    "halt",
    "chmod -r 777 /",
    "reg delete",
    ">:(){",
    ":(){",
)

# 敏感环境变量名片段：含这些关键字的变量会被剔除
_SENSITIVE_ENV_HINTS = ("key", "token", "secret", "password", "credential", "auth")


def _strip_sensitive_env(env: dict[str, str]) -> dict[str, str]:
    """保留运行所需环境，但剔除含敏感关键字的变量，避免命令接触密钥。"""
    return {name: value for name, value in env.items() if not any(h in name.lower() for h in _SENSITIVE_ENV_HINTS)}


def _deny_reason(command: str) -> str | None:
    """命中危险命令特征时返回原因，否则返回 None。"""
    low = command.lower().strip()
    if not low:
        return "命令为空"
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in low:
            return f"命令命中危险特征「{pattern}」，已拒绝执行"
    return None


def _kill_tree(proc: subprocess.Popen) -> None:
    """终止整个进程树：Windows 用 taskkill /T /F，POSIX 用进程组 SIGKILL。"""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
            )
        except Exception:  # noqa: BLE001
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def _run_sandboxed(command: str, timeout: int, work_dir: Path) -> str:
    """在持久化工作目录中执行命令（local 兜底），写入的文件保留在 work_dir 供下载。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"start_new_session": True} if os.name != "nt" else {}
    proc = None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=work_dir,
            env=_strip_sensitive_env(dict(os.environ)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        out = ((stdout or "") + (stderr or "")).strip()
        return f"退出码：{proc.returncode}\n{out}" if out else f"退出码：{proc.returncode}"
    except subprocess.TimeoutExpired:
        # 超时后子进程可能仍存活，主动清理整个进程树
        if proc is not None:
            _kill_tree(proc)
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        return f"命令执行超时（>{timeout}s），已强制终止。"
    finally:
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass


def _truncate(text: str, limit: int) -> str:
    """按字符上限截断输出，防止超长刷屏。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n……（输出过长，已截断，共 {len(text)} 字符）"


# ---- OpenSandbox 沙箱复用池 ----
# 创建沙箱（拉镜像 + 启动 execd + 就绪探测）较慢（实测约 7s），若每条命令都新建/销毁，
# Agent 循环多次调用 run_command 时会明显拖慢流程。这里按配置指纹缓存并复用沙箱，
# 仅当配置变化或空闲超时后才销毁重建。同一时刻只复用同一个沙箱（持锁串行执行命令）。
_SANDBOX_LOCK = threading.Lock()
_sandbox_pool: dict[str, object] = {}
_sandbox_last_used: dict[str, float] = {}
_SANDBOX_IDLE_TTL = 300.0  # 秒：沙箱空闲超过该时长，下次调用时销毁重建，避免遗留容器


def _sandbox_fingerprint(settings, work_dir: Path) -> str:
    """沙箱复用指纹：镜像 / 服务端 / 工作目录 / 挂载点任一变化都需要重建沙箱。"""
    return "|".join(
        [
            settings.opensandbox_image,
            settings.opensandbox_domain,
            settings.opensandbox_protocol,
            settings.opensandbox_api_key or "",
            str(work_dir),
            settings.sandbox_mount_target,
        ]
    )


def _destroy_sandbox(sandbox) -> None:
    """尽力销毁沙箱容器，避免遗留。"""
    if sandbox is None:
        return
    try:
        sandbox.destroy()
    except Exception:  # noqa: BLE001
        pass


def _close_stale_sandbox(fp: str) -> None:
    """销毁指纹不符或空闲超时的旧沙箱（须持 _SANDBOX_LOCK 调用）。"""
    for k in [k for k in _sandbox_pool if k != fp]:
        _destroy_sandbox(_sandbox_pool.pop(k))
        _sandbox_last_used.pop(k, None)
    last = _sandbox_last_used.get(fp)
    if last is not None and (time.time() - last) > _SANDBOX_IDLE_TTL:
        _destroy_sandbox(_sandbox_pool.pop(fp))
        _sandbox_last_used.pop(fp, None)


def _destroy_all_sandboxes() -> None:
    """进程退出时清理池中所有沙箱（atexit 注册）。"""
    with _SANDBOX_LOCK:
        for sandbox in _sandbox_pool.values():
            _destroy_sandbox(sandbox)
        _sandbox_pool.clear()
        _sandbox_last_used.clear()


atexit.register(_destroy_all_sandboxes)


def _get_or_create_sandbox(settings, work_dir: Path, request_timeout: timedelta) -> tuple:
    """从复用池取或新建沙箱；返回 (sandbox, None) 或 (None, 错误文案)。须持 _SANDBOX_LOCK 调用。"""
    fp = _sandbox_fingerprint(settings, work_dir)
    _close_stale_sandbox(fp)
    sandbox = _sandbox_pool.get(fp)
    if sandbox is not None:
        _sandbox_last_used[fp] = time.time()
        return sandbox, None
    try:
        from opensandbox import SandboxSync
        from opensandbox.config import ConnectionConfigSync

        sandbox = SandboxSync.create(
            settings.opensandbox_image,
            connection_config=ConnectionConfigSync(
                domain=settings.opensandbox_domain,
                protocol=settings.opensandbox_protocol,
                api_key=settings.opensandbox_api_key or None,
                request_timeout=request_timeout,
            ),
            timeout=timedelta(minutes=5),
            ready_timeout=timedelta(seconds=60),
            volumes=_sandbox_volumes(settings, work_dir),
        )
    except ImportError:
        return None, "OpenSandbox SDK 未安装（pip install opensandbox），无法使用沙箱执行。"
    except Exception as exc:  # noqa: BLE001
        return None, f"创建 OpenSandbox 沙箱失败：{exc}"
    _sandbox_pool[fp] = sandbox
    _sandbox_last_used[fp] = time.time()
    return sandbox, None


def _sandbox_volumes(settings, work_dir: Path):
    """构建挂载到沙箱的 Volume 列表；未配置工作目录时返回 None（不挂载）。

    注：opensandbox SDK 0.1.15 的 to_api_volume 对未设置的 pvc/ossfs 用 `is not None` 判断，
    而字段默认值是 UNSET 单例，会触发 `'Unset' object has no attribute 'claim_name'`。
    这里显式传 pvc=None、ossfs=None 绕过该 SDK bug。
    """
    if not settings.sandbox_work_dir.strip():
        return None
    from opensandbox.api.lifecycle.models import Host, Volume

    work_dir.mkdir(parents=True, exist_ok=True)
    return [
        Volume(
            name="sandbox-work",
            mount_path=settings.sandbox_mount_target,
            host=Host(path=str(work_dir)),
            pvc=None,
            ossfs=None,
        )
    ]


def _run_opensandbox(command: str, timeout: int, settings, work_dir: Path) -> str:
    """在 OpenSandbox（用户 Docker 部署的）沙箱中执行命令，返回退出码 + stdout/stderr。

    沙箱按配置指纹复用（见上方复用池）：创建沙箱较慢，Agent 循环内逐条新建/销毁会拖慢流程。
    命令在挂载到工作目录的 sandbox_mount_target 中执行，相对路径写入的文件会持久化到宿主机工作目录。
    """
    try:
        from opensandbox.models.execd import RunCommandOpts
    except ImportError:
        return "OpenSandbox SDK 未安装（pip install opensandbox），无法使用沙箱执行。"
    # request_timeout 是单次 HTTP 请求超时，需覆盖慢速的沙箱创建（首次拉镜像等），
    # 不能复用命令执行超时；命令超时改由 RunCommandOpts.timeout 让服务端强制终止。
    request_timeout = timedelta(seconds=max(180, timeout + 60))
    with _SANDBOX_LOCK:
        sandbox, err = _get_or_create_sandbox(settings, work_dir, request_timeout)
        if sandbox is None:
            return err
        try:
            execution = sandbox.commands.run(
                command,
                opts=RunCommandOpts(
                    timeout=timedelta(seconds=timeout),
                    # 在挂载的工作目录中执行，保证相对路径写入的文件落盘到宿主机工作目录（与 local 后端一致）
                    working_directory=settings.sandbox_mount_target if settings.sandbox_work_dir.strip() else None,
                ),
            )
            stdout = "\n".join(line.text for line in (execution.logs.stdout or []))
            stderr = "\n".join(line.text for line in (execution.logs.stderr or []))
            exit_code = getattr(execution, "exit_code", None)
            out = ((stdout or "") + (stderr or "")).strip()
            parts = [f"退出码：{exit_code}"] if exit_code is not None else []
            if out:
                parts.append(out)
            return "\n".join(parts)
        except Exception as exc:  # noqa: BLE001
            return f"沙箱命令执行失败：{exc}"


def _work_dir(settings) -> Path:
    """沙箱/宿主机共享工作目录（绝对路径，自动创建）。相对路径以 backend 根目录为基准。"""
    d = Path(settings.sandbox_work_dir).expanduser()
    if not d.is_absolute():
        d = Path(__file__).resolve().parents[2] / d
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def make_run_command_tool(settings):
    """构建命令执行工具；沙箱后端与超时/输出上限取自 Settings。"""

    @tool
    def run_command(command: str, timeout: int = 0) -> str:
        """在隔离沙箱中执行系统命令，强制人工审批（HITL）。

        Args:
            command: 要执行的命令行（如 ls -la、echo hello）。
            timeout: 可选超时秒数；为 0 时使用系统默认超时。
        """
        reason = _deny_reason(command)
        if reason is not None:
            return f"已拒绝执行：{reason}"
        eff_timeout = timeout if timeout > 0 else settings.sandbox_timeout
        work_dir = _work_dir(settings)
        if settings.sandbox_backend == "opensandbox":
            result = _run_opensandbox(command, eff_timeout, settings, work_dir)
        else:
            result = _run_sandboxed(command, eff_timeout, work_dir)
        return _truncate(result, settings.sandbox_max_output)

    return run_command
