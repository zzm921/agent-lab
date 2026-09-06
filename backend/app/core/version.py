"""系统版本号工具：评测报告与在线样本标记被测系统版本，支撑「PR-over-PR」分数对比。

- 优先取 git 最新提交短哈希（评测结果可回溯到代码版本）；
- 非 git 环境（如 CI 拉包）回退 'dev'；
- 结果缓存，避免每次评测重复调用 git。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_CACHE: dict[str, str] = {}


def get_app_version() -> str:
    if "version" not in _CACHE:
        _CACHE["version"] = _detect()
    return _CACHE["version"]


def _detect() -> str:
    try:
        root = Path(__file__).resolve().parents[2]  # backend/
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=root,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 — git 不可用时回退，不影响评测
        pass
    return "dev"
