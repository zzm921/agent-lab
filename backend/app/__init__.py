"""应用包初始化：把本地 vendor 目录加入 sys.path。

qdrant-client 依赖（含 grpcio 等大件）因沙箱限制无法写入 conda site-packages，
统一安装到 backend/_vendor，在此预先挂载，保证任意模块可 `import qdrant_client`。
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent.parent / "_vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
