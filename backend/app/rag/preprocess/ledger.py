"""增量入库台账：doc_path → 内容 hash（SHA256），判断文档是否需要重建索引。

解决「数百份文档每次全量重建（重解析 + 重向量化）耗时耗钱」：
- hash 未变 → 跳过该文档（不重新向量化）；
- hash 变了 / 新文档 → 删旧块插新块（按 payload 的 source 过滤删除）；
- 文件消失 / 从容器内被移除 → 清理台账并删除其在库中的块。

台账文件为 JSON（data/ingest/ledger.json），原子写（tmp + replace）防中断损坏。
容器子文档（虚拟路径 父容器/条目名）在条目中记录 container 字段，
清理时按「父容器是否存活 + 本轮该容器产出的子文档集合」双条件判定。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.rag.preprocess.cleaning.dedup import exact_hash

DEFAULT_LEDGER_PATH = Path("data/ingest/ledger.json")


class LedgerStore:
    """内容 hash 台账：load → update/remove → save（显式落盘）。"""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_LEDGER_PATH
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 台账损坏按空台账处理：全量重建（宁多算不漏算）
            self._entries = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def clear(self) -> None:
        self._entries = {}

    # ---- 查询 / 维护 ----

    def hash_of(self, text: str) -> str:
        """文档内容指纹：归一化文本 SHA256（与去重同口径）。"""
        return exact_hash(text)

    def get(self, source: str) -> str | None:
        entry = self._entries.get(source)
        return entry.get("content_hash") if entry else None

    def update(self, source: str, content_hash: str, container: str | None = None) -> None:
        entry: dict = {
            "content_hash": content_hash,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if container is not None:
            entry["container"] = container
        self._entries[source] = entry

    def remove(self, source: str) -> None:
        self._entries.pop(source, None)

    def sources(self) -> list[str]:
        return list(self._entries)

    # ---- 清理判定 ----

    @staticmethod
    def _alive(source: str) -> bool:
        """来源是否仍然存在：真实文件存在，或父容器文件存在（容器子文档虚拟路径）。"""
        p = Path(source)
        if p.exists():
            return True
        parent = p.parent
        # 容器子文档（a.zip/one.md）的父路径是容器文件本身（有后缀名的真实文件）
        return parent.suffix != "" and parent.is_file()

    def stale_sources(
        self,
        current_sources: set[str],
        children_by_container: dict[str, set[str]] | None = None,
        in_dir: Path | str | None = None,
    ) -> list[str]:
        """返回应从库中删除的过期来源（文件消失 / 容器内被移除）。

        - in_dir：本次处理的输入目录——只清理该目录范围内的台账条目，
          避免用子集目录跑批时误删全量库中的其他文档；
        - children_by_container：本轮每个容器实际产出的子文档集合；
          容器存活但子文档不在集合内 → 该子文档已从容器中移除。
        """
        children_by_container = children_by_container or {}
        prefix = str(in_dir) if in_dir else ""
        stale: list[str] = []
        for source, entry in self._entries.items():
            if prefix and not source.startswith(prefix):
                container = entry.get("container") or ""
                if not container.startswith(prefix):
                    continue  # 范围外的条目不参与清理
            if source in current_sources:
                continue
            container = entry.get("container")
            if container is not None:
                if not self._alive(container):
                    stale.append(source)  # 容器文件本身已删除
                elif source not in children_by_container.get(container, set()):
                    stale.append(source)  # 容器还在，但该子文档已被移除
            elif not self._alive(source):
                stale.append(source)
        return stale
