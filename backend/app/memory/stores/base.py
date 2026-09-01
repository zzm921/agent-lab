"""存储层抽象：StoreBackend 定义检索后端契约。

RAG 方案只依赖本接口，不关心底层是 Qdrant / Elasticsearch / 内存——
后续接入 ES 时新增一个实现 StoreBackend 的类即可无缝替换。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StoreBackend(ABC):
    """检索后端统一契约：入库 + 稠密检索 + 混合检索 + 规模。"""

    name: str = "store"
    """后端标识（如 qdrant / elasticsearch / memory）。"""

    @abstractmethod
    def add(self, text: str, metadata: dict | None = None) -> None:
        """写入一条文本（含可选元数据）。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 3, volume_filter: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """稠密向量检索，返回 [{text, score, metadata}]（score 越高越相关）。

        volume_filter：metadata.volume 的精确卷名白名单；None=全库检索。
        不支持过滤的后端应忽略该参数（安全 no-op）。
        """

    @abstractmethod
    def __len__(self) -> int:
        """当前入库的文档条数。"""

    @abstractmethod
    def clear(self) -> None:
        """清空集合全部数据（语料变更后重建用）。"""

    @abstractmethod
    def all_texts(self) -> list[str]:
        """返回集合内全部已入库文本（供语料指纹比对，判断是否需要重建）。"""

    def hybrid_search(self, query: str, top_k: int = 3, volume_filter: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """混合检索（稠密 + 关键词/稀疏）；无混合能力的后端默认退化为稠密检索。"""
        return self.search(query, top_k, volume_filter=volume_filter)

    def delete_source(self, source: str) -> int:
        """按来源（metadata.source）删除该文档的全部块，返回删除条数。

        增量更新的核心操作：文档内容变更时先删旧块再插新块。
        默认不支持删除（返回 0）；memory / qdrant / elasticsearch 后端各自实现。
        """
        return 0

    @property
    @abstractmethod
    def collection(self) -> str:
        """后端内唯一的库/索引/集合名。"""
