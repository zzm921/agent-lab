"""Modular RAG 三级缓存 + 跨轮 seed 持久化公共件。

三级缓存的职责划分（阶段 2.1，性能治理）：
- L1 查询缓存（schemes/modular.py 内）：消解后 query → 最终命中，命中后**复用命中 + 重跑答案充分性验证**，
  省掉整条消解/路由/检索/后处理成本，只保留验证闸门（正确性门禁不降级）；
- L2 嵌入缓存（本模块 CachedEmbeddings）：query 文本 → 向量，省重复 embedding 调用（生产为远端 API 计费/延迟）；
- L3 检索缓存（schemes/modular.py 内）：query+检索策略 → RRF 融合命中，命中后仍走后处理与验证。

跨轮 seed 持久化（阶段 2.3，CrossTurnSeedStore）：按会话把上一轮已验证命中落盘，进程重启后仍可复用。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TTLCache:
    """带 TTL 的有界 LRU 缓存：读取时惰性驱逐过期项，容量超限淘汰最久未用项。

    - TTL：语料重建/策略变化后的旧命中按有效期自然过期（不做指纹比对，避免每次全量扫描语料）；
    - LRU：max_entries 上限内淘汰最久未用，控制内存与脏数据窗口。
    key 需可哈希（str / tuple）；value 任意对象。
    """

    def __init__(self, max_entries: int = 128, ttl_s: float = 300.0):
        self.max_entries = max(1, max_entries)
        self.ttl_s = ttl_s
        self._store: OrderedDict[Any, tuple[float, Any]] = OrderedDict()

    def get(self, key) -> Any:
        item = self._store.get(key)
        if item is None:
            return None
        ts, value = item
        if self.ttl_s > 0 and time.monotonic() - ts > self.ttl_s:
            del self._store[key]
            return None
        self._store.move_to_end(key)  # 命中即视为最近使用
        return value

    def set(self, key, value) -> None:
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class CachedEmbeddings:
    """嵌入缓存代理（L2）：按文本记忆 embed_query / embed_sparse_query 结果。

    语义上与原 embeddings 完全等价（纯记忆层），只省重复 embedding 调用——
    同一查询多轮重复提问时避免对同一文本反复向量化（生产为远端 DashScope API，重复即重复计费/延迟）。
    语料重建不影响 query 向量有效性（向量只依赖文本与模型），无需失效；
    模型实例/维度变化时由上层重建新代理（新实例自带新缓存）。
    """

    def __init__(self, embeddings, max_entries: int = 1024):
        self._inner = embeddings
        self._dense: OrderedDict[str, Any] = OrderedDict()
        self._sparse: OrderedDict[str, Any] = OrderedDict()
        self._max = max(1, max_entries)

    def _memo(self, store: OrderedDict[str, Any], key: str, compute: Callable[[], Any]) -> Any:
        hit = store.get(key)
        if hit is None:
            hit = compute()
            store[key] = hit
            store.move_to_end(key)
            while len(store) > self._max:
                store.popitem(last=False)
        return hit

    def embed_query(self, text: str) -> Any:
        vec = self._memo(self._dense, text, lambda: self._inner.embed_query(text))
        # 返回副本：防下游（如向量库写入/归一化）原地修改缓存对象污染后续命中
        return list(vec) if isinstance(vec, list) else vec

    def embed_sparse_query(self, text: str) -> Any:
        return self._memo(self._sparse, text, lambda: self._inner.embed_sparse_query(text))


class CrossTurnSeedStore:
    """跨轮 seed 持久化（阶段 2.3）：按会话把上一轮已验证命中落盘，进程重启后仍可复用。

    内存层始终生效（set/get/clear 进程内立即可见，保持原有跨轮行为不变）；
    enabled=True 时增加磁盘持久层（启动加载 + 变更写穿），重启后新会话首轮即可复用上轮证据。

    治理上限（防无界增长/文件膨胀）：
    - max_hits_per_session：单会话命中条数上限（与 modular _SEED_MAX=5 对齐），存前 N 条高分命中；
    - max_sessions：会话数上限，超限淘汰最久未更新的会话（LRU）；
    - ttl_s：种子有效期（秒），超期视为过期（读取/加载时剔除，不写入盘）。
    持久化失败只告警不阻断检索链路（种子的价值是「省重复检索」，丢了也不影响正确性）。
    """

    def __init__(
        self,
        data_path: str | Path,
        enabled: bool = True,
        max_sessions: int = 100,
        ttl_s: float = 86400.0,
        max_hits_per_session: int = 5,
    ):
        self.data_path = Path(data_path)
        self.enabled = enabled
        self.max_sessions = max(1, max_sessions)
        self.ttl_s = ttl_s
        self.max_hits = max(1, max_hits_per_session)
        self._lock = threading.Lock()
        # sid -> (wall_ts, hits)；hits 按分数降序存前 N 条（modular _cross_turn_seed 依赖降序截断）
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        if enabled:
            self._load()

    def get(self, session_id: str) -> list[dict[str, Any]] | None:
        item = self._cache.get(session_id)
        if item is None:
            return None
        ts, hits = item
        if self.ttl_s > 0 and time.time() - ts > self.ttl_s:
            self._cache.pop(session_id, None)
            return None
        return hits

    def set(self, session_id: str, hits: list[dict[str, Any]]) -> None:
        top = sorted(hits, key=lambda h: h.get("score") or 0.0, reverse=True)[: self.max_hits]
        if self.max_sessions > 0 and session_id not in self._cache and len(self._cache) >= self.max_sessions:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            self._cache.pop(oldest, None)
        self._cache[session_id] = (time.time(), top)
        if self.enabled:
            self._flush()

    def clear(self, session_id: str) -> None:
        if self._cache.pop(session_id, None) is not None and self.enabled:
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            payload = {sid: {"ts": ts, "hits": hs} for sid, (ts, hs) in self._cache.items()}
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            self.data_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("[seed] 跨轮 seed 持久化失败（忽略，不影响检索链路）: %s", exc)

    def _load(self) -> None:
        try:
            data = json.loads(self.data_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        now = time.time()
        for sid, entry in data.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("hits"), list) or not entry["hits"]:
                continue
            ts = float(entry.get("ts") or 0.0)
            if self.ttl_s > 0 and now - ts > self.ttl_s:
                continue
            self._cache[sid] = (ts, entry["hits"][: self.max_hits])
