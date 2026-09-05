"""长期记忆存储：持久化 + 分类 + 元数据 + 治理（语义合并 / 上限 LRU / TTL）+ 审计。

数据模型（每条记录 JSONL 一行）：
{id, kind, text, vector, importance, created_at, last_access_at,
 access_count, source_session, sources, ttl, merge_count, history, updated_at}

向量随记录落盘，启动 load 直接重建内存索引（不重算 embedding）——
单机「温/冷合一」形态：检索毫秒级、不调 LLM。

写入采用「合并式更新」（企业级 Mem0 式 提取→匹配→合并）：
- 相似度 ≥ MERGE_HIGH：同一事实的补充/更新 → 旧值入 history 归档、新表述作当前值；
- 模糊带 [MERGE_LOW, MERGE_HIGH)：交调用方裁决（LLM 判定 merge/conflict/add）；
  无裁决时含「改口」触发词 → 合并并标 conflict，否则保守新增（宁重不漏）；
- 相似度 < MERGE_LOW：视为不同事实，另存一条。
这样「用户改口」不覆盖丢信息：旧值留在 history 供追溯、绝不再作为当前值召回。

审计：所有 add（新增）/ merge（合并归档）/ conflict（改口合并）/ delete 操作追加到
目录级统一 `_audit.jsonl`（{ts, ns, scope, action, kind, importance, text, reason}），
供用户掌控回溯。
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from app.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

MEMORY_KINDS = ("fact", "preference", "episodic", "procedural")

# 合并判定阈值：≥ MERGE_HIGH 直接合并（语义几乎相同）；[MERGE_LOW, MERGE_HIGH) 交调用方裁决；
# < MERGE_LOW 视为不同事实。MERGE_HIGH 复用构造参数 dedup_threshold（默认 0.92）。
MERGE_LOW = 0.6

# 「改口/变更」触发词：新表述含这些词且落在模糊带时，按「用户推翻旧记忆」处理（合并并归档）。
_CONFLICT_RE = re.compile(r"(改成|换成|改为|换为|改用|不再|别用|不用|取消|不喜欢|纠正|其实)")


def is_conflict_rewrite(text: str) -> bool:
    """判定新表述是否为对旧记忆的「改口/推翻」（变更触发词命中）。"""
    return bool(text and _CONFLICT_RE.search(text))


class LongMemoryStore:
    """会话级/全局级长期记忆：写入（去重）、召回（阈值+kind 过滤）、管理与遗忘。"""

    def __init__(
        self,
        session_id: str,
        embeddings,
        path: str | None = None,
        *,
        top_k: int = 3,
        threshold: float = 0.0,
        dedup_threshold: float = 0.92,
        max_per_namespace: int = 500,
        ttl_days: int = 0,
    ):
        self.session_id = session_id
        self.embeddings = embeddings
        self.path = Path(path) if path else None
        self.top_k = top_k
        self.threshold = threshold
        self.dedup_threshold = dedup_threshold
        self.max_per_namespace = max_per_namespace
        self.ttl_days = ttl_days
        # 审计统一落在 memory_dir 下的 _audit.jsonl（会话库/常驻库共用，按 ns 区分）
        self._audit_path = (self.path.parent / "_audit.jsonl") if self.path is not None else None
        self._store = VectorStore(embeddings, name=f"memory:{session_id}")
        self._index: dict[str, int] = {}  # record_id -> 内存索引
        self._load()

    # ---------- 内部工具 ----------

    def _rebuild_index(self) -> None:
        self._index = {
            meta.get("id"): i
            for i, meta in enumerate(self._store.metadatas)
            if meta.get("id")
        }

    def _load(self) -> None:
        if self.path is not None:
            self._store.load(str(self.path))
        self._rebuild_index()

    def _persist(self) -> None:
        if self.path is not None:
            self._store.save(str(self.path))

    def _drop(self, index: int) -> None:
        self._store.delete(index)
        self._rebuild_index()

    # ---------- 审计 ----------

    def _write_audit(self, action: str, text: str, kind: str, importance: float, reason: str = "") -> None:
        """追加一条操作审计；写入失败静默吞掉（审计是增强项，不影响主链路）。"""
        if self._audit_path is None:
            return
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "ns": self.session_id,
            "scope": "global" if str(self.session_id).startswith("_global") else "session",
            "action": action,  # add | merge | conflict | delete
            "kind": kind,
            "importance": round(float(importance), 3),
            "text": (text or "")[:200],
        }
        if reason:
            rec["reason"] = reason
        try:
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("memory audit 写入失败（已忽略）: %s", exc)

    def list_audit(self, limit: int = 50, scope: str | None = None) -> list[dict[str, Any]]:
        """读取审计流水（按时间倒序，最新在前）；scope 可按 session/global 过滤。"""
        if self._audit_path is None or not self._audit_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with open(self._audit_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        if scope:
            rows = [r for r in rows if r.get("scope") == scope]
        return rows[-limit:][::-1]

    def _enforce_limits(self) -> None:
        """写入后治理：TTL 清理 + 每命名空间上限 LRU 淘汰（按 last_access_at 升序）。"""
        if self.ttl_days > 0:
            cutoff = time.time() - self.ttl_days * 86400
            for i in sorted(
                (i for i, m in enumerate(self._store.metadatas) if (m.get("created_at") or 0) < cutoff),
                reverse=True,
            ):
                self._drop(i)
        while self.max_per_namespace > 0 and len(self._store) > self.max_per_namespace:
            idx = min(
                range(len(self._store.metadatas)),
                key=lambda i: self._store.metadatas[i].get("last_access_at", 0),
            )
            self._drop(idx)

    def _match(self, text: str, top_k: int = 3, threshold: float = MERGE_LOW) -> list[dict[str, Any]]:
        """语义匹配：返回 [{id, text, score, idx, metadata}]（不触碰访问统计）。"""
        if not self._store.texts:
            return []
        out = []
        for h in self._store.search(text, top_k=top_k, threshold=threshold):
            rec_id = (h.get("metadata") or {}).get("id")
            idx = self._index.get(rec_id) if rec_id else None
            if idx is None:
                continue
            out.append({**h, "idx": idx})
        return out

    def match(self, text: str, top_k: int = 3, threshold: float = MERGE_LOW) -> list[dict[str, Any]]:
        """供调用方预裁决的匹配（不含内部 idx）：返回 [{id, text, score, metadata}]，不触碰访问统计。"""
        return [{k: v for k, v in h.items() if k != "idx"} for h in self._match(text, top_k, threshold)]

    # ---------- 写入 ----------

    def _create_record(
        self,
        text: str,
        kind: str,
        importance: float,
        source_session: str | None,
        ttl: int | None,
    ) -> dict[str, Any]:
        now = time.time()
        rec_id = uuid.uuid4().hex
        meta = {
            "id": rec_id,
            "kind": kind,
            "importance": importance,
            "created_at": now,
            "last_access_at": now,
            "access_count": 0,
            "source_session": source_session or self.session_id,
            "sources": [source_session] if source_session else [],
            "ttl": ttl,
            "merge_count": 0,
            "history": [],
        }
        self._store.add(text, meta)
        self._index[rec_id] = len(self._store) - 1
        self._enforce_limits()
        self._persist()
        self._write_audit("add", text, kind, importance)
        return {"id": rec_id, "action": "add"}

    def _merge_record(
        self,
        idx: int,
        new_text: str,
        kind: str,
        importance: float,
        source_session: str | None,
        action: str,  # merge | conflict
        reason: str,
    ) -> dict[str, Any]:
        """合并式更新：旧值入 history 归档、新表述作当前值；重要度取高、来源并集。
        action 决定审计标识：merge=同义补充/更新；conflict=用户改口。"""
        old_meta = dict(self._store.metadatas[idx])
        old_text = self._store.texts[idx]
        now = time.time()
        sources = list(old_meta.get("sources") or (([old_meta["source_session"]] if old_meta.get("source_session") else [])))
        if source_session and source_session not in sources:
            sources.append(source_session)
        history = list(old_meta.get("history") or []) + [
            {"text": old_text, "importance": old_meta.get("importance", 0), "time": now, "reason": reason or action}
        ]
        importance = max(old_meta.get("importance", 0), importance)
        meta = {
            **old_meta,
            "kind": kind,
            "importance": importance,
            "source_session": source_session or old_meta.get("source_session"),
            "sources": sources,
            "merge_count": (old_meta.get("merge_count") or 0) + 1,
            "updated_at": now,
            "history": history,
        }
        self._store.update(idx, text=new_text, metadata=meta)
        self._persist()
        self._write_audit(action, new_text, kind, importance, reason=reason)
        return {"id": old_meta.get("id"), "action": action}

    def add(
        self,
        text: str,
        kind: str = "fact",
        importance: float = 0.5,
        source_session: str | None = None,
        ttl: int | None = None,
    ) -> dict[str, Any]:
        """写入一条记忆（确定性合并）：≥ 高阈值直接合并归档；模糊带含改口触发词 → 合并标
        conflict，否则保守新增（宁重不漏）；< 低阈值视为不同事实另存。写后落盘 + 治理。"""
        match = self._match(text)
        if match:
            best = match[0]
            sim = best["score"]
            if sim >= self.dedup_threshold:
                return self._merge_record(best["idx"], text, kind, importance, source_session, "merge", "补充/更新")
            if sim >= MERGE_LOW and is_conflict_rewrite(text):
                return self._merge_record(best["idx"], text, kind, importance, source_session, "conflict", "用户改口")
        return self._create_record(text, kind, importance, source_session, ttl)

    def add_judged(
        self,
        text: str,
        kind: str = "fact",
        importance: float = 0.5,
        source_session: str | None = None,
        ttl: int | None = None,
        decision: str = "add",  # add | merge | conflict
        reason: str = "",
        match_id: str | None = None,
    ) -> dict[str, Any]:
        """按调用方（如巩固模块经 LLM 裁决后）的显式决定写入：merge/conflict 合并进 match_id
        对应记录；add 或 match_id 无效则新建。裁决失败回退走 add 的确定性合并。"""
        if decision in ("merge", "conflict") and match_id:
            idx = self._index.get(match_id)
            if idx is not None:
                return self._merge_record(idx, text, kind, importance, source_session, decision, reason or decision)
        return self._create_record(text, kind, importance, source_session, ttl)

    # ---------- 召回 ----------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """按语义召回：阈值过滤 + kind 过滤；命中即更新访问统计（供 LRU/老化）。"""
        top_k = top_k or self.top_k
        threshold = self.threshold if threshold is None else threshold
        hits = self._store.search(query, top_k=max(top_k * 2, 8), threshold=threshold)
        if kind:
            hits = [h for h in hits if (h.get("metadata") or {}).get("kind") == kind]
        now = time.time()
        changed = False
        for h in hits:
            rec_id = (h.get("metadata") or {}).get("id")
            idx = self._index.get(rec_id)
            if idx is None:
                continue
            meta = dict(self._store.metadatas[idx])
            meta["last_access_at"] = now
            meta["access_count"] = meta.get("access_count", 0) + 1
            self._store.metadatas[idx] = meta
            changed = True
        if changed:
            self._persist()
        return hits[:top_k]

    def constant_memories(self, top_k: int, min_importance: float = 0.7) -> list[tuple[str, dict]]:
        """常驻记忆：按 importance 降序取 top-k（注入 system 用，不依赖查询）。"""
        cands = [
            (text, meta)
            for text, meta in zip(self._store.texts, self._store.metadatas)
            if meta.get("importance", 0) >= min_importance
        ]
        cands.sort(key=lambda t: t[1].get("importance", 0), reverse=True)
        return cands[:top_k]

    # ---------- 管理 ----------

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for text, meta in zip(self._store.texts, self._store.metadatas):
            if kind and meta.get("kind") != kind:
                continue
            items.append({"id": meta.get("id"), "text": text, **meta})
        return items

    def delete(self, rec_id: str) -> bool:
        idx = self._index.get(rec_id)
        if idx is None:
            return False
        text = self._store.texts[idx]
        meta = self._store.metadatas[idx]
        self._drop(idx)
        self._persist()
        self._write_audit("delete", text, meta.get("kind", "fact"), meta.get("importance", 0.0))
        return True

    def __len__(self) -> int:
        return len(self._store)

    def to_jsonl_snapshot(self) -> str:
        """返回当前全部记录的可读 JSON（管理面板展示用）。"""
        return json.dumps(self.list(), ensure_ascii=False, indent=2)
