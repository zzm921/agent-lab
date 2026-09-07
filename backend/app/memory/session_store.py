"""会话存储：LangGraph 检查点（文件持久化，多会话）+ 会话元数据（列表/标题/TTL）+ 记忆库。

checkpointer 采用文件型 SQLite（langgraph-checkpoint-sqlite 的 AsyncSqliteSaver）：
- 落盘目录由 settings.checkpoint_dir 控制，空则回退进程内存 MemorySaver（测试/离线，重启即失）；
- SQLite saver 是异步上下文管理器，需在事件循环中初始化（SessionStore.init_checkpointer）。

会话元数据（会话列表 API 的数据源）：
- 记录 {session_id, title, client_key, created_at, last_active, message_count}，
  按客户端隔离：每个试用者只看到/管理自己的会话；
- 首条消息自动生成标题；TTL 治理：last_active 超过 checkpoint_ttl_days 天的会话
  由 gc_ttl 清理（连带删除 checkpoint 线程 + 会话记忆文件）。

常驻（全局）记忆按客户端隔离：每个试用者（按设备指纹 X-Client-Id 或 IP 标识）
各有一份常驻库，互不可见，防止「记忆串台」；会话记忆仍按随机 session_id 隔离。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.memory.long_memory import LongMemoryStore

_GLOBAL_ID = "_global"
_MAX_TITLE = 30
logger = logging.getLogger(__name__)


def _safe_key(key: str) -> str:
    """把 client_key（cid:xxx / ip:xxx）转成文件安全名，避免冒号等非法字符进路径。"""
    return re.sub(r"[^0-9A-Za-z_\-]", "_", key)[:64]


def _clean_title(text: str) -> str:
    """首条消息 → 会话标题：压缩空白、截断。"""
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= _MAX_TITLE else t[:_MAX_TITLE] + "…"


class SessionStore:
    """单机形态的会话与记忆存储。

    memory_dir 为空时记忆库仅存内存（测试/离线回退，不落盘）；
    配置了 memory_dir 时，每会话一个 {memory_dir}/{session_id}.jsonl，
    每个客户端的常驻记忆为 {memory_dir}/_global_{client_key}.jsonl。

    checkpoint_dir 为空时 checkpointer 为进程内存（重启即失）；
    配置后为文件型 SQLite（多会话历史持久化）。sessions_meta_path 为空时元数据仅内存。
    """

    def __init__(
        self,
        memory_dir: str | None = None,
        checkpoint_dir: str | None = None,
        sessions_meta_path: str | None = None,
        checkpoint_ttl_days: int = 7,
        events_dir: str | None = None,
        **memory_kwargs,
    ):
        # 文件型 SQLite saver 需在事件循环中初始化；未 init 前先用内存 saver 兜底
        self.checkpointer = MemorySaver()
        self._cp_gen = None  # AsyncSqliteSaver.from_conn_string 的 async generator 句柄（防 GC 关闭连接）
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_ttl_days = checkpoint_ttl_days
        self._sessions_meta_path = sessions_meta_path
        self._events_dir = events_dir
        self._memory_dir = memory_dir
        self._memory_kwargs = memory_kwargs
        self._long_memories: dict[str, LongMemoryStore] = {}
        self._constants: dict[str, LongMemoryStore] = {}
        # 会话元数据：session_id → meta；启动时从文件载入
        self._meta: dict[str, dict[str, Any]] = {}
        self._load_meta()

    # ---- checkpointer 持久化 ----

    async def init_checkpointer(self) -> None:
        """启动时调用：把内存 saver 换成文件型 SQLite saver（AsyncSqliteSaver）。

        checkpoint_dir 为空（测试/离线）时保持内存 saver 不变。
        仅在事件循环中可初始化（SQLite saver 是异步上下文管理器）。
        """
        if not self._checkpoint_dir:
            return
        if not isinstance(self.checkpointer, MemorySaver):
            return  # 已初始化过
        try:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            db_path = os.path.join(self._checkpoint_dir, "checkpoints.sqlite")
            saver = AsyncSqliteSaver.from_conn_string(db_path)  # 直接传文件路径（非 URI）
            # 关键：from_conn_string 返回 async generator，连接生命周期由其 async with 管理；
            # 必须持有该句柄（self._cp_gen），否则被 GC 时 __aexit__ 会关闭连接。
            self._cp_gen = saver
            self.checkpointer = await saver.__aenter__()  # 保持连接，进程生命周期内复用
            await self.checkpointer.setup()  # 建表（checkpoints/writes），供 adelete_thread 等使用
            logger.info("checkpointer 已切换到文件型 SQLite：%s", db_path)
        except Exception as exc:  # noqa: BLE001 — 持久化失败不阻断服务，回退内存
            logger.warning("checkpointer 文件持久化初始化失败，回退内存（重启即失）: %s", exc)

    # ---- 会话事件流（历史回放）----

    def _events_path(self, session_id: str) -> str | None:
        if not self._events_dir:
            return None
        return f"{self._events_dir}/{session_id}.jsonl"

    def record_events(self, session_id: str, events):
        """把 SSE 事件生成器 tee 到会话事件文件：边透传边落盘，供切换会话时重放。

        events_dir 未配置（测试/离线）时原样透传不落盘；落盘失败静默吞掉，
        绝不阻断/污染主流程（与 telemetry 同款「只读不写也不出错」原则）。
        """
        path = self._events_path(session_id)
        if path is None:
            return events
        os.makedirs(os.path.dirname(path), exist_ok=True)

        async def gen():
            pending: list[dict] = []

            def flush() -> None:
                if not pending:
                    return
                try:
                    with open(path, "a", encoding="utf-8") as f:
                        for ev in pending:
                            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                except OSError as exc:
                    logger.warning("会话事件落盘失败（已忽略）: %s", exc)
                finally:
                    pending.clear()

            try:
                async for ev in events:
                    pending.append(ev)
                    if len(pending) >= 32:  # 攒批落盘，避免流式 delta 逐条开文件
                        flush()
                    yield ev
                flush()
            except GeneratorExit:
                # 客户端断连：把已产出的部分落盘再退出，历史尽量完整
                flush()
                raise

        return gen()

    def load_events(self, session_id: str) -> list[dict]:
        """读取会话已落盘的事件流（顺序即直播顺序），供前端重放。"""
        path = self._events_path(session_id)
        if path is None:
            return []
        try:
            with open(path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("会话事件读取失败（已忽略）: %s", exc)
            return []

    # ---- 会话元数据（多会话）----

    def _load_meta(self) -> None:
        """启动时把 {sessions_meta_path} 载入内存（JSONL，一行一个会话）。"""
        if not self._sessions_meta_path:
            return
        try:
            with open(self._sessions_meta_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("session_id"):
                        self._meta[rec["session_id"]] = rec
        except FileNotFoundError:
            pass  # 首次启动，无历史会话

    def _save_meta(self) -> None:
        """把全部会话元数据原子写回文件（小文件全量重写，简单可靠）。"""
        if not self._sessions_meta_path:
            return
        try:
            os.makedirs(os.path.dirname(self._sessions_meta_path) or ".", exist_ok=True)
            tmp = f"{self._sessions_meta_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for rec in self._meta.values():
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp, self._sessions_meta_path)
        except OSError as exc:
            logger.warning("会话元数据写入失败（已忽略）: %s", exc)

    def create(self, client_key: str = "default", title: str = "") -> str:
        """新建会话：分配 session_id 并登记元数据（标题为空则按首条消息自动生成）。"""
        session_id = uuid.uuid4().hex
        now = time.time()
        self._meta[session_id] = {
            "session_id": session_id,
            "title": title,
            "client_key": client_key,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }
        self._save_meta()
        return session_id

    def list(self, client_key: str | None = None) -> list[dict[str, Any]]:
        """会话列表：按最后活跃时间倒序（新在前）；client_key 非空时只返回该客户端会话。"""
        recs = list(self._meta.values())
        if client_key:
            recs = [r for r in recs if r.get("client_key") == client_key]
        recs.sort(key=lambda r: r.get("last_active", 0), reverse=True)
        return recs

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self._meta.get(session_id)

    def touch(self, session_id: str, client_key: str = "default") -> None:
        """对话每轮调用：更新最后活跃时间与消息数（标题由 title_if_empty 负责）。"""
        rec = self._meta.get(session_id)
        if rec is None:
            # 兼容历史 session_id（前端直连无元数据）：补登记
            rec = {
                "session_id": session_id,
                "title": "",
                "client_key": client_key,
                "created_at": time.time(),
                "last_active": time.time(),
                "message_count": 0,
            }
            self._meta[session_id] = rec
        rec["last_active"] = time.time()
        rec["message_count"] = int(rec.get("message_count", 0)) + 1
        self._save_meta()

    def title_if_empty(self, session_id: str, client_key: str, message: str) -> None:
        """首条消息自动标题：会话尚未有标题且无历史消息时，用本条消息生成标题。"""
        rec = self._meta.get(session_id)
        if rec is None:
            rec = {
                "session_id": session_id,
                "title": _clean_title(message),
                "client_key": client_key,
                "created_at": time.time(),
                "last_active": time.time(),
                "message_count": 0,
            }
            self._meta[session_id] = rec
            self._save_meta()
            return
        if rec.get("title"):
            return
        rec["title"] = _clean_title(message)
        self._save_meta()

    def rename(self, session_id: str, title: str) -> dict[str, Any] | None:
        """重命名会话；标题为空则清空（等待下一条消息重新生成）。"""
        rec = self._meta.get(session_id)
        if rec is None:
            return None
        rec["title"] = title
        self._save_meta()
        return rec

    async def delete(self, session_id: str) -> bool:
        """删除会话：移除元数据 + 删除 checkpoint 线程 + 删除会话记忆文件。

        返回是否存在该会话。删除失败（文件锁等）静默吞掉，只保证元数据移除。
        """
        if session_id not in self._meta:
            return False
        del self._meta[session_id]
        self._save_meta()
        # 1) checkpoint 线程（文件 saver 时生效；内存 saver 的 adelete_thread 仅接受 thread_id）
        try:
            await self.checkpointer.adelete_thread(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除会话 checkpoint 失败（已忽略）: %s", exc)
        # 2) 会话记忆文件
        if self._memory_dir:
            try:
                path = f"{self._memory_dir}/{session_id}.jsonl"
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                logger.warning("删除会话记忆文件失败（已忽略）: %s", exc)
        # 3) 会话记忆库内存缓存
        self._long_memories.pop(session_id, None)
        # 4) 会话事件流文件（历史回放源）
        events_path = self._events_path(session_id)
        if events_path and os.path.exists(events_path):
            try:
                os.remove(events_path)
            except OSError as exc:
                logger.warning("删除会话事件文件失败（已忽略）: %s", exc)
        return True

    async def gc_ttl(self) -> int:
        """TTL 治理：清理超过 checkpoint_ttl_days 天未活跃的会话（含 checkpoint/记忆）。

        返回清理的会话数；ttl_days<=0 表示不启用。
        """
        days = self._checkpoint_ttl_days
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        expired = [
            sid
            for sid, rec in self._meta.items()
            if rec.get("last_active", 0) < cutoff
        ]
        for sid in expired:
            await self.delete(sid)
        if expired:
            logger.info("TTL 清理过期会话 %d 个（>%d 天未活跃）", len(expired), days)
        return len(expired)

    # ---- 记忆库（原有）----

    def _store_path(self, session_id: str) -> str | None:
        if not self._memory_dir:
            return None
        return f"{self._memory_dir}/{session_id}.jsonl"

    def long_memory(self, session_id: str, embeddings) -> LongMemoryStore:
        """获取（必要时创建）某会话的长期记忆库。"""
        store = self._long_memories.get(session_id)
        if store is None:
            store = LongMemoryStore(
                session_id,
                embeddings,
                self._store_path(session_id),
                **self._memory_kwargs,
            )
            self._long_memories[session_id] = store
        return store

    def constant_memory(self, embeddings, client_key: str = "default") -> LongMemoryStore:
        """获取（必要时创建）某客户端的常驻记忆库：按 client_key 各一份，互不可见。

        client_key 由请求层判定（设备指纹 X-Client-Id 优先、IP 兜底）；
        未传时回退 "default"（兼容测试/无请求场景）。
        """
        key = client_key or "default"
        store = self._constants.get(key)
        if store is None:
            store = LongMemoryStore(
                f"{_GLOBAL_ID}:{key}",
                embeddings,
                self._store_path(f"{_GLOBAL_ID}_{_safe_key(key)}"),
                **self._memory_kwargs,
            )
            self._constants[key] = store
        return store
