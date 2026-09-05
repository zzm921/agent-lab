"""长期记忆存储测试：持久化往返 / 语义去重更新 / 阈值过滤 / 上限 LRU / TTL / 审计。"""
import time

from app.memory.long_memory import LongMemoryStore


def _store(embeddings, tmp_path, **kw):
    return LongMemoryStore("s1", embeddings, str(tmp_path / "mem.jsonl"), **kw)


def test_persist_roundtrip(embeddings, tmp_path):
    """落盘→重载：向量与元数据一致，且不重算 embedding（load 直接重建索引）。"""
    store = _store(embeddings, tmp_path)
    store.add("用户喜欢深色主题", kind="preference", importance=0.9)
    store.add("用户生日是 1995-08-20", kind="fact", importance=0.7)

    reloaded = LongMemoryStore("s1", embeddings, str(tmp_path / "mem.jsonl"))
    assert len(reloaded) == 2
    by_text = {i["text"]: i for i in reloaded.list()}
    assert by_text["用户喜欢深色主题"]["importance"] == 0.9
    assert by_text["用户喜欢深色主题"]["kind"] == "preference"
    hits = reloaded.search("主题配色")
    assert any("深色" in h["text"] for h in hits)


def test_dedup_updates_not_duplicates(embeddings, tmp_path):
    """语义去重：相似度 ≥ dedup_threshold 时合并（旧值归档）而非追加。"""
    store = _store(embeddings, tmp_path)
    r1 = store.add("用户喜欢深色主题，主色是紫色", importance=0.9)
    r2 = store.add("用户喜欢深色主题，主色是紫色", importance=0.8)
    assert r1["action"] == "add"
    assert r2["action"] == "merge"
    assert len(store) == 1
    rec = store.list()[0]
    assert rec["importance"] == 0.9  # 合并重要度取高
    assert rec["merge_count"] == 1
    assert len(rec["history"]) == 1  # 旧值已归档


def test_threshold_filter(embeddings, tmp_path):
    """召回阈值：低于阈值不注入（零命中）。"""
    store = LongMemoryStore("s1", embeddings, str(tmp_path / "m.jsonl"), threshold=0.99)
    store.add("完全无关的字符串内容A")
    assert store.search("与记忆完全无关的查询B") == []


def test_lru_eviction(embeddings, tmp_path):
    """每命名空间上限：超限按 last_access_at 升序 LRU 淘汰最久未访问的。"""
    store = LongMemoryStore("s1", embeddings, str(tmp_path / "m.jsonl"), max_per_namespace=2)
    # 三条互不重叠的记忆（避免被语义去重合并）
    store.add("AAPL苹果公司发布财报数据123")
    store.add("XYZ天气晴朗适合出游456")
    store.add("QWE密码箱在床头柜789")
    assert len(store) == 2
    # 优先淘汰最早写入（last_access_at 最小）的第一条
    texts = [i["text"] for i in store.list()]
    assert "AAPL苹果公司发布财报数据123" not in texts


def test_ttl_cleanup(embeddings, tmp_path):
    """TTL 过期清理：写入治理时剔除创建时间超过 ttl 的记录。"""
    store = LongMemoryStore("s1", embeddings, str(tmp_path / "m.jsonl"), ttl_days=1)
    store.add("旧记忆")
    # 伪造创建时间为 2 天前，再触发一次写入治理
    store._store.metadatas[0]["created_at"] = time.time() - 2 * 86400
    store.add("新记忆")
    texts = [i["text"] for i in store.list()]
    assert "旧记忆" not in texts
    assert "新记忆" in texts


def test_search_updates_access_stats(embeddings, tmp_path):
    """召回即更新访问统计（供 LRU / 老化提示使用）。"""
    store = _store(embeddings, tmp_path)
    store.add("用户喜欢深色主题")
    store.search("主题")
    assert store.list()[0]["access_count"] == 1


def test_audit_add_update_delete(embeddings, tmp_path):
    """审计：add / merge（去重合并归档）/ delete 各记一条，按时间倒序、scope 可过滤。"""
    store = LongMemoryStore("_global:dev-a", embeddings, str(tmp_path / "mem.jsonl"))
    store.add("用户喜欢深色主题", kind="preference", importance=0.9)
    store.add("用户喜欢深色主题", kind="preference", importance=0.8)  # 完全相同 → 去重 merge
    rec = store.list()[0]
    store.delete(rec["id"])

    items = store.list_audit()
    assert [i["action"] for i in items] == ["delete", "merge", "add"]  # 最新在前
    assert all(i["scope"] == "global" for i in items)  # _global 命名空间判定为全局
    assert items[0]["text"].startswith("用户喜欢深色主题")
    # scope 过滤
    assert store.list_audit(scope="session") == []


def test_audit_session_scope(embeddings, tmp_path):
    """审计：普通会话命名空间判定 scope=session。"""
    store = _store(embeddings, tmp_path)
    store.add("用户生日是 1995-08-20")
    items = store.list_audit()
    assert len(items) == 1
    assert items[0]["action"] == "add"
    assert items[0]["scope"] == "session"
