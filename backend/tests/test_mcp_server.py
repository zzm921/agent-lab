"""mcp-notes 便签 MCP Server 测试：NotesStore 持久化 + 工具函数往返（不联网）。"""
import app.mcp_server.notes_server as ns


def test_store_save_get_roundtrip(tmp_path):
    store = ns.NotesStore(tmp_path / "notes.json")
    store.save("开会", "明天上午 10 点开会")
    out = store.get("开会")
    assert "开会" in out
    assert "明天上午 10 点开会" in out


def test_store_same_title_overwrites(tmp_path):
    store = ns.NotesStore(tmp_path / "notes.json")
    store.save("待办", "旧内容")
    store.save("待办", "新内容")
    assert "旧内容" not in store.get("待办")
    assert "新内容" in store.get("待办")


def test_store_list_orders_by_updated_at(tmp_path):
    store = ns.NotesStore(tmp_path / "notes.json")
    store.save("甲", "第一条内容很长很长很长很长很长很长很长很长很长很长很长很长很长很长")
    store.save("乙", "第二条")
    listing = store.list()
    # 后保存的应排前面（倒序）
    assert listing.index("乙") < listing.index("甲")
    # 长内容应截断为预览
    assert "…" in listing


def test_store_delete(tmp_path):
    store = ns.NotesStore(tmp_path / "notes.json")
    store.save("临时", "内容")
    assert "已删除" in store.delete("临时")
    assert "未找到" in store.get("临时")
    assert "未找到" in store.delete("不存在")


def test_store_empty_and_missing_file(tmp_path):
    store = ns.NotesStore(tmp_path / "none.json")
    assert store.list() == "暂无便签"
    assert "未找到" in store.get("任何")


def test_store_creates_directory(tmp_path):
    target = tmp_path / "a" / "b" / "notes.json"
    store = ns.NotesStore(target)
    store.save("x", "y")
    assert target.exists()


def test_tools_roundtrip(monkeypatch, tmp_path):
    """工具函数（绕过 FastMCP 框架）经临时 store 直接往返。"""
    monkeypatch.setattr(ns, "store", ns.NotesStore(tmp_path / "notes.json"))
    assert "已保存" in ns.save_note("备忘", "记得打卡")
    listing = ns.list_notes()
    assert "备忘" in listing
    assert "记得打卡" in listing
    assert "记得打卡" in ns.get_note("备忘")
    assert "已删除" in ns.delete_note("备忘")
    assert ns.list_notes() == "暂无便签"
