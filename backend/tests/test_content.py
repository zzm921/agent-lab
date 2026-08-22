"""能力展示内容 API 测试：tags 权威索引与卡片解析。"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_content_tags():
    resp = client.get("/api/content")
    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload["tags"], list)
    assert len(payload["tags"]) >= 6  # 六大技术域
    for tag in payload["tags"]:
        assert tag["id"]
        assert tag["title"]
        assert isinstance(tag["cards"], list)


def test_content_cards():
    resp = client.get("/api/content")
    assert resp.status_code == 200
    cards = resp.json()["cards"]
    assert len(cards) >= 20  # 21 张卡片
    ids = [c["id"] for c in cards]
    assert "react" in ids
    assert "rag" in ids

    react = next(c for c in cards if c["id"] == "react")
    # 元数据齐全
    assert react["name"] == "ReAct 边想边做"
    assert react["mode"] == "react"
    assert react["difficulty"] == "int"
    assert react["completeLevel"] == 100
    assert isinstance(react["tags"], list)
    assert isinstance(react["techFilters"], list)
    assert react["accent"]
    # 正文以 Markdown 呈现，含原理解释与代码块
    assert "## 为什么需要它" in react["body"]
    assert "```python" in react["body"]


def test_content_cards_follow_tags_order():
    resp = client.get("/api/content")
    payload = resp.json()
    card_ids = [c["id"] for c in payload["cards"]]
    order = [cid for tag in payload["tags"] for cid in tag["cards"]]
    assert card_ids == [cid for cid in order if cid in card_ids]
