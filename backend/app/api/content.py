"""能力展示内容 API：从 backend/content/ 下的 md 文件实时读取并解析。

每次请求都会重新读盘解析，编辑/上传 md 后无需重启服务，前端刷新页面即可生效。
"""
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/content", tags=["content"])

# backend/content/ 目录：md 的唯一数据源
CONTENT_DIR = Path(__file__).resolve().parents[2] / "content"


def _parse_md(text: str) -> tuple[dict, str]:
    """把一份 md 拆分为 frontmatter（dict）与正文（str）。"""
    data: dict = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end == -1:
            raise HTTPException(status_code=500, detail="frontmatter 缺少结束符 '---'")
        parsed = yaml.safe_load(text[3:end])
        data = parsed if isinstance(parsed, dict) else {}
        body = text[end + 4 :].lstrip("\n")
    return data, body


def _read_tags() -> list[dict]:
    """读取 tags.md，返回标签注册表（含各标签的卡片 id 列表）。"""
    path = CONTENT_DIR / "tags.md"
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"缺少权威索引 {path.name}")
    data, _ = _parse_md(path.read_text(encoding="utf-8"))
    tags = data.get("tags")
    if not isinstance(tags, list):
        raise HTTPException(status_code=500, detail=f"{path.name} 中缺少 tags 列表")
    return tags


def _read_cards() -> list[dict]:
    """按 tags.md 中的卡片顺序，逐个读取并解析卡片 md（含标签内 groups 的卡片）。"""
    cards: list[dict] = []
    for tag in _read_tags():
        card_ids: list[str] = list(tag.get("cards") or [])
        for group in tag.get("groups") or []:
            card_ids.extend(group.get("cards") or [])
        for card_id in card_ids:
            path = CONTENT_DIR / f"{card_id}.md"
            if not path.exists():
                continue
            data, body = _parse_md(path.read_text(encoding="utf-8"))
            data["id"] = card_id
            data["body"] = body
            cards.append(data)
    return cards


@router.get("")
def get_content() -> dict:
    """返回全部能力展示内容：标签清单 + 各卡片元数据与 md 正文。"""
    return {"tags": _read_tags(), "cards": _read_cards()}
