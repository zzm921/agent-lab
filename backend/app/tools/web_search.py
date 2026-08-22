"""网页搜索工具：DuckDuckGo HTML 接口，无需 Key，网络失败时降级返回提示。"""
import html as html_mod
import re

import httpx
from langchain_core.tools import tool

from app.core.errors import RetryableToolError
from app.tools.retry import is_retryable_exception


def _strip_tags(raw: str) -> str:
    return html_mod.unescape(re.sub(r"<[^>]+>", "", raw))


@tool
def web_search(query: str) -> str:
    """在网页上搜索给定关键词，返回前几条结果的标题、链接与摘要；网络不可用时返回提示。"""
    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        resp.raise_for_status()
        page = resp.text
        blocks = re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', page)
        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', page)
        lines = []
        for i, (href, title) in enumerate(blocks[:3]):
            snippet = ""
            if i < len(snippets):
                snippet = _strip_tags(snippets[i])
            lines.append(f"{i + 1}. {_strip_tags(title)}\n   {href}\n   {snippet}")
        if not lines:
            return "未搜索到相关结果。"
        return "\n".join(lines)
    except RetryableToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        # 瞬时错误（网络超时/连接重置/5xx/429）抛出，交给工具层直接重试（透明重试）；
        # 其余确定性错误（参数/4xx 等）降级为提示文案返回给模型。
        if is_retryable_exception(exc):
            raise RetryableToolError(f"网页搜索瞬时失败（网络或服务端问题）：{exc}") from exc
        return "网页搜索暂时不可用（网络异常），请稍后重试或改用其它方式。"
