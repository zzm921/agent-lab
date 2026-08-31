"""多模态 OCR 调用：扫描页/图片 → qwen3.5-flash 识别文本。

项目约束：LLM 场景统一使用多模态模型 qwen3.5-flash（settings.chat_model），
多模态模型必须走 MultiModalConversation.call() 端点（文本端点会报 url error）。
响应 content 为 [{...}] 列表结构时经 _content_text 归一化为纯文本。

同步实现，供离线建库脚本直接调用；若日后接入在线 API（事件循环内），
调用方必须用 asyncio.to_thread() 包裹，防止阻塞 SSE 流式输出。
失败重试 1 次，仍失败抛 OcrError（中文 actionable 信息），由管线进 DLQ。
"""
from __future__ import annotations

import base64
import logging
from http import HTTPStatus

from dashscope import MultiModalConversation

logger = logging.getLogger(__name__)

# OCR 提示词：按阅读顺序输出纯文本；表格行内「列名: 值」扁平化，便于关键词召回
OCR_PROMPT = (
    "请识别图片中的全部文字，按阅读顺序输出纯文本，不要输出任何解释或说明。"
    "表格按行输出，每行格式为「列名1: 值1 | 列名2: 值2」；"
    "标题单独成行。无法识别的内容跳过。"
)

_OCR_RETRY = 1  # 失败重试次数（指数退避略过：OCR 为低频离线操作，立即重试即可）


class OcrError(Exception):
    """OCR 调用失败（重试后仍失败）。"""


def ocr_image(image_bytes: bytes, *, model: str | None = None) -> str:
    """识别单张图片，返回纯文本。image_bytes 为 PNG/JPEG 原始字节。"""
    from app.config import settings

    target = model or settings.chat_model
    data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
    messages = [
        {
            "role": "user",
            "content": [{"image": data_url}, {"text": OCR_PROMPT}],
        }
    ]
    last_error = ""
    for attempt in range(1 + _OCR_RETRY):
        resp = MultiModalConversation.call(model=target, messages=messages)
        if resp.status_code == HTTPStatus.OK:
            message = resp.output.choices[0].message
            return _content_text(message.get("content") or "")
        last_error = _friendly_ocr_error(resp, target)
        logger.warning("OCR 调用失败（第 %s 次，model=%s）：%s", attempt + 1, target, last_error)
    raise OcrError(last_error)


def _friendly_ocr_error(resp, model: str) -> str:
    """DashScope 非 200 → 可操作的中文提示（项目约定：不暴露晦涩英文原始报错）。"""
    status = getattr(resp, "status_code", "")
    message = getattr(resp, "message", "") or ""
    lower = message.lower()
    if "url error" in lower or "please check url" in lower:
        return (
            f"OCR 调用失败(status={status})：模型名称与调用端点不匹配（当前 model={model}）。\n"
            f"  处理：多模态模型必须走 MultiModalConversation.call() 端点（本模块已使用），\n"
            f"  请确认 settings.chat_model 为多模态模型（如 qwen3.5-flash）且账号已开通。\n"
            f"  原始信息：{message}"
        )
    if status == 401 or "invalid api-key" in lower:
        return f"OCR 调用失败(status={status})：API Key 无效或未配置 llm_api_key。原始信息：{message}"
    return f"OCR 调用失败(status={status}): {message}"


def _content_text(content) -> str:
    """DashScope content 归一化为纯文本（与 dashscope_chat 同规则，避免跨层耦合）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for elem in content:
            if isinstance(elem, str):
                parts.append(elem)
            elif isinstance(elem, dict):
                text = elem.get("text")
                if text:
                    parts.append(text)
        return "".join(parts)
    return ""
