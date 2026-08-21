"""DashScope 原生 SDK 适配：自定义 BaseChatModel，明确区分 reason 与 output。

- 通过 dashscope.Generation.call（阿里云官方 SDK）调用通义千问，开启 enable_thinking=True；
- 响应中 `reasoning_content` 即「思考过程」(reason)，`content` 即「最终输出」(output)；
- 用 DashScopeTurn 数据结构明确承载这两类结果，供上层发射 thinking / message 事件；
- 支持工具调用（tools）与流式输出（stream=True, incremental_output=True）。

本模块保持 LangChain BaseChatModel 契约，因此可无缝用于 create_agent / 中间件架构。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Iterator

import dashscope
from dashscope import Generation
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr


@dataclass
class DashScopeTurn:
    """一次模型调用的结构化结果：明确区分 reason 与 output。

    - reasoning:    思考过程（reasoning_content，前端以灰色斜体展示）
    - output:       最终输出（content，前端回答区展示）
    - tool_calls:   工具调用（DashScope 原始结构：id/type/function.name/function.arguments）
    - finish_reason: 结束原因（stop / tool_calls / length / ...）
    """

    reasoning: str = ""
    output: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""


def _to_dashscope_messages(messages: list[BaseMessage]) -> list[dict]:
    """LangChain 消息列表 → DashScope 请求消息格式。

    仅透传 role/content/tool_calls/tool_call_id；reasoning_content 不写回请求。
    """
    result = []
    for m in messages:
        if isinstance(m, SystemMessage):
            result.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            result.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            item = {"role": "assistant", "content": m.content or ""}
            if getattr(m, "tool_calls", None):
                item["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                        "id": tc.get("id", ""),
                    }
                    for tc in m.tool_calls
                ]
            result.append(item)
        elif isinstance(m, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "content": m.content or "",
                    "tool_call_id": getattr(m, "tool_call_id", ""),
                }
            )
        else:
            result.append({"role": "user", "content": getattr(m, "content", "") or ""})
    return result


def _tool_to_ds(tool: BaseTool | dict) -> dict:
    """BaseTool 或 dict → DashScope tools 条目（OpenAI 风格 function schema）。"""
    if isinstance(tool, dict):
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": getattr(tool, "args", None) or {"type": "object", "properties": {}},
        },
    }


def _parse_message(message: dict) -> DashScopeTurn:
    """解析 DashScope 响应 message 字典：分离 reasoning_content(reason) 与 content(output)。"""
    return DashScopeTurn(
        reasoning=message.get("reasoning_content") or "",
        output=message.get("content") or "",
        tool_calls=list(message.get("tool_calls") or []),
        finish_reason="",
    )


def _to_lc_tool_calls(ds_tool_calls: list[dict]) -> list[dict]:
    """DashScope 完整 tool_calls → LangChain AIMessage.tool_calls。"""
    result = []
    for tc in ds_tool_calls:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            parsed_args = {}
        result.append(
            {
                "name": fn.get("name") or "",
                "args": parsed_args,
                "id": tc.get("id") or "",
                "type": "tool_call",
            }
        )
    return result


def _to_tool_call_chunks(ds_tool_calls: list[dict]) -> list[dict]:
    """DashScope 流式增量 tool_calls → LangChain ToolCallChunk（供 AIMessageChunk.__add__ 合并）。"""
    chunks = []
    for tc in ds_tool_calls:
        fn = tc.get("function", {})
        chunks.append(
            {
                "index": tc.get("index"),
                "id": tc.get("id") or "",
                "name": fn.get("name") or "",
                "args": fn.get("arguments") or "",
            }
        )
    return chunks


class DashScopeChatModel(BaseChatModel):
    """基于 DashScope 官方 SDK 的 ChatModel：流式、工具调用、思考(reasoning_content) 分离。"""

    model_name: str = "qwen-plus"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.3
    enable_thinking: bool = True

    _tools: list[dict] = PrivateAttr(default_factory=list)
    _tool_choice: Any = PrivateAttr(default=None)
    _model_settings: dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """配置 SDK 全局参数：API Key 与（可选的）服务地址。"""
        dashscope.api_key = self.api_key
        if self.base_url:
            dashscope.base_http_api_url = self.base_url.rstrip("/")

    @property
    def _llm_type(self) -> str:
        return "dashscope-chat"

    def bind_tools(self, tools: list[BaseTool | dict], **kwargs: Any):
        """保存工具与 tool_choice（与 factory._get_bound_model 的调用契约一致）。"""
        self._tools = [_tool_to_ds(t) for t in tools]
        self._tool_choice = kwargs.get("tool_choice")
        self._model_settings.update({k: v for k, v in kwargs.items() if k != "tool_choice"})
        return self

    def bind(self, **kwargs: Any):
        """无工具时的绑定：保存额外模型参数。"""
        self._model_settings.update(kwargs)
        return self

    def _payload(self, messages: list[BaseMessage], stream: bool) -> dict:
        """构造 Generation.call 请求参数。"""
        payload = {
            "model": self.model_name,
            "messages": _to_dashscope_messages(messages),
            "result_format": "message",
            "stream": stream,
            "enable_thinking": self.enable_thinking,
            "incremental_output": True,
        }
        if self._tools:
            payload["tools"] = self._tools
        if self._tool_choice is not None:
            payload["tool_choice"] = self._tool_choice
        payload.update(self._model_settings)
        if "temperature" not in payload:
            payload["temperature"] = self.temperature
        return payload

    def _to_message(self, turn: DashScopeTurn) -> AIMessage:
        """DashScopeTurn → LangChain AIMessage（reasoning_content 放入 additional_kwargs）。"""
        extra = {"reasoning_content": turn.reasoning} if turn.reasoning else {}
        return AIMessage(
            content=turn.output,
            tool_calls=_to_lc_tool_calls(turn.tool_calls),
            additional_kwargs=extra,
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """非流式调用：供 planner / reflection / worker 使用。"""
        resp = Generation.call(**self._payload(list(messages), stream=False))
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"DashScope 调用失败(status={resp.status_code}): {getattr(resp, 'message', '')}"
            )
        turn = _parse_message(resp.output.choices[0].message)
        turn.finish_reason = resp.output.choices[0].get("finish_reason") or ""
        return ChatResult(generations=[ChatGeneration(message=self._to_message(turn))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        """流式调用：逐 token 产出，reasoning_content 与 content 分别透出。"""
        for resp in Generation.call(**self._payload(list(messages), stream=True)):
            if resp.status_code != HTTPStatus.OK:
                raise RuntimeError(
                    f"DashScope 流式调用失败(status={resp.status_code}): {getattr(resp, 'message', '')}"
                )
            output = getattr(resp, "output", None)
            choices = output.get("choices") if isinstance(output, dict) else []
            if not choices:
                continue
            msg = choices[0].get("message") or {}
            reasoning = msg.get("reasoning_content") or ""
            content = msg.get("content") or ""
            extra = {"reasoning_content": reasoning} if reasoning else {}
            chunk = AIMessageChunk(
                content=content,
                additional_kwargs=extra,
                tool_call_chunks=_to_tool_call_chunks(msg.get("tool_calls") or []),
            )
            yield ChatGenerationChunk(message=chunk)
