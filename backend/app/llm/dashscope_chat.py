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
from pydantic import BaseModel, PrivateAttr


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


def _tool_to_ds(tool: BaseTool | dict | type) -> dict:
    """BaseTool / dict / Pydantic 模型类 → DashScope tools 条目（OpenAI 风格 function schema）。

    with_structured_output 会以 [Pydantic 模型类] 形式调用 bind_tools，需将模型类
    转为 JSON Schema；dict 直接透传（已是最外层 function schema）。
    """
    if isinstance(tool, dict):
        return tool
    if isinstance(tool, type) and issubclass(tool, BaseModel):
        schema = tool.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": schema.get("title") or tool.__name__,
                "description": schema.get("description") or "",
                "parameters": schema,
            },
        }
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

    def bind_tools(self, tools: list[BaseTool | dict | type], **kwargs: Any):
        """返回绑定后的新副本（不修改原实例），符合 LangChain bind_tools 语义。

        with_structured_output 会以 [Pydantic 模型类] 形式调用 bind_tools；若原地修改，
        会把结构化输出工具污染到共享的 llm 实例（如 reflection 评审器把 CritiqueResult
        绑到生成器上，导致生成器误以为存在该工具）。
        """
        bound = self.model_copy(deep=True)
        bound._tools = [_tool_to_ds(t) for t in tools]
        tc = kwargs.get("tool_choice")
        if tc == "any":  # OpenAI 语义；DashScope 仅支持 none/auto/指定函数
            tc = "auto"
        bound._tool_choice = tc
        bound._model_settings.update(
            {k: v for k, v in kwargs.items() if k != "tool_choice" and not k.startswith("ls_")}
        )
        return bound

    def bind(self, **kwargs: Any):
        """无工具时的绑定：在副本上保存额外模型参数，不污染原实例。"""
        bound = self.model_copy(deep=True)
        bound._model_settings.update(kwargs)
        return bound

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
