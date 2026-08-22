"""DashScope 原生 SDK 适配单测：reason/output 分离、消息转换、工具调用增量合并。"""
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from app.llm.dashscope_chat import (
    DashScopeChatModel,
    DashScopeTurn,
    _parse_message,
    _to_dashscope_messages,
    _to_lc_tool_calls,
    _to_tool_call_chunks,
    _tool_to_ds,
)


def test_parse_message_distinguishes_reason_and_output():
    """响应 message 解析：reasoning_content(reason) 与 content(output) 明确分离。"""
    msg = {
        "role": "assistant",
        "content": "最终答案是 391",
        "reasoning_content": "先竖式计算 23*17...",
        "tool_calls": [],
    }
    turn = _parse_message(msg)
    assert isinstance(turn, DashScopeTurn)
    assert turn.reasoning == "先竖式计算 23*17..."
    assert turn.output == "最终答案是 391"
    assert turn.tool_calls == []


def test_parse_message_with_tool_calls():
    """响应 message 解析：携带工具调用时 output/reasoning 可为空。"""
    msg = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "",
        "tool_calls": [
            {
                "index": 0,
                "id": "call_x",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'},
            }
        ],
    }
    turn = _parse_message(msg)
    assert turn.output == ""
    assert turn.reasoning == ""
    assert turn.tool_calls[0]["function"]["name"] == "calculator"


def test_to_lc_tool_calls():
    """DashScope 完整 tool_calls → LangChain 格式（arguments 解析为 dict）。"""
    ds = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
        }
    ]
    lc = _to_lc_tool_calls(ds)
    assert lc == [
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "call_1", "type": "tool_call"}
    ]


def test_to_lc_tool_calls_bad_json_guards():
    """arguments 非合法 JSON 时兜底为空 dict，不抛异常。"""
    ds = [{"id": "c", "function": {"name": "x", "arguments": "{oops"}}]
    assert _to_lc_tool_calls(ds)[0]["args"] == {}


def test_tool_call_chunks_accumulate_via_add():
    """流式增量 tool_calls → ToolCallChunk，经 AIMessageChunk.__add__ 合并为完整调用。"""
    c1 = AIMessageChunk(
        content="",
        tool_call_chunks=_to_tool_call_chunks(
            [
                {
                    "index": 0,
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression": "1'},
                }
            ]
        ),
    )
    c2 = AIMessageChunk(
        content="",
        tool_call_chunks=_to_tool_call_chunks(
            [{"index": 0, "id": "", "function": {"name": "", "arguments": '+1"}'}}]
        ),
    )
    merged = c1 + c2
    assert merged.tool_calls == [
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "call_9", "type": "tool_call"}
    ]


def test_to_dashscope_messages_mapping():
    """LangChain 消息 → DashScope 请求格式：system/user/assistant(tool_calls)/tool。"""
    msgs = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="你好"),
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "1"}, "id": "c1"}]),
        ToolMessage(content="2", tool_call_id="c1"),
    ]
    out = _to_dashscope_messages(msgs)
    assert out[0] == {"role": "system", "content": "你是助手"}
    assert out[1] == {"role": "user", "content": "你好"}
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["function"]["name"] == "calculator"
    assert out[3] == {"role": "tool", "content": "2", "tool_call_id": "c1"}


def test_tool_to_ds_from_dict_passthrough():
    """dict 类型工具直接透传。"""
    d = {"type": "function", "function": {"name": "x"}}
    assert _tool_to_ds(d) is d


def test_to_message_carries_reasoning_in_kwargs():
    """DashScopeTurn → AIMessage：content=output，reasoning_content 放入 additional_kwargs。"""
    model = DashScopeChatModel(api_key="k")
    turn = DashScopeTurn(reasoning="思考一下", output="答案", finish_reason="stop")
    msg = model._to_message(turn)
    assert msg.content == "答案"
    assert msg.additional_kwargs["reasoning_content"] == "思考一下"
    assert msg.tool_calls == []


def test_to_message_with_tool_calls():
    """DashScopeTurn 携带 tool_calls 时转换为 LangChain AIMessage.tool_calls。"""
    model = DashScopeChatModel(api_key="k")
    turn = DashScopeTurn(
        tool_calls=[
            {"id": "c", "function": {"name": "calculator", "arguments": '{"expression": "3"}'}}
        ]
    )
    msg = model._to_message(turn)
    assert msg.tool_calls[0]["name"] == "calculator"
    assert msg.tool_calls[0]["args"] == {"expression": "3"}


def test_payload_sets_thinking_and_tools():
    """请求参数：enable_thinking 生效，tools 透传，temperature 缺省用实例值。"""
    model = DashScopeChatModel(api_key="k", temperature=0.5)
    model = model.bind_tools([{"type": "function", "function": {"name": "calculator"}}])
    payload = model._payload([HumanMessage(content="hi")], stream=False)
    assert payload["enable_thinking"] is True
    assert payload["result_format"] == "message"
    assert payload["tools"][0]["function"]["name"] == "calculator"
    assert payload["temperature"] == 0.5
    assert payload["messages"][0] == {"role": "user", "content": "hi"}


def test_bind_tools_does_not_mutate_original():
    """bind_tools / bind 返回副本，不污染原实例。

    回归：reflection 评审器用 with_structured_output 绑定 CritiqueResult 时，
    若原地修改会把该工具泄漏到共享的生成器 llm 上。
    """
    model = DashScopeChatModel(api_key="k")
    bound = model.bind_tools([{"type": "function", "function": {"name": "calculator"}}])
    assert bound._payload([HumanMessage(content="hi")], stream=False)["tools"][0]["function"]["name"] == "calculator"
    assert model._payload([HumanMessage(content="hi")], stream=False).get("tools") is None

    bound2 = model.bind(temperature=0.9)
    assert bound2._payload([HumanMessage(content="hi")], stream=False)["temperature"] == 0.9
    assert model._payload([HumanMessage(content="hi")], stream=False)["temperature"] == 0.3
