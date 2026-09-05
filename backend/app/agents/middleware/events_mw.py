"""StreamEventsMiddleware：统一发射模型输出与工具执行事件（含 HITL 审批）。

替代旧的手写 tools 节点（app/tools/runner.py::make_tools_node）：
- awrap_model_call：直接调用模型的 astream 逐 token 生成，替代 factory 内 ainvoke 的
  一次性返回；按 DashScope 返回值类型分流——
  reasoning_content(reason) → thinking 事件（思考过程，前端灰斜体），
  content(output) → message 事件（最终输出）；
  每个片段实时打印到后端控制台，工具调用经 tool_call_chunks 合并回 AIMessage。
- awrap_tool_call：按审批策略（harness.should_approve，含强制 HITL 工具）决定是否 interrupt 审批，
  随后发射 tool_start / tool_end，并执行工具（两层重试：工具层透明重试 + Agent 层上限；
  异常经结构化错误文本返回给模型思考后重试）。
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command, interrupt

from app.agents.context_manage import maybe_offload
from app.agents.harness import should_approve
from app.core.errors import RetryableToolError
from app.core.events import emit_text, event
from app.security import StreamMasker, is_untrusted_tool, mask_sensitive, scan_output, wrap_untrusted
from app.tools.retry import format_tool_error, invoke_with_retry


def _content_text(chunk) -> str:
    """从模型输出块中提取纯文本内容（兼容 str 与 content block 列表）。"""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block["text"])
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content) if content else ""


def _reasoning_text(chunk) -> str:
    """从模型输出块中提取思考内容（DashScope 的 reasoning_content）。"""
    extra = getattr(chunk, "additional_kwargs", None) or {}
    reasoning = extra.get("reasoning_content")
    if reasoning:
        return reasoning if isinstance(reasoning, str) else str(reasoning)
    reasoning = getattr(chunk, "reasoning_content", None)
    if reasoning:
        return reasoning if isinstance(reasoning, str) else str(reasoning)
    return ""


def resolve_guards(settings) -> dict:
    """按配置解析输出 Guardrail 开关（security.md 第三层）。settings 为 None 时全部关闭。"""
    sec = bool(getattr(settings, "security_enabled", True))
    return {
        "mask_output": sec and bool(getattr(settings, "mask_sensitive_output", True)),
        "block_output": sec and bool(getattr(settings, "guard_output", True)),
    }


def _maybe_wrap_tool_result(msg: ToolMessage, name: str, enabled: bool = True) -> ToolMessage:
    """不可信外部内容工具（网页/命令）返回经来源分级包装后再给模型，防间接注入。

    只包装「外部内容来源」工具（web_search / run_command）的成功返回，
    内部工具（calculator 等）原样透传，避免污染正常工具结果。
    """
    if not enabled or not is_untrusted_tool(name):
        return msg
    content = getattr(msg, "content", None)
    return ToolMessage(content=wrap_untrusted(str(content), name), tool_call_id=msg.tool_call_id)


async def stream_model_call(llm, messages, emit, *, tools=None, tool_choice=None, model_settings=None, system_prompt="", output_event="message", guards: dict | None = None):
    """用 astream 逐 token 生成并实时下发 thinking/message 事件，返回合并后的 AIMessage。

    供 create_agent 中间件与手写 StateGraph 节点（plan-execute / reflection）共用：
    - 有工具时 bind_tools，否则 bind；system_prompt 非空时前置为系统消息；
    - reasoning_content(reason) → thinking 事件（思考过程），content(output) → output_event 事件
      （默认 message；reflection 修订稿用 revise，前端按修订稿样式展示）；
    - tool_calls 经 chunk 合并回 AIMessage，交由调用方路由到工具循环；
    - 输出 Guardrail（security.md 第三层）：guards 控制敏感脱敏（流式实时生效）与
      违规阻断（全文扫描后追加 guard_refused 提示）；guards 为 None 时不做输出防护，
      调用方需传入 resolve_guards(settings) 的结果。
    """
    guards = guards or {}
    mask_enabled = bool(guards.get("mask_output"))
    block_enabled = bool(guards.get("block_output"))
    masker = StreamMasker() if mask_enabled else None
    bound = (
        llm.bind_tools(tools, tool_choice=tool_choice, **(model_settings or {}))
        if tools
        else llm.bind(**(model_settings or {}))
    )
    if system_prompt:
        messages = [SystemMessage(system_prompt), *messages]

    chunks = []
    reason_texts: list[str] = []
    output_texts: list[str] = []
    async for chunk in bound.astream(messages):
        chunks.append(chunk)
        reasoning = _reasoning_text(chunk)
        if reasoning:
            print(f"[model-stream] reason: {reasoning!r}", flush=True)
            reason_texts.append(reasoning)
            emit_text(emit, "thinking", reasoning)
        text = _content_text(chunk)
        if text:
            print(f"[model-stream] output: {text!r}", flush=True)
            output_texts.append(text)
            if masker is not None:
                # 流式脱敏：只发射已到安全边界的脱敏前缀，未完成 token 留在缓冲
                masked = masker.push(text)
                if masked:
                    emit_text(emit, output_event, masked)
            else:
                emit_text(emit, output_event, text)
    # 冲刷脱敏缓冲尾部（末尾未到空白边界的最后一个 token）
    if masker is not None:
        tail = masker.flush()
        if tail:
            emit_text(emit, output_event, tail)

    if not chunks:
        raise RuntimeError("模型流式调用未返回任何内容")

    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    tool_calls = getattr(merged, "tool_calls", None) or []
    reasoning_full = "".join(reason_texts)
    raw_output = "".join(output_texts)
    # 落库（会话历史）也保存脱敏后的内容，避免后续轮次再次暴露敏感数据
    msg = AIMessage(
        content=mask_sensitive(raw_output) if mask_enabled else raw_output,
        tool_calls=tool_calls,
        additional_kwargs={"reasoning_content": reasoning_full} if reasoning_full else {},
    )
    print(f"[model-stream] done: reasoning_len={len(reasoning_full)} output_len={len(msg.content)} tool_calls={tool_calls!r}", flush=True)

    # 输出 Guardrail：敏感数据泄露阻断提示（流式已发，事后提示；只针对最终答案类事件）
    if block_enabled and output_event in ("message", "revise"):
        verdict = scan_output(raw_output)
        if verdict.blocked:
            emit(event("guard_refused", reason=verdict.reason, matched=verdict.matched))

    # 只有工具调用且无思考/输出时补占位文案，保证思考区非空
    if tool_calls and not reasoning_full and not msg.content:
        emit_text(emit, "thinking", "（正在决定下一步行动）")
    return msg


async def _execute_tool_call(request, handler, emit, do_approval: bool, harness=None):
    """执行单个工具调用：护栏检查（熔断/次数上限）+ 可选 HITL 审批 + 工具事件 + 异常兜底。

    返回最终 ToolMessage 或 Command。do_approval=False 时跳过 interrupt，
    供无 checkpointer 的 multi-agent worker 使用。harness 为 None 时跳过护栏。
    """
    call = request.tool_call
    name = call["name"]
    args = call.get("args", {})
    cid = call.get("id")
    session_id = request.runtime.config["configurable"]["thread_id"]

    # 护栏：熔断 / 工具调用次数上限。命中直接短路，不执行也不触发审批。
    # 熔断按「工具+参数」计：相同参数重复失败才拦截，换参数重试始终放行
    if harness is not None and not harness.circuit_allows(session_id, name, args):
        msg = f"工具 {name} 在当前参数下已连续失败触发熔断保护，请更换参数重试或改用其它工具。"
        emit(event("tool_start", tool=name, args=args))
        emit(event("tool_end", tool=name, args=args, result=msg, success=False))
        return ToolMessage(content=msg, tool_call_id=cid)
    if harness is not None and harness.tool_calls_exceeded(session_id):
        msg = f"本轮工具调用已达上限（{harness.tool_calls_limit()} 次），已停止执行。"
        emit(event("tool_start", tool=name, args=args))
        emit(event("tool_end", tool=name, args=args, result=msg, success=False))
        return ToolMessage(content=msg, tool_call_id=cid)

    # 故障注入钩子（验证两层重试与熔断机制用）：
    # 按注入类型分类：
    # - 参数/业务错误（error/business/400/401/403/404，retryable=False）→ 不直接重试，
    #   错误文本直接返回给模型思考后重试（不执行工具，故不触发审批）；
    # - 瞬时错误（timeout/conn_reset/dns/429/5xx，retryable=True）→ 仍走 HITL 审批，
    #   批准后进入工具层透明重试（发 tool_retry 事件，指数退避，耗尽返回结构化错误给模型）。
    fault = harness.fault_spec(name) if harness is not None else None
    if fault is not None and not fault["retryable"]:
        emit(event("tool_start", tool=name, args=args))
        emit(event("tool_end", tool=name, args=args, result=fault["message"], success=False))
        harness.record_tool_failure(session_id, name, args)
        return ToolMessage(content=fault["message"], tool_call_id=cid)

    # Agent 层重试上限：同一工具连续失败（可换参数）达到上限后，直接提示模型改用其它工具，不执行也不审批
    if harness is not None and harness.tool_exhausted(session_id, name):
        msg = f"工具 {name} 已连续失败 {harness.agent_retry_limit()} 次，请停止使用该工具，改用其它工具或向用户说明。"
        emit(event("tool_start", tool=name, args=args))
        emit(event("tool_end", tool=name, args=args, result=msg, success=False))
        return ToolMessage(content=msg, tool_call_id=cid)

    if do_approval:
        decision = interrupt({"tool_calls": [{"name": name, "args": args, "id": cid}]})
        action = decision.get("action", "approve")
        if action == "reject":
            emit(event("tool_end", tool=name, args=args, result="用户拒绝了该工具调用", success=False))
            return ToolMessage(
                content="用户拒绝了该工具调用，请改用其它方式或询问用户。",
                tool_call_id=cid,
            )
        modified = decision.get("modified_args") or {}
        if cid in modified:
            request = request.override(tool_call={**call, "args": modified[cid]})
        elif name in modified:
            # 兜底：部分模型未回传工具调用 id 时前端按名称回填，按名称匹配保证修改生效
            request = request.override(tool_call={**call, "args": modified[name]})

    effective = request.tool_call
    name = effective["name"]
    args = effective.get("args", {})
    settings = getattr(harness, "settings", None) if harness is not None else None
    emit(event("tool_start", tool=name, args=args))

    async def _run():
        if fault is not None:  # 瞬时故障注入：每次尝试均模拟瞬时失败，直至工具层直接重试耗尽
            raise RetryableToolError(fault["message"])
        return await handler(request)

    # 工具层透明重试（仅瞬时错误直接重试，指数退避+抖动；参数/业务错误不重试直接返回）
    result, success, error, retries = await invoke_with_retry(_run, name, settings, emit)
    if success:
        if isinstance(result, Command):
            return result
        content = getattr(result, "content", None)
        text = str(content)
        info = None
        # 大文件落盘（预算裁剪）：单条工具输出超阈值 → 写盘 + 上下文只留指针
        if settings is not None and getattr(settings, "context_mgmt_enabled", True) \
                and getattr(settings, "context_offload_enabled", True):
            text, info = maybe_offload(text, session_id=session_id, tool_name=name, settings=settings)
            if info is not None:
                emit(event("context", kind="offload", tool=name, chars=info["chars"], file=info["file"]))
        emit(event("tool_end", tool=name, args=args, result=text, success=True))
        if harness is not None:
            harness.record_tool_success(session_id, name, args)
        # 来源可信分级：不可信外部内容工具（网页/命令/记忆）返回包装后再给模型，防间接注入
        mark = bool(getattr(settings, "security_enabled", True) and getattr(settings, "mark_untrusted", True))
        if info is not None:
            result = ToolMessage(content=text, tool_call_id=cid)
        return _maybe_wrap_tool_result(result, name, enabled=mark)
    # 失败：结构化错误文本返回给模型（Agent 层思考后重试：模型修正参数/换工具后再调）
    msg = format_tool_error(name, error, retried=retries)
    emit(event("tool_end", tool=name, args=args, result=msg, success=False))
    if harness is not None:
        harness.record_tool_failure(session_id, name, args)
    return ToolMessage(content=msg, tool_call_id=cid)


def _approval_policy(request) -> str:
    return request.runtime.config["configurable"]["approval_policy"]


class StreamEventsMiddleware(AgentMiddleware):
    """react / plan_execute / multi_agent orchestrator 通用：thinking/message + 工具事件 + HITL。"""

    def __init__(self, emit, harness=None):
        self._emit = emit
        self._harness = harness

    async def awrap_model_call(self, request, handler):
        """用 astream 逐 token 生成并实时下发（复用 stream_model_call），替代 ainvoke 的一次性返回。"""
        msg = await stream_model_call(
            request.model,
            request.messages,
            self._emit,
            tools=request.tools,
            tool_choice=request.tool_choice,
            model_settings=request.model_settings,
            system_prompt=request.system_prompt,
            guards=resolve_guards(getattr(self._harness, "settings", None)),
        )
        return ModelResponse(result=[msg], structured_response=None)

    async def awrap_tool_call(self, request, handler):
        # 审批判定统一收敛到护栏层（harness.should_approve：always 或强制 HITL 工具）
        name = request.tool_call["name"]
        return await _execute_tool_call(
            request, handler, self._emit,
            should_approve(_approval_policy(request), name),
            harness=self._harness,
        )


class WorkerEventsMiddleware(AgentMiddleware):
    """multi-agent worker 用：只发射工具事件，不触发 HITL（worker 无 checkpointer）。

    子代理作为工具被编排者调用，本身不持有 checkpointer，无法持久化 interrupt，
    因此工具审批统一收敛到编排者层（StreamEventsMiddleware.awrap_tool_call）。
    """

    def __init__(self, emit, harness=None):
        self._emit = emit
        self._harness = harness

    async def awrap_tool_call(self, request, handler):
        return await _execute_tool_call(request, handler, self._emit, do_approval=False, harness=self._harness)
