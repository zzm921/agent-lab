"""两层重试机制：工具层直接重试（透明重试） + Agent 层思考后重试。

分层原则：
- 工具层重试（Transparent Retry）：对与参数无关的瞬时错误（网络超时/连接重置/
  服务端 5xx/限流 429 等），用相同参数直接重试，最多 tool_retry_max 次，
  指数退避 + 抖动；模型完全无感知，只看到最终结果。
- Agent 层重试（Model-mediated Retry）：对参数/策略错误（400/404/401/403/业务逻辑），
  把结构化错误文本（含错误类型/详情/建议）返回给模型，由模型重新思考、调整参数或换工具
  后再调用（LangGraph ReAct 循环天然具备）；harness 按「同工具连续失败次数」设上限
  （agent_retry_max），达到后提示模型改用其它工具，避免无限烧迭代。
"""
from __future__ import annotations

import asyncio
import random

from app.core.errors import RetryableToolError
from app.core.events import event

# 内置可重试异常：超时与连接类错误均属瞬时抖动，与参数无关，直接重试
_RETRYABLE_BUILTIN = (
    TimeoutError,
    ConnectionError,  # 含 ConnectionResetError / ConnectionAbortedError 等
    ConnectionRefusedError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


def is_retryable_status(code) -> bool:
    """瞬时状态码判定：限流 429 或服务端 5xx 可重试（参数正确，只是时机不对）。"""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return False
    return code == 429 or 500 <= code <= 599


def _status_of(exc) -> int | None:
    """尽力从异常中提取 HTTP 状态码（httpx / requests 风格）。"""
    for attr in ("status_code", "status"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def is_retryable_exception(exc) -> bool:
    """判定异常是否为可重试的瞬时错误（工具层直接重试）：
    - RetryableToolError 显式标记；
    - 网络/超时/连接类内置异常；
    - httpx 的 TimeoutException / NetworkError（Connect/Read/Protocol 等）；
    - 携带 429 或 5xx 状态码的异常。
    其余（400/404/401/403/业务逻辑等）属于确定性错误，不直接重试，交给模型思考后重试。
    """
    if isinstance(exc, RetryableToolError):
        return True
    if isinstance(exc, _RETRYABLE_BUILTIN):
        return True
    if isinstance(exc, OSError):
        # 其余 OSError（如 FileNotFoundError）属确定性错误，不直接重试
        return False
    if _is_httpx_retryable(exc):
        return True
    return is_retryable_status(_status_of(exc))


def _is_httpx_retryable(exc) -> bool:
    """识别 httpx 的网络/超时异常（不硬依赖 httpx 包，避免可选依赖缺失时崩溃）。"""
    try:
        import httpx
    except ImportError:
        return False
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.NetworkError):  # 含 ConnectError / ReadError / RemoteProtocolError
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_status(exc.response.status_code if exc.response else None)
    return False


def exp_backoff_value(attempt: int, base: float, cap: float) -> float:
    """纯指数退避值（不含抖动）：min(cap, base * 2^(attempt-1))，供展示退避曲线。"""
    base = max(0.0, float(base))
    cap = max(base, float(cap))
    return round(min(base * (2 ** max(0, attempt - 1)), cap), 3)


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    """指数退避 + 抖动：实际睡眠 = 纯指数值 × (0.5 ~ 1.0)。"""
    return round(exp_backoff_value(attempt, base, cap) * random.uniform(0.5, 1.0), 3)


def format_tool_error(tool_name: str, exc: BaseException | None, retried: int = 0) -> str:
    """构造返回给模型的结构化错误文本：含错误类型/详情/建议，便于模型解析后调整参数或换工具。

    工具层重试已耗尽（retried>0）时提示可稍后重试；永久错误提示修正参数或换方式。
    """
    kind = "瞬时错误" if is_retryable_exception(exc) else "参数或策略错误"
    detail = str(exc) if exc else "未知错误"
    if retried > 0:
        suggestion = (
            f"这是{kind}，已用相同参数自动重试 {retried} 次仍失败。"
            "可稍后重试；或修正参数、换一种调用方式、改用其它工具。"
        )
    else:
        suggestion = (
            "请修正参数或换一种调用方式重试；不要用相同参数反复重试。"
            "若为权限/资源/不存在类问题，可改用其它工具或如实向用户说明。"
        )
    return (
        f"工具 {tool_name} 执行失败。\n"
        f"错误类型：{kind}\n"
        f"错误详情：{detail}\n"
        f"建议：{suggestion}"
    )


async def invoke_with_retry(run, tool_name: str, settings=None, emit=None):
    """工具层重试（透明重试）：对瞬时错误用相同参数直接重试，指数退避 + 抖动。

    参数：
        run: 零参数 async 可调用，执行一次工具（如 `lambda: tool.ainvoke(args)`）。
        tool_name: 工具名（事件上报用）。
        settings: 配置对象；读取 tool_retry_max / tool_retry_base_delay / tool_retry_max_delay。
        emit: 事件回调；每次重试前发射 tool_retry 事件（前端展示「重试中 n/m」）。
    返回 (result, success, error, retries)：
        success=True → result 为工具输出（可能是 Command，透传）；error=None；
        success=False → result=None，error 为最后一次异常（重试耗尽或不可重试）；
        retries 为实际直接重试的次数（供格式化错误文案用）。
    """
    max_attempts = max(1, int(getattr(settings, "tool_retry_max", 3)))
    base = float(getattr(settings, "tool_retry_base_delay", 0.5))
    cap = float(getattr(settings, "tool_retry_max_delay", 4.0))
    attempt = 1
    retries = 0
    while True:
        try:
            result = await run()
            return result, True, None, retries
        except Exception as exc:  # noqa: BLE001
            if not is_retryable_exception(exc) or attempt > max_attempts:
                return None, False, exc, retries
            delay = backoff_delay(attempt, base, cap)
            if emit is not None:
                emit(
                    event(
                        "tool_retry",
                        tool=tool_name,
                        attempt=attempt,
                        max=max_attempts,
                        delay=delay,
                        base_delay=exp_backoff_value(attempt, base, cap),
                        reason=str(exc),
                    )
                )
            await asyncio.sleep(delay)
            attempt += 1
            retries += 1
