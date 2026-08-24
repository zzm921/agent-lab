"""统一异常类型。"""


class AgentError(Exception):
    """Agent 运行过程中的可预期错误。"""


class ToolError(Exception):
    """工具执行失败。"""


class RetryableToolError(ToolError):
    """工具执行失败的「瞬时错误」标记：网络超时/连接重置/服务端 5xx/限流 429 等。

    与失败原因无关、换时间点用相同参数大概率能成功的错误抛出此类，
    由工具层重试（透明重试）直接重试；其余错误类型不直接重试，返回给模型思考后重试。
    """


class ConfigError(Exception):
    """配置缺失或错误（如缺少 API Key）。"""


class LLMError(Exception):
    """LLM 调用失败：带场景/模型/参数/方法上下文，便于定位具体实例。

    LoggedChatModel 把底层供应商异常统一包装为本异常，并保留 cause。
    """

    def __init__(
        self,
        message: str = "",
        *,
        scenario: str = "",
        model: str = "",
        params: dict | None = None,
        method: str = "",
        cause: Exception | None = None,
    ):
        self.scenario = scenario
        self.model = model
        self.params = params or {}
        self.method = method
        self.cause = cause
        detail = message or (f"{type(cause).__name__}: {cause}" if cause else "未知错误")
        super().__init__(f"[LLM {scenario}/{model}] {method or 'call'} 失败：{detail}")
