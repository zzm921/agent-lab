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
