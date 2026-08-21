"""统一异常类型。"""


class AgentError(Exception):
    """Agent 运行过程中的可预期错误。"""


class ToolError(Exception):
    """工具执行失败。"""


class ConfigError(Exception):
    """配置缺失或错误（如缺少 API Key）。"""
