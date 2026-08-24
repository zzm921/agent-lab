"""LLM 模块：按业务场景统一管理模型与参数。

对外主入口为 LLMService（app.llm.service）；按需从 app.llm.client 取工厂函数。
"""
from app.llm.service import DEFAULT_PROFILES, LLMProfile, LLMService, LoggedChatModel

__all__ = ["LLMProfile", "LLMService", "LoggedChatModel", "DEFAULT_PROFILES"]
