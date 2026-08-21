"""测试用 Fake 模型：确定性输出，支持脚本化 tool_calls，无需网络与 Key。"""
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """按 script 队列依次返回预设消息；队列耗尽时返回默认回答。"""

    model_name: str = "fake-chat"
    script: list[BaseMessage] = []

    def _next(self) -> BaseMessage:
        if self.script:
            return self.script.pop(0)
        return AIMessage(content="（模拟模型默认回答）")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools, **kwargs):
        # 模拟模型支持工具绑定；实际行为由 script 控制
        return self


class FakeEmbeddings(Embeddings):
    """基于字符序号的确定性向量（固定长度 32），供检索逻辑测试。"""

    def _vec(self, text: str) -> list[float]:
        vec = [float(ord(c)) for c in text]
        if len(vec) < 32:
            vec += [0.0] * (32 - len(vec))
        return vec[:32]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]
