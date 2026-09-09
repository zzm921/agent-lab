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

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        """模拟 json_mode 结构化输出：解析 script 消息的 JSON 内容为 dict。

        langchain 默认的 json_mode 实现（prompt 注入 + 解析）与 script 机制不兼容，
        Fake 下会返回 None 而非解析结果；这里直接以 JsonOutputParser 解析 script 内容，
        使结构化输出主路径在测试/评测中可被覆盖。
        """
        method = kwargs.get("method") or "json_mode"
        if method != "json_mode":
            return super().with_structured_output(schema, include_raw=include_raw, **kwargs)
        from langchain_core.output_parsers import JsonOutputParser

        return self | JsonOutputParser()


class FakeEmbeddings(Embeddings):
    """基于字符序号的确定性向量（固定长度 32），供检索逻辑测试。"""

    def _vec(self, text: str) -> list[float]:
        vec = [float(ord(c)) for c in text]
        if len(vec) < 32:
            vec += [0.0] * (32 - len(vec))
        return vec[:32]

    def _sparse(self, text: str) -> dict:
        """确定性稀疏向量：字符序号作为索引、频次作为权重。"""
        counts: dict[int, int] = {}
        for ch in text:
            idx = ord(ch)
            counts[idx] = counts.get(idx, 0) + 1
        return {"indices": sorted(counts), "values": [float(counts[i]) for i in sorted(counts)]}

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_sparse_query(self, text: str) -> dict:
        return self._sparse(text)

    def embed_sparse_documents(self, texts: list[str]) -> list[dict]:
        return [self._sparse(t) for t in texts]
