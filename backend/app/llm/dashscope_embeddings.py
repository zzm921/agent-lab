"""DashScope 官方 SDK Embedding 适配：稠密向量 + 稀疏向量（混合检索用）。

- 稠密：text-embedding-v3（OpenAI 兼容模型，维度默认 1024）；
- 稀疏：text-sparse-embedding-v1（用于 Qdrant 混合检索，输出 token_ids + 权重）；
  账号未开通该模型时回退本地 n-gram 稀疏向量，保证 advanced 方案的混合检索仍可用。
- 实现 LangChain Embeddings 接口（embed_query / embed_documents），
  并额外提供 embed_sparse_* 供 QdrantStore 的 sparse 向量使用。
"""
from __future__ import annotations

import hashlib
import logging
import re

from http import HTTPStatus

import dashscope
from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings

from app.core.errors import ConfigError

logger = logging.getLogger(__name__)

# 本地稀疏向量桶数：字符 n-gram 哈希到固定区间，避免 Qdrant 稀疏索引维度过大
_SPARSE_BUCKETS = 2**16

# 稀疏模型回退仅警告一次，避免入库/检索时反复刷屏
_SPARSE_FALLBACK_WARNED = False


def _warn_sparse_fallback(model: str, exc) -> None:
    global _SPARSE_FALLBACK_WARNED
    if _SPARSE_FALLBACK_WARNED:
        return
    _SPARSE_FALLBACK_WARNED = True
    logger.warning(
        "稀疏模型 %s 不可用（%s），advanced 方案回退本地 n-gram 稀疏向量（仅警告一次）",
        model,
        exc,
    )


def local_sparse(text: str, buckets: int = _SPARSE_BUCKETS) -> dict:
    """不依赖外部模型的确定性稀疏向量：中文字符二元组 + 英文单词，词频作权重。

    账号未开通 text-sparse-embedding-v1 时作为回退，为混合检索提供关键词信号。
    """
    counts: dict[int, int] = {}

    def _hit(seed: str) -> int:
        return int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:8], 16) % buckets

    for i in range(len(text) - 1):  # 中文等无空格语言的字符二元组
        idx = _hit(text[i : i + 2])
        counts[idx] = counts.get(idx, 0) + 1
    for word in re.findall(r"[A-Za-z0-9]+", text):  # 英文单词
        idx = _hit("w:" + word.lower())
        counts[idx] = counts.get(idx, 0) + 1
    indices = sorted(counts)
    return {"indices": indices, "values": [float(counts[i]) for i in indices]}


class DashScopeEmbeddings(Embeddings):
    """基于 DashScope 官方 SDK 的 Embedding 模型（稠密 + 稀疏）。"""

    api_key: str = ""
    model: str = "text-embedding-v3"
    sparse_model: str = "text-sparse-embedding-v1"
    text_type: str = "document"  # DashScope embedding 的 text_type 参数（document/query）

    def __init__(self, api_key: str, model: str = "text-embedding-v3", sparse_model: str = "text-sparse-embedding-v1", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.sparse_model = sparse_model
        dashscope.api_key = api_key

    def _call(self, model: str, input_texts: list[str], text_type: str):
        resp = TextEmbedding.call(model=model, input=input_texts, text_type=text_type)
        if resp.status_code != HTTPStatus.OK:
            raise ConfigError(
                f"DashScope Embedding 调用失败(status={resp.status_code}): {getattr(resp, 'message', '')}"
            )
        return resp

    @staticmethod
    def _output_embeddings(resp) -> list[dict]:
        """兼容新旧 SDK：新版 resp.output 为 dict（{"embeddings": [...]}），旧版为对象。"""
        output = resp.output
        if isinstance(output, dict):
            return output.get("embeddings") or []
        return list(getattr(output, "embeddings") or [])

    def embed_query(self, text: str) -> list[float]:
        resp = self._call(self.model, [text], text_type="query")
        return list(self._output_embeddings(resp)[0]["embedding"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._call(self.model, texts, text_type="document")
        return [list(item["embedding"]) for item in self._output_embeddings(resp)]

    # ---- 稀疏向量（混合检索） ----

    def embed_sparse_query(self, text: str) -> dict:
        """返回 {indices, values}，供 Qdrant SparseVector 使用。

        账号未开通稀疏模型（400 Model not exist）时回退本地 n-gram 稀疏向量。
        """
        try:
            resp = self._call(self.sparse_model, [text], text_type="query")
            return self._to_sparse(self._output_embeddings(resp)[0])
        except ConfigError as exc:
            _warn_sparse_fallback(self.sparse_model, exc)
            return local_sparse(text)

    def embed_sparse_documents(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        try:
            resp = self._call(self.sparse_model, texts, text_type="document")
            return [self._to_sparse(item) for item in self._output_embeddings(resp)]
        except ConfigError as exc:
            _warn_sparse_fallback(self.sparse_model, exc)
            return [local_sparse(t) for t in texts]

    @staticmethod
    def _to_sparse(item: dict) -> dict:
        return {"indices": list(item.get("token_ids") or []), "values": list(item.get("embeddings") or [])}
