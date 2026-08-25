"""RAG 离线建库：把内嵌语料预写入指定 RAG 方案的向量库（Qdrant/ES）。

在线服务启动时不再现场入库（避免能力加载慢）；本模块供 scripts/ingest_*.py 调用，
在线上前完成建库。幂等：语料未变则跳过；force=True 强制清空重建。
"""
from __future__ import annotations

from app.config import settings
from app.llm.client import create_embeddings
from app.memory.corpus import KNOWLEDGE_CORPUS
from app.rag.manager import RagManager


def build_corpus(scheme_ids: list[str] | None = None, force: bool = False) -> list[dict]:
    """构建 embeddings 并把语料写入指定方案（缺省为全部已注册方案）。

    方案内部需要 LLM 的阶段（Query 重写等）按命名场景从全局 LLMService 懒取模型，
    未配聊天 Key 时自动回退确定性规则实现，故建库不依赖聊天模型。
    """
    embeddings = create_embeddings(fake=False)  # 未配 Embedding Key 时抛 ConfigError
    rag = RagManager(settings, embeddings, top_k=settings.rag_top_k, scheme_ids=scheme_ids)
    if force:
        for scheme in rag.schemes.values():
            scheme.store.clear()
    rag.ingest_all(KNOWLEDGE_CORPUS)
    return rag.list()
