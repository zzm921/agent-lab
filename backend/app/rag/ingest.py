"""RAG 离线建库：把内嵌语料预写入指定 RAG 方案的向量库（Qdrant/ES）。

在线服务启动时不再现场入库（避免能力加载慢）；本模块供 scripts/ingest_*.py 调用，
在线上前完成建库。幂等：语料未变则跳过；force=True 强制清空重建。
"""
from __future__ import annotations

from app.config import settings
from app.core.errors import ConfigError
from app.llm.client import create_chat_model, create_embeddings
from app.memory.corpus import KNOWLEDGE_CORPUS
from app.rag.manager import RagManager


def build_corpus(scheme_ids: list[str] | None = None, force: bool = False) -> list[dict]:
    """构建 embeddings/chat 并只把语料写入指定方案（缺省为全部已注册方案）。"""
    embeddings = create_embeddings(fake=False)  # 未配 Embedding Key 时抛 ConfigError
    llm = None
    try:
        llm = create_chat_model(fake=False, scenario="rag_rewrite")  # advanced 重写专用场景；未配则回退规则重写
    except ConfigError:
        llm = None
    rag = RagManager(settings, embeddings, top_k=settings.rag_top_k, llm=llm, scheme_ids=scheme_ids)
    if force:
        for scheme in rag.schemes.values():
            scheme.store.clear()
    rag.ingest_all(KNOWLEDGE_CORPUS)
    return rag.list()
