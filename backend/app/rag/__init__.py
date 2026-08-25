"""RAG 方案包：多 RAG 方案实现（naive / advanced / modular / …）。

目录结构（按职责分层，避免平铺混乱）：
- 顶层：抽象与基础设施（base / manager / ingest）；
- schemes/：各代 RAG 方案（naive / advanced / modular）；
- routing/：前置路由与查询理解（语义路由 / 改写 / 分解 / 指代消解）；
- retrieval/：检索与后处理（召回融合 / 重排 / 压缩 / 多跳规划-执行-验证 / 充分性闸门）；
- docs/：架构与测试方案等 Markdown 文档；
- corpus/：知识库语料 Markdown。
"""
from app.rag.base import RagScheme, RetrieveResult
from app.rag.manager import RagManager
from app.rag.retrieval.context_compress import ContextCompressor
from app.rag.routing.classifier import RouteDecision
from app.rag.routing.query_decompose import QueryDecomposer
from app.rag.schemes.advanced import AdvancedRagScheme
from app.rag.schemes.modular import ExecutionPlan, ModuleCall, ModularRagScheme

__all__ = [
    "RagScheme",
    "RetrieveResult",
    "RagManager",
    "AdvancedRagScheme",
    "ModularRagScheme",
    "ExecutionPlan",
    "ModuleCall",
    "RouteDecision",
    "QueryDecomposer",
    "ContextCompressor",
]
