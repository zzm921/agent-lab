"""RAG 方案包：多 RAG 方案实现（naive / advanced / modular / …）。"""
from app.rag.advanced import AdvancedRagScheme
from app.rag.base import RagScheme, RetrieveResult
from app.rag.classifier import RouteDecision
from app.rag.context_compress import ContextCompressor
from app.rag.manager import RagManager
from app.rag.modular import ExecutionPlan, ModuleCall, ModularRagScheme
from app.rag.query_decompose import QueryDecomposer

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
