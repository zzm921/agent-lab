"""前置路由与查询理解（pre-retrieval）：语义路由 / 查询改写 / 查询分解 / 指代消解。"""
from app.rag.routing.classifier import RouteDecision, build_classifier
from app.rag.routing.deictic_resolver import DeicticResolver, build_deictic_resolver
from app.rag.routing.query_decompose import QueryDecomposer, build_decomposer
from app.rag.routing.query_rewrite import QueryRewriter, build_rewriter

__all__ = [
    "RouteDecision",
    "build_classifier",
    "DeicticResolver",
    "build_deictic_resolver",
    "QueryDecomposer",
    "build_decomposer",
    "QueryRewriter",
    "build_rewriter",
]
