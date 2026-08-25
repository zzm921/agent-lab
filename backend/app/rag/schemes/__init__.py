"""各代 RAG 方案：naive（朴素）/ advanced（高级）/ modular（模块化）。"""
from app.rag.schemes.advanced import AdvancedRagScheme
from app.rag.schemes.modular import ExecutionPlan, ModuleCall, ModularRagScheme
from app.rag.schemes.naive import NaiveRagScheme

__all__ = ["NaiveRagScheme", "AdvancedRagScheme", "ModularRagScheme", "ExecutionPlan", "ModuleCall"]
