"""RAG 建库前置处理包：文档接入 → 解析路由 → 清洗 → 报告/DLQ。"""
from app.rag.preprocess.pipeline import run_pipeline

__all__ = ["run_pipeline"]
