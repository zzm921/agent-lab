"""Agent 评测与 RAG 评测两个独立子包。

- eval.agent：Agent 全量评测（任务层约束 + 架构层不变量 + 答案质量层）；
- eval.rag：modular RAG 评测（语料 + 评测集 + 运行器 + 语义层）。

互不依赖，供各自评测脚本与回归门禁共用。
"""
