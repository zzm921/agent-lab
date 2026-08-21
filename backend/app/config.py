"""全局配置：从环境变量 / .env 读取，集中管理。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 阿里云百炼（DashScope 原生 SDK）大模型
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    chat_model: str = "qwen-plus"
    # 开启思考：返回 reasoning_content（思考过程）与 content（最终输出）两类结果
    enable_thinking: bool = True

    # Embedding（RAG / 长期记忆能力，OpenAI 兼容接口，默认与百炼同一 Key）
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v3"

    # MCP Servers（JSON 字符串）
    mcp_servers: str = "{}"

    # 运行参数
    max_iterations: int = 8
    rag_top_k: int = 3
    context_threshold: int = 12
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8000"]


settings = Settings()
