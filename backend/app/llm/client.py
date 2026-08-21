"""大模型工厂：阿里云百炼（DashScope 原生 SDK）Chat + Embedding，测试时注入 Fake。"""
from app.config import settings
from app.core.errors import ConfigError
from app.llm.dashscope_chat import DashScopeChatModel
from app.llm.fake_model import FakeChatModel, FakeEmbeddings


def create_chat_model(fake: bool = False):
    """构建 DashScope ChatModel（官方 SDK）；fake=True 时返回 FakeChatModel。"""
    if fake:
        return FakeChatModel()
    if not settings.llm_api_key:
        raise ConfigError("未配置 LLM_API_KEY（阿里云百炼 DashScope API Key），请在 backend/.env 中设置后重启服务")
    return DashScopeChatModel(
        model_name=settings.chat_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.3,
        enable_thinking=settings.enable_thinking,
    )


def create_embeddings(fake: bool = False):
    """构建 Embedding 模型；fake=True 时返回 FakeEmbeddings。"""
    if fake:
        return FakeEmbeddings()
    if not settings.embedding_api_key:
        raise ConfigError("未配置 EMBEDDING_API_KEY，RAG 与长期记忆能力不可用")
    # return OpenAIEmbeddings(
    #     model=settings.embedding_model,
    #     api_key=settings.embedding_api_key,
    #     base_url=settings.embedding_base_url,
    # )
    return FakeEmbeddings()
