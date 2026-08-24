"""大模型工厂：基于 LLMService 按业务场景构建 ChatModel，保留兼容入口。

- llm_service：全局 LLMService 单例（默认场景配置见 service.DEFAULT_PROFILES）；
- create_chat_model(fake, scenario)：按场景取 ChatModel（含日志/错误包装）；
- create_embeddings(fake)：Embedding 模型（RAG / 长期记忆，OpenAI 兼容接口）。
"""
from app.config import settings
from app.core.errors import ConfigError
from app.llm.dashscope_embeddings import DashScopeEmbeddings
from app.llm.fake_model import FakeEmbeddings
from app.llm.service import DEFAULT_PROFILES, LLMService

# 全局 LLM 服务单例：构建时仅注册供应商与场景配置，不联网；实例在首次 get() 时惰性创建
llm_service = LLMService(profiles=DEFAULT_PROFILES)


def create_chat_model(fake: bool = False, scenario: str = "chat"):
    """按场景构建 ChatModel；fake=True 返回 Fake 场景（测试/离线用）。

    未配 LLM_API_KEY 时抛 ConfigError（由调用方决定是否回退规则实现）。
    """
    return llm_service.get("fake" if fake else scenario)


def create_embeddings(fake: bool = False):
    """构建 Embedding 模型；fake=True 返回 FakeEmbeddings。"""
    if fake:
        return FakeEmbeddings()
    if not settings.embedding_api_key:
        raise ConfigError("未配置 EMBEDDING_API_KEY，RAG 与长期记忆能力不可用")
    return DashScopeEmbeddings(
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        sparse_model=settings.sparse_embedding_model,
    )
