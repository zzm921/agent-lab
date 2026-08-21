"""reflection 模式：生成草稿 → 自我反思 → 迭代修订（create_agent 实现）。

ReflectionMiddleware 在单次模型调用内完成 草稿 → 批评 → 修订 → 再批评 循环，
无工具循环；输出 message / reflect / revise 事件与旧图一致。
"""
from langchain.agents import create_agent

from app.agents.middleware.reflection_mw import ReflectionMiddleware


def build_reflection_agent(llm, tools, emit, settings, checkpointer=None):
    """构建 reflection 代理：单次模型调用内完成生成-反思-修订循环。"""
    return create_agent(
        model=llm,
        tools=[],
        middleware=[ReflectionMiddleware(llm, emit, settings)],
        checkpointer=checkpointer,
    )
