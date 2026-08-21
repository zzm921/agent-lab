"""内置能力定义：能力元信息 + 可用性判断（基于是否具备 Embedding 能力）。"""

BUILTIN_CAPABILITIES = [
    {
        "id": "calculator",
        "name": "计算器",
        "source": "builtin",
        "requires": None,
        "desc": "安全计算数学表达式，支持四则运算、百分号换算与括号。",
        "example": "帮我计算 (137×0.85−20)÷3 等于多少",
        "code_key": "calculator",
    },
    {
        "id": "time_now",
        "name": "当前时间",
        "source": "builtin",
        "requires": None,
        "desc": "返回当前日期与时间。",
        "example": "现在几点？今天是几号？",
        "code_key": "time_now",
    },
    {
        "id": "web_search",
        "name": "网页搜索",
        "source": "builtin",
        "requires": None,
        "desc": "在网页上搜索关键词并返回前几条结果的标题与摘要。",
        "example": "搜索一下 Qwen3 的发布时间",
        "code_key": "web_search",
    },
    {
        "id": "rag",
        "name": "知识库检索 (RAG)",
        "source": "builtin",
        "requires": "embedding",
        "desc": "在平台内置知识库中向量检索最相关片段，生成有依据的回答。",
        "example": "根据内置知识库回答：LangGraph 的 StateGraph 如何定义状态？",
        "code_key": "rag",
    },
    {
        "id": "memory",
        "name": "长期记忆",
        "source": "builtin",
        "requires": "embedding",
        "desc": "跨轮次记住关键事实，后续对话按语义召回。",
        "example": "记住我叫小明，正在做 AI Agent 项目；下一轮再问我叫什么",
        "code_key": "memory",
    },
]
