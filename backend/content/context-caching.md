---
id: context-caching
name: 上下文缓存与渐进式披露
shortDesc: Prompt Caching 复用前缀 KV 缓存降本降延迟；渐进式披露 / JIT 检索让模型只按需加载上下文。
icon: zap
difficulty: int
tags: [Context-Engineering, Prompt-Caching, JIT, Progressive-Disclosure]
techFilters: [LangGraph]
accent: '#eab308'
experience: false
---
## 为什么需要它

上下文缓存（Prompt Caching）对前缀不变的请求复用 KV 缓存，把重复的 System Prompt、工具定义、长文档一次付费、多次复用，显著降低 Token 成本与首字延迟。渐进式披露（Progressive Disclosure）则反其道而行——不把全部指令一次性塞入，先给轻量索引，模型按需（JIT 动态检索）只拉取当前步骤真正需要的内容，保持高信噪比。

## 怎么解决

难点在于二者协同——缓存要求前缀稳定，动态检索却会改变上下文导致 cache miss。业界做法：静态指令（System Prompt、工具定义）固定在最前保证前缀命中，动态内容追加在后，压缩从尾部开始以保护缓存前缀。

## 核心实现

```python
# Prompt Caching：静态前缀缓存复用 + 动态内容后置
messages = [
    {"role": "system", "content": SYSTEM_PROMPT,
     "cache_control": {"type": "ephemeral"}},   # 静态：稳定前缀，命中缓存
    {"role": "system", "content": TOOL_DEFS},
    *dynamic_history,                            # 动态：追加在后，不破坏前缀
]

# 渐进式披露：先给索引，按需 JIT 加载
ctx = index_of(available_docs)
docs = await llm.ask("当前步骤需要哪些上下文?", ctx)
```

## 收益与边界

- 前缀 KV 缓存复用，重复上下文近乎零成本
- 渐进式披露：先索引后按需加载，上下文精炼
- 缓存前缀与动态检索分层设计，兼顾成本与质量
