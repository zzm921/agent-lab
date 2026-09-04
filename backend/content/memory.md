---
id: memory
name: 跨轮长期记忆
shortDesc: 跨会话记住关键事实并按语义召回，Agent 不再每次从零开始。
icon: history
difficulty: int
completeLevel: 70
tags: [Memory, Vector-Store, Persistence]
techFilters: [Qdrant]
accent: '#a855f7'
enabledTools: [memory]
prompts:
  - 记住：我喜欢深色主题，主色是紫色。之后再问我「我喜欢什么主题配色」。
  - 记住我的生日是 1995-08-20，然后问我「我的生日是哪天」并回答。
  - 新会话继续：你还记得上次让我记住的偏好吗？直接用起来，不用再问我。
---
## 为什么需要它

模型本身无状态，"记忆"来自外部存储。长期记忆把关键事实写入向量库，后续对话按语义召回，让 Agent 跨会话不"从零开始"。行业按认知科学把 Agent 记忆分为四类：

| 类型 | 含义 | 例子 |
|---|---|---|
| 工作记忆 | 当前上下文 | 本轮对话 |
| 情景记忆 | 具体过往事件 | "上周你说项目要重构" |
| 语义记忆 | 提炼的事实/偏好 | "喜欢深色主题" |
| 程序记忆 | 操作技能 | "测试用真实 DB" |

**关键概念：记忆 ≠ RAG。** RAG 是检索机制、数据是外部静态知识、无写入路径；记忆是概念/数据、来自交互历史、有「写入—巩固—更新—遗忘」完整生命周期。二者技术同构（向量检索 + 注入上下文），但记忆多出 RAG 不具备的写入链路。本项目刻意让记忆走工具通道（memory_recall）、RAG 走前置注入，两轨对照。

## 怎么解决

难点在写入与召回的取舍——什么值得记、何时忘、如何注入。企业级做法是**分级触发 + 注入位置规范**：

| 记忆类型 | 触发 | 谁触发 | 注入位置 |
|---|---|---|---|
| 常驻记忆（画像/偏好/约束） | 会话启动主动预载 | 系统 | system prompt |
| 语义记忆（事实/背景） | 前置自动检索（RAG 式）或工具按需 | 系统/模型 | user message |
| 情景记忆（具体事件） | 工具按需 | 模型 | user message（tool result） |

- **主动 vs 被动**：常驻预载是"主动"；按需召回是"被动"。被动又有两种实现——工具（模型按按钮，模型驱动）vs 前置注入（系统按按钮，系统驱动）。
- **成本**：检索本身毫秒级（本地向量索引，不调 LLM）；真正成本是"注入的记忆 token"。治理手段：相似度阈值（不达标不注入）、top-k 预算、已注入去重、条件触发——把每轮成本从"固定"压到"命中才花"。

## 核心实现

```python
# 写入：语义去重（太相似则更新而非追加）+ 重要性
def remember(text, kind, importance):
    vec = embed(text)
    dup = search(vec, top=1, threshold=0.92)
    if dup:  update(dup.id, vec, {text, kind, importance, ts})   # 纠偏
    else:    insert({id, text, vec, kind, importance, ts, ...})

# 加载：启动全量 load 索引（向量随记录持久化，不重算 embedding）
def load():  for rec in read_jsonl(path):  index.add(rec)

# 召回：阈值过滤 + 新鲜度重排
def recall(query, top_k, threshold):
    hits = index.search(query)
    hits = [h for h in hits if h.sim >= threshold]
    return sort(hits, key=sim * freshness(h))[:top_k]

# 注入：规范化记忆块 + 兜底句
def inject(hits):
    lines = [f"[{i}] ({kind}·重要度{imp}) {text} —— {date}" for i, h in enumerate(hits)]
    return "【长期记忆检索结果】\n" + "\n".join(lines) + \
           "\n请优先参考以上记忆回答；若与本次说明矛盾，以本次说明为准。"
```

召回注入的示例 prompt（memory_recall 返回给模型的文本）：

```
【长期记忆检索结果】
[1] (偏好·重要度0.9) 用户喜欢深色主题，主色是紫色 —— 记录于 2026-08-01
[2] (事实·重要度0.7) 用户的生日是 1995-08-20 —— 记录于 2026-07-15
请优先参考以上记忆回答；若与本次说明矛盾，以本次说明为准。
```

命中 2 天以上记忆时追加老化提示：`注意：以下记忆来自 N 天前，可能已过时。使用前请与当前实际情况核对，确认仍适用再采用。`

常驻记忆注入 system 的示例块：

```
## 用户记忆（来自历史会话，仅供参考）
以下记忆可能过时或不准确，请作为背景参考；若与用户本次说明冲突，以本次说明为准：
- [偏好] 喜欢深色主题，主色是紫色
- [画像] 后端工程师，熟悉 Go
```

自动提取（轮末巩固）的示例 prompt：

```
从以下对话中提取值得长期记住的事实。
只提取：用户画像、偏好、项目决策、外部资源。
不提取：可从代码/文件/命令历史推导的信息、临时状态。
每条输出 JSON：{text, type, importance(0~1)}；importance 低于 0.5 的不要输出。
```

## 本项目的做法

**已实现（落地完成，详见《记忆体系规划与落地实施方案》）**：

- **存储**：向量 + 元数据持久化到 JSONL（会话记忆 `data/memory/{session_id}.jsonl`；常驻记忆按客户端隔离 `_global_{设备/IP}.jsonl`），启动全量载入内存索引、不重算 embedding；记录带 kind / importance / 时间戳 / 来源 / TTL；每命名空间上限 LRU + TTL 遗忘治理；
- **写入**：memory_write 支持 kind（fact/preference/episodic/procedural）与 importance、scope（session/global）；语义去重（相似度 ≥0.92 更新而非追加）；轮末自动提取巩固后台静默运行，提取用独立轻量场景 memory_consolidate（关闭 thinking，实测 48s→3s），按 LLM 判定的 scope 自动分流——global（长期偏好/约束/稳定画像）写当前客户端常驻库跨会话生效、session（临时上下文）写会话库；importance 过滤、吞错不阻断主链路、不产事件不拖慢响应；
- **加载**：常驻记忆会话启动注入 system（默认开启，importance ≥0.7 的 top-k）；memory_recall 工具按需召回（被动加载·模型驱动），返回规范化注入块 + 老化提示（命中 2 天以上附带）；召回按 UNTRUSTED 外部来源注入隔离（防记忆投毒）；
- **管理**：GET/DELETE/POST 记忆管理 API + 前端「记忆管理」面板（查看 / 删除 / 手动写入）；SSE 透出 memory_write / memory_read / memory_constant 事件卡片。

## 收益与边界

- 四类记忆模型，跨会话语义召回，记住偏好与关键事实
- 短期 in-context + 长期向量组合，平衡成本与覆盖
- 记忆 ≠ RAG：独立链路、独立工具，不与 RAG 检索混用
- 边界：单机形态温/冷合一（内存索引 + JSONL），云端演进为对象存储 + 向量库 + 缓存分层；语义记忆「前置自动检索」（每轮系统驱动注入 user）与小模型选择器为演进项/分阶段落地
