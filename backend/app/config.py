"""全局配置：从环境变量 / .env 读取，集中管理。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 阿里云百炼（DashScope 原生 SDK）大模型
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    chat_model: str = "qwen3.5-flash"
    # 开启思考：返回 reasoning_content（思考过程）与 content（最终输出）两类结果
    enable_thinking: bool = True

    # Embedding（RAG / 长期记忆能力，OpenAI 兼容接口，默认与百炼同一 Key）
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v3"

    # 检索后端选型：qdrant（默认）| elasticsearch | memory（强制内存，离线/测试）
    rag_store_backend: str = "qdrant"
    # Qdrant Cloud（多 RAG 方案向量库，可选；不配置则回退内存检索）
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_prefix: str = "knowledge"  # 集合名 = {prefix}_{scheme_id}，如 knowledge_naive
    qdrant_embedding_dim: int = 1024  # text-embedding-v3 默认维度
    # Elasticsearch（rag_store_backend=elasticsearch 时生效；ES dense_vector kNN 相似度检索）
    es_url: str = ""
    es_api_key: str = ""
    es_username: str = ""
    es_password: str = ""
    es_index_prefix: str = "knowledge"  # 索引名 = {prefix}_{scheme_id}，如 knowledge_advanced
    es_embedding_dim: int = 1024  # dense_vector 维度，与 embedding 模型一致
    # 已注册的 RAG 方案（每个方案一个独立 Qdrant 集合；graph/agentic 后续扩展）
    rag_schemes: list[str] = ["naive", "advanced", "modular", "agentic"]
    rag_default_scheme: str = "naive"
    # 知识库检索（RAG）总开关：默认开启（项目约定：能力后端默认就绪），
    # 每轮是否真正检索由请求/前端开关（rag_enabled）控制；设为 false 可整体关闭
    rag_enabled: bool = True
    # Advanced 方案：Query 重写生成的查询变体数（LLM Multi-Query）与重排模型
    rag_rewrite_variants: int = 3
    rag_rerank_model: str = "qwen3-rerank"
    # Modular 方案多跳迭代检索：最大检索跳数（LLM 路径默认 3，规则兜底上限 2）
    rag_max_hops: int = 3
    # Agentic 方案（多 Agent 编排）预算治理：步数/纠错轮数/超时/token/单工具上限/并行度
    rag_agent_max_steps: int = 8  # 全轮工具调用总上限（含被护栏拦截的调用），超限停止检索
    rag_agent_correction_rounds: int = 2  # CRAG 纠错回环上限，超出仍不足则如实上报缺口
    rag_agent_timeout_s: float = 90.0  # 单查询墙钟超时（秒），超时后不再发起新决策与工具波次
    rag_agent_token_budget: int = 0  # 角色 LLM 累计 token 预算（0=不限），达阈值后续角色规则回退；暂放开便于排查多轮遗忘
    rag_agent_tool_call_cap: int = 3  # 单工具整轮调用上限（multi_hop 固定更紧）
    rag_agent_parallel: int = 4  # 一波内并行工具调用数（首发按事实清单并行）

    # MCP Servers（JSON 字符串，声明可用 server；stdio 子进程由服务启动时自动拉起）
    mcp_servers: str = "{}"
    # 是否默认启用 MCP 能力；默认 false（关闭），服务启动仍会建立连接（discover），
    # 但能力不进目录，需在页面开启后才进入能力选配
    mcp_enabled: bool = False

    # 每日对话配额：限制「一台电脑 / 一个 IP」每天的对话次数（部署防滥用）
    quota_enabled: bool = True
    quota_daily_limit: int = 100  # 每客户端每天最多发起的对话次数
    quota_store_path: str = "./data/quota.json"  # 计数持久化文件（空字符串表示仅内存）

    # 运行参数
    # 日志级别：应用统一日志输出的最低级别（默认 INFO，可设 DEBUG/WARNING/ERROR 降噪）
    log_level: str = "INFO"
    max_iterations: int = 8
    max_steps: int = 5  # 各 Agent 循环的轮数上限（模型思考/工具回合数），超限强制结束防死循环
    rag_top_k: int = 3
    context_threshold: int = 12
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8000"]

    # 上下文管理与压缩（context_manage.py 四层管线，详见实施方案）
    context_mgmt_enabled: bool = True  # 总开关（关闭则整条管线与落盘全部不生效）
    # 层1 大文件落盘（预算裁剪 Budget）：超大单条工具输出 → 写盘 + 上下文只留指针
    context_offload_enabled: bool = True
    context_offload_threshold: int = 3000  # 单条工具结果超过该字符数 → 落盘
    context_offload_dir: str = "./data/offload"  # 相对 backend 根解析，自动创建
    context_offload_preview: int = 200  # 指针文本里的开头预览长度
    context_offload_max_per_session: int = 50  # 每会话落盘文件上限，超限删最旧
    # 层2 snip-compact（对话修剪）：历史过长时掐头去尾，裁掉中间旧消息（只管条数，不截工具内容）
    context_snip_enabled: bool = True
    context_snip_max_messages: int = 50  # 触发阈值：消息总数
    context_snip_keep_head: int = 3  # 保留开头条数（系统指令+初始目标）
    context_snip_keep_tail: int = 47  # 保留结尾条数（与当前任务相关）
    # 层3 micro-compact（旧工具结果体积）：更早的超长工具结果截断到头部，保留最近几条原文
    context_micro_enabled: bool = True
    context_micro_keep_recent: int = 6  # 保留最近几条工具结果原文，更早的超长结果截断
    context_micro_truncate_chars: int = 300  # 旧工具结果截断到该字符数（保留头部关键信息）
    # 层4 auto-compact（LLM 摘要，Stage 2，默认关）：万不得已时 LLM 全局摘要
    context_auto_compact_enabled: bool = False
    context_auto_compact_threshold: int = 100  # 摘要触发阈值
    context_auto_compact_keep_recent: int = 20  # 摘要后保留的最近消息条数

    # 长期记忆（memory 能力，独立于 RAG：跨会话事实/偏好，走工具按需召回）
    memory_enabled: bool = True  # 总开关（关闭则记忆工具/常驻注入/自动巩固全部不生效）
    memory_dir: str = "./data/memory"  # 记忆库落盘目录（每会话一个 {session_id}.jsonl + 全局 _global.jsonl）
    memory_top_k: int = 3  # 召回条数上限（注入预算）
    memory_threshold: float = 0.3  # 召回相似度阈值（不达标不注入，命中率低则零注入）
    memory_dedup_threshold: float = 0.92  # 语义去重阈值（相似则更新而非追加）
    memory_max_per_namespace: int = 500  # 每命名空间记录上限，超限按最近访问 LRU 淘汰
    memory_ttl_days: int = 90  # 记忆 TTL（天），默认 90 天启用过期清理；0 表示不启用
    memory_consolidate_enabled: bool = True  # 轮末自动提取巩固开关
    memory_consolidate_min_importance: float = 0.5  # 巩固提取的重要性下限，低于则丢弃
    memory_constant_enabled: bool = True  # 常驻记忆注入 system（会话启动默认开启）
    memory_constant_min_importance: float = 0.7  # 常驻注入的重要性下限
    memory_constant_top_k: int = 5  # 常驻注入条数上限
    memory_old_days_hint: int = 2  # 召回命中超过该天数的记忆附「可能过时」老化提示
    memory_proactive_enabled: bool = True  # L2 主动语义召回（每轮前置把当前对话转 query 召回相关记忆注入 user）
    memory_proactive_selector: bool = True  # 触发判断：轻量 LLM 先判「本轮是否需要记忆背景」，否则跳过召回
    memory_proactive_threshold: float = 0.3  # 主动召回相似度阈值（不达标不注入）
    memory_proactive_top_k: int = 3  # 主动召回每轮注入条数上限
    memory_proactive_max_chars: int = 400  # 主动召回注入字符预算（超预算截断）

    # 运行记录（telemetry）可观测性：每轮对话的 SSE 事件流 + LLM 调用明细落盘，
    # 供前端「运行记录」面板查看与回放（企业级 Trace 最小闭环）。
    telemetry_enabled: bool = True  # 总开关（关闭则不落盘、API 返回空）
    telemetry_dir: str = "./data/telemetry"  # 运行记录根目录（相对 backend 根解析）
    telemetry_ttl_days: int = 7  # 运行记录 TTL（天），超期清理
    telemetry_max_runs: int = 500  # 全库最大运行记录数，超限删最旧
    # LLM 成本估算单价（元 / 百万 token）：仅用于运行记录成本展示，可按实际价格调整；
    # 未配置时按 0 计（展示 token 数但成本为 0）
    llm_price_input_per_1m: float = 0.3
    llm_price_output_per_1m: float = 0.6

    # 护栏：工具调用上限与熔断
    tool_max_calls: int =10 # 单轮最多工具调用次数，达到后拒绝后续调用
    circuit_fail_threshold: int = 3  # 同一会话内“同一工具+相同参数”连续失败次数，达到即熔断该参数调用（换参重试放行）
    circuit_cooldown: int = 30  # 熔断冷却秒数，冷却结束放行一次探测（half-open）

    # 重试：工具层直接重试（透明重试）+ Agent 层思考后重试上限
    tool_retry_max: int = 3  # 工具层：瞬时错误（超时/连接重置/5xx/429）用相同参数直接重试的最大次数
    tool_retry_base_delay: float = 1.5  # 重试指数退避基础秒数（每次 ×2，上限封顶）；默认放大便于前端观察退避过程
    tool_retry_max_delay: float = 8.0  # 单次重试最长退避秒数
    agent_retry_max: int = 3  # Agent 层：同一工具（可换参数）连续失败达到该次数后，提示模型改用其它工具

    # 命令执行沙箱（run_command 工具）
    sandbox_backend: str = "opensandbox"  # opensandbox（默认，Docker 部署）| local（本机轻量沙箱兜底）
    sandbox_timeout: int = 10  # 单条命令最大执行秒数，超时硬杀进程树
    sandbox_max_output: int = 4000  # 命令输出最大字符数，超出截断
    # 沙箱/宿主机共享工作目录：沙箱写入该目录的文件会持久化到宿主机，
    # 后端通过 /api/sandbox/files 提供列表与下载（前端可点击下载）。
    # opensandbox 后端以 Volume 挂载进沙箱（需服务端 allowed_host_paths 放行该路径）；
    # local 后端直接以其作为命令工作目录。
    sandbox_work_dir: str = "./data/sandbox-work"
    sandbox_mount_target: str = "/work"  # 工作目录在沙箱容器内的挂载点
    # OpenSandbox 连接配置（sandbox_backend=opensandbox 时生效；服务端由用户自行 Docker 部署）
    opensandbox_domain: str = "localhost:8090"
    opensandbox_protocol: str = "http"
    opensandbox_api_key: str = ""
    opensandbox_image: str = "opensandbox/code-interpreter:latest"

    # 安全防护（security.md：输入/输出 Guardrail、来源可信分级、敏感数据脱敏）
    security_enabled: bool = True        # 安全防护总开关
    guard_input: bool = True             # 输入 Guardrail：越狱/提示注入特征过滤
    guard_output: bool = True            # 输出 Guardrail：敏感数据泄露阻断提示
    mask_sensitive_output: bool = True   # 输出敏感数据脱敏（手机号/身份证/银行卡/密钥）
    mark_untrusted: bool = True          # 不可信外部来源标记（Prompt 注入防御）


settings = Settings()
