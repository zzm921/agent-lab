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
    # 稀疏向量模型（为后续 advanced 混合检索预留；账号未开通时回退本地 n-gram）
    sparse_embedding_model: str = "text-sparse-embedding-v1"

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
    # 已注册的 RAG 方案（每个方案一个独立 Qdrant 集合；modular/graph/agentic 后续扩展）
    rag_schemes: list[str] = ["naive", "advanced"]
    rag_default_scheme: str = "naive"
    # 知识库检索（RAG）总开关：默认开启（项目约定：能力后端默认就绪），
    # 每轮是否真正检索由请求/前端开关（rag_enabled）控制；设为 false 可整体关闭
    rag_enabled: bool = True
    # Advanced 方案：Query 重写生成的查询变体数（LLM Multi-Query）与重排模型
    rag_rewrite_variants: int = 3
    rag_rerank_model: str = "qwen3-rerank"

    # MCP Servers（JSON 字符串，声明可用 server；stdio 子进程由服务启动时自动拉起）
    mcp_servers: str = "{}"
    # 是否默认连接 MCP Server；默认 true，服务启动时自动发现（stdio 拉起）已配置的 server
    mcp_enabled: bool = True

    # 每日对话配额：限制「一台电脑 / 一个 IP」每天的对话次数（部署防滥用）
    quota_enabled: bool = True
    quota_daily_limit: int = 100  # 每客户端每天最多发起的对话次数
    quota_store_path: str = "./data/quota.json"  # 计数持久化文件（空字符串表示仅内存）

    # 运行参数
    max_iterations: int = 8
    max_steps: int = 5  # 各 Agent 循环的轮数上限（模型思考/工具回合数），超限强制结束防死循环
    rag_top_k: int = 3
    context_threshold: int = 12
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8000"]

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


settings = Settings()
