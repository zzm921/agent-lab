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
