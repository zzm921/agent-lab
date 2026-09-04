"""API 请求/响应模型。"""
from pydantic import BaseModel, Field

MODE_IDS = ["react", "plan_execute", "reflection", "multi_agent"]
STRATEGY_IDS = ["standard", "few_shot", "cot"]
POLICY_IDS = ["always", "never"]
RAG_SCHEME_IDS = ["naive", "advanced"]


class StreamRequest(BaseModel):
    session_id: str = ""
    message: str = Field(min_length=1, description="用户输入/任务")
    mode: str = Field(default="react", description="推理模式")
    enabled_capabilities: list[str] = Field(default_factory=list, description="启用的能力 id 列表")
    prompt_strategy: str = Field(default="standard", description="提示词策略")
    approval_policy: str = Field(default="always", description="HITL 审批策略")
    rag_scheme: str = Field(default="naive", description="RAG 方案 id（当前仅 naive）")
    rag_enabled: bool = Field(default=True, description="本轮是否启用知识库检索（RAG）前置检索；默认开启，前端开关可关闭")
    memory_enabled: bool = Field(default=True, description="本轮是否启用长期记忆能力（工具/常驻注入/轮末巩固）；默认开启")
    context_keep_rounds: int = Field(
        default=0,
        ge=0,
        le=50,
        description="「每轮压缩」演示：>0 时每轮都压缩并保留最近 N 轮对话原文（更早历史被裁剪/截断）；0 使用系统默认阈值",
    )


class ApproveRequest(BaseModel):
    approval_id: str = Field(description="审批编号")
    decision: str = Field(default="approve", description="approve | reject | modify")
    modified_args: dict | None = Field(default=None, description="decision=modify 时提供的新参数")


class StopRequest(BaseModel):
    session_id: str = Field(description="会话 id")


class FaultRequest(BaseModel):
    tool: str = Field(description="工具名")
    mode: str = Field(
        default="off",
        description="故障注入类型：瞬时错误（timeout/conn_reset/dns/http_429/http_5xx → 工具层直接重试）"
        "或参数/业务错误（error/business/http_400/http_401/http_403/http_404 → 返回给模型重试）；off 恢复正常",
    )


class SourceRequest(BaseModel):
    module: str = Field(description="源码模块 key")


class McpToggleRequest(BaseModel):
    enabled: bool = Field(description="是否开启 MCP 服务（连接注册的 MCP Server 并发现工具）")


class MemoryWriteRequest(BaseModel):
    text: str = Field(description="要记住的事实文本")
    kind: str = Field(default="fact", description="记忆分类：fact | preference | episodic | procedural")
    importance: float = Field(default=0.5, ge=0, le=1, description="重要度 0~1")
    scope: str = Field(default="session", description="写入范围：session（会话）| global（全局常驻）")
    session_id: str = Field(default="", description="scope=session 时的目标会话")
