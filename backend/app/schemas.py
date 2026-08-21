"""API 请求/响应模型。"""
from pydantic import BaseModel, Field

MODE_IDS = ["react", "plan_execute", "reflection", "multi_agent"]
STRATEGY_IDS = ["standard", "few_shot", "cot"]
POLICY_IDS = ["always", "never"]


class StreamRequest(BaseModel):
    session_id: str = ""
    message: str = Field(min_length=1, description="用户输入/任务")
    mode: str = Field(default="react", description="推理模式")
    enabled_capabilities: list[str] = Field(default_factory=list, description="启用的能力 id 列表")
    prompt_strategy: str = Field(default="standard", description="提示词策略")
    approval_policy: str = Field(default="always", description="HITL 审批策略")


class ApproveRequest(BaseModel):
    approval_id: str = Field(description="审批编号")
    decision: str = Field(default="approve", description="approve | reject | modify")
    modified_args: dict | None = Field(default=None, description="decision=modify 时提供的新参数")


class StopRequest(BaseModel):
    session_id: str = Field(description="会话 id")


class SourceRequest(BaseModel):
    module: str = Field(description="源码模块 key")
