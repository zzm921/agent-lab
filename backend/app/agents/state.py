"""Agent 状态定义（LangGraph TypedDict 状态）。"""
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class PlanStep(TypedDict, total=False):
    id: str
    description: str
    status: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    mode: str
    plan: list[str]
    current_step: int
    iteration: int
    final: str | None
    critique: str | None
    route: dict | None
