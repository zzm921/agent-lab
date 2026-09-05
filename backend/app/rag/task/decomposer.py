"""任务拆解器：复合问题 → 子查询节点 DAG（外层任务闭环的规划角色）。

LLM 决策（rag_task_decompose 场景，轻量 JSON）+ 确定性规则回退（单节点 = 原查询）。
LLM 只产出节点规格，不执行检索；解析失败/调用异常回退，保证任务不因单次模型故障中断。
节点产出统一封顶（max_nodes）并做依赖校验，防止 DAG 死锁/环。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model

logger = logging.getLogger(__name__)

SCENARIO_DECOMPOSE = "rag_task_decompose"

# 并列/选择/对比复合标记（规则粗筛用）：命中任一 → 疑似复合，交给 LLM 拆解
_COMPOUND_MARKERS = ("和", "与", "以及", "还有", "同时", "分别", "对比", "区别", "差异", "、", "；", ";", "?", "？")


def _extract_json(content: str) -> dict[str, Any] | None:
    """从 LLM 输出提取首个 JSON 对象；不可解析返回 None（调用方回退规则）。"""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _tokens_of(resp: Any) -> dict[str, int]:
    """从响应提取 token 用量（LangChain usage_metadata 优先，DashScope 原生次之）。"""
    usage = getattr(resp, "usage_metadata", None) or {}
    prompt = usage.get("input_tokens") or 0
    completion = usage.get("output_tokens") or 0
    if not prompt and not completion:
        native = (getattr(resp, "response_metadata", None) or {}).get("token_usage") or {}
        prompt = native.get("prompt_tokens") or 0
        completion = native.get("completion_tokens") or 0
    return {"prompt": int(prompt or 0), "completion": int(completion or 0)}


def task_llm_json(
    scenario: str,
    system: str,
    user: str,
    parse: Callable[[dict[str, Any]], Any],
    ledger: dict[str, int] | None = None,
) -> Any:
    """任务层 LLM 决策：调模型 → 任务账本记账 → 解析 JSON。

    返回 None 表示未调 LLM（无 Key）或解析失败——调用方回退规则；异常同样回退。
    ledger（{"prompt","completion"} 就地累加）供任务账本 token 预算使用。
    """
    llm = get_chat_model(scenario)
    if llm is None:
        return None
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as exc:  # noqa: BLE001 — 任务层决策失败回退规则，不中断任务
        logger.warning("[task:%s] LLM 调用失败: %s", scenario, exc)
        return None
    if ledger is not None:
        usage = _tokens_of(resp)
        ledger["prompt"] += usage["prompt"]
        ledger["completion"] += usage["completion"]
    content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
    data = _extract_json(content)
    if data is None:
        return None
    return parse(data)


def _normalize_nodes(raw: list[dict[str, Any]], max_nodes: int) -> list[dict[str, Any]]:
    """节点校验与归一：重编号 n1..nN、去空查询、依赖只保留已存在 id、封顶 max_nodes。"""
    seen_queries: set[str] = set()
    nodes: list[dict[str, Any]] = []
    for item in raw[:max_nodes]:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query") or "").strip()
        if not q or q in seen_queries:
            continue
        seen_queries.add(q)
        nodes.append({
            "id": f"n{len(nodes) + 1}",
            "query": q,
            "deps": [d for d in (item.get("deps") or []) if isinstance(d, str) and d],
            "reason": str(item.get("reason") or ""),
        })
    return nodes


class TaskDecomposer:
    """任务拆解器：复合问题 → [{id, query, deps, reason}]（封顶 max_nodes）。"""

    def __init__(self, max_nodes: int = 4):
        self.max_nodes = max_nodes

    @staticmethod
    def _system() -> str:
        return (
            "你是检索任务拆解器。把用户问题拆解为若干子查询节点（每个节点一个独立可检索的查询，"
            "节点间可声明依赖：先查的节点是后查节点的依赖）。规则："
            "1) 简单问题只输出 1 个节点；只有真正需要拆解的复合问题才拆成多个；"
            "2) 每个节点 query 是可直接检索的具体查询，不要带『查一下/帮我』等套话；"
            "3) 若节点 B 需要节点 A 的检索结果才能检索（链式依赖），B 的 deps 填 A 的 id；"
            "4) 节点数不超过 4；"
            "5) 若多个子查询互不依赖、检索一次可并行覆盖（如并列多事实），只输出 1 个节点承载完整问题，"
            "不要拆开——内层检索会自动多路并行。\n"
            '输出严格 JSON：{"nodes": [{"id": "n1", "query": "...", "deps": [], "reason": "简短理由"}], '
            '"reason": "整体拆解理由"}'
        )

    @staticmethod
    def _user(query: str) -> str:
        return f"用户问题：{query}"

    @staticmethod
    def _looks_simple(query: str) -> bool:
        """规则粗筛：短查询（≤8 字）且无并列/选择/对比标记 → 判定简单问题。

        简单问题跳过拆解 LLM（零额外成本，整问题单节点直通内层）；只要疑似复合
        （长查询或含标记）就交给 LLM 拆解——宁可多一次调用，不漏链式依赖。
        """
        q = (query or "").strip()
        if len(q) > 8:
            return False
        return not any(m in q for m in _COMPOUND_MARKERS)

    @staticmethod
    def _simple_single(query: str) -> tuple[list[dict[str, Any]], str, str]:
        """规则判定简单：不调拆解 LLM，单节点 = 原查询直通内层。"""
        return (
            [{"id": "n1", "query": query, "deps": [], "reason": "规则判定简单问题：单节点直通"}],
            "规则判定简单问题：不调拆解 LLM，整问题单节点",
            "规则判定简单问题",
        )

    def _fallback(self, query: str) -> tuple[list[dict[str, Any]], str, str]:
        """规则回退：单节点 = 原查询（不拆解，保证任务可继续）。"""
        return (
            [{"id": "n1", "query": query, "deps": [], "reason": "规则回退：单节点"}],
            "规则回退：不拆解，整问题单节点",
            "拆解规则回退",
        )

    def decompose(self, query: str, ledger: dict[str, int] | None = None) -> tuple[list[dict[str, Any]], str, str]:
        """复合问题 → 节点 DAG。返回 (nodes, thought, note)；nodes 已封顶/校验，非空。

        单一入口（合并后）：简单问题由规则粗筛直接单节点直通（不调拆解 LLM），
        疑似复合才走 LLM 拆解，失败回退单节点。
        """
        if self._looks_simple(query):
            return self._simple_single(query)

        def parse(data: dict[str, Any]) -> tuple[list[dict[str, Any]], str] | None:
            raw = data.get("nodes")
            if not isinstance(raw, list) or not raw:
                return None
            nodes = _normalize_nodes(raw, self.max_nodes)
            if not nodes:
                return None
            return nodes, str(data.get("reason") or "")

        outcome = task_llm_json(SCENARIO_DECOMPOSE, self._system(), self._user(query), parse, ledger=ledger)
        if outcome is None:
            return self._fallback(query)
        nodes, thought = outcome
        logger.info("[task] 拆解 %d 节点（%s）", len(nodes), thought)
        return nodes, thought, ""
