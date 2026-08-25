"""多跳检索规划器（MultiHop Planner）：规划-执行-验证的「规划」阶段。

对齐 Modular RAG 企业级架构的「先规划后执行」：多跳问题（流程/原因链/实体链）
先一次性拆出子查询计划（每步含目标维度 target、可独立检索的 query、可预判实体 entity
与依赖 depends_on），再交由执行器按计划逐跳检索——相比"走一步看一步"的贪心迭代，
规划让全局视野不漏查、依赖显式化、并可预判哪些步骤可能被既有证据复用。

- LLMMultiHopPlanner：按命名场景 rag_plan 懒取聊天模型（模型/参数见 service.DEFAULT_PROFILES），
  一次输出结构化计划 JSON（steps + reason），无模型（未配 Key）/解析失败回退确定性规则；
- RuleMultiHopPlanner：无 LLM（离线/仅配 Embedding）时的确定性回退——
  实体链（X的{关系}...）拆两跳：先定位中间实体（X的{关系}是谁），再查其属性（原查询）；
  流程/原因型退化为单步计划（原查询 + 目标=流程），靠验证闸门驱动补缺。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model
from app.rag.retrieval.iterative_retrieval import HopPlan, PlanStep

# 实体链关系词：X的{关系} 的规则拆跳仅用于无 LLM 兜底（中间实体 = 领导的上级链）
_RELATIONS = "领导|上级|主管|经理|负责人|老板|下属|同事|秘书|助理"
_ENTITY_CHAIN = re.compile(rf"(?P<subject>.+?)的(?P<rel>{_RELATIONS})(?P<rest>.+)")


class MultiHopPlanner(ABC):
    """多跳规划器抽象：输入原问题，输出子查询计划（目标/依赖/可预判实体）。"""

    @abstractmethod
    def plan(self, query: str) -> HopPlan:
        """返回子查询计划：每步含 target / query / entity / depends_on。"""


class LLMMultiHopPlanner(MultiHopPlanner):
    """LLM 规划：按场景懒取模型，一次调用输出结构化计划 JSON；异常回退规则规划器。"""

    # 本阶段使用的模型/参数场景（qwen3.5-flash / temp=0.2 / max_tokens=500 / thinking=True，见 service.DEFAULT_PROFILES）
    scenario = "rag_plan"

    def __init__(self):
        self._fallback = RuleMultiHopPlanner()

    def plan(self, query: str) -> HopPlan:
        llm = get_chat_model(self.scenario)
        if llm is None:
            return self._fallback.plan(query)
        try:
            messages = [
                SystemMessage(
                    content=(
                        "你是多跳检索的「子查询规划器」。给定原始问题，一次性输出一份子查询执行计划：\n"
                        "1. 把多跳问题按依赖关系拆成若干子查询（如「张三的领导有几天年假」："
                        "先查「张三的领导是谁」、再查「王刚的年假有多少天」）；\n"
                        "2. 每步包含：\n"
                        "   - target：本步要解决的\"目标维度/事实\"（一句话，如\"领导是谁\"/\"年假天数\"）；\n"
                        "   - query：可独立检索的子查询；\n"
                        "   - entity：若本步可预判其关键实体/概念，填该实体（如第2步可预判为\"王刚\"）；"
                        "否则填 null；\n"
                        "   - depends_on：本步依赖的先前 target 列表（无则 []）；\n"
                        "3. 子查询之间尽量无重叠；若已是单跳事实题，输出 1 个步骤即可。\n"
                        "输出必须严格是以下 JSON（不要输出任何其他文字）：\n"
                        '{"steps": [{"target": "...", "query": "...", "entity": "..."|null, '
                        '"depends_on": ["..."]}], "reason": "一句话说明规划思路"}'
                    )
                ),
                HumanMessage(content=query),
            ]
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            return self._parse(query, content)
        except Exception:  # noqa: BLE001 — 模型抖动时回退规则规划
            return self._fallback.plan(query)

    @staticmethod
    def _parse(query: str, content: str) -> HopPlan:
        """提取 JSON 并校验字段；非法/不可解析抛异常交给调用方回退。"""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("LLM 规划输出中未找到 JSON")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("规划 JSON 必须是对象")
        steps = []
        for item in data.get("steps") or []:
            if not isinstance(item, dict):
                continue
            target = str(item.get("target") or "").strip()
            q = str(item.get("query") or "").strip()
            if not target or not q:
                continue
            entity = item.get("entity")
            steps.append(
                PlanStep(
                    target=target,
                    query=q,
                    entity=str(entity).strip() if entity else None,
                    depends_on=[str(d).strip() for d in (item.get("depends_on") or []) if str(d).strip()],
                )
            )
        if not steps:
            raise ValueError("规划 JSON 无有效步骤")
        return HopPlan(steps=steps, reason=str(data.get("reason") or "") or "LLM 规划")


class RuleMultiHopPlanner(MultiHopPlanner):
    """确定性规则规划（无 LLM 回退）。

    - 实体链（X的{关系}...）：拆两跳——先定位中间实体（X的{关系}是谁），
      再查其属性（原查询整体，交给检索/验证）；依赖显式化（第2步依赖第1步）；
    - 流程/原因链：单步计划（原查询 + 目标=流程），具体补缺交给验证闸门驱动。
    """

    def plan(self, query: str) -> HopPlan:
        match = _ENTITY_CHAIN.search(query)
        if match:
            subject = match.group("subject").strip()
            rel = match.group("rel").strip()
            rest = match.group("rest").strip()
            step1 = PlanStep(target=f"{subject}的{rel}是谁", query=f"{subject}的{rel}是谁")
            step2 = PlanStep(
                target=f"{rel}{rest}" if rest else f"{rel}的属性",
                query=query,
                depends_on=[step1.target],
            )
            return HopPlan(
                steps=[step1, step2],
                reason="实体链：先定位中间实体，再查其属性",
            )
        return HopPlan(
            steps=[PlanStep(target="流程", query=query)],
            reason="流程/原因链：先按原查询检索，再经验证闸门判断缺口",
        )


def build_planner() -> MultiHopPlanner:
    """构造规划器：有 LLM 场景配置用 LLM 规划（内部懒取），否则规则回退。"""
    return LLMMultiHopPlanner()
