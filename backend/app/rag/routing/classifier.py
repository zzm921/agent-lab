"""查询语义路由（Query Router）：多维路由决策，驱动模块化检索编排。

对齐 Modular RAG 企业级架构的「前置语义分类（Query Router）」：
不再输出单一标签，而是输出一份**结构化路由决策（RouteDecision）**，包含
- D1 retrieval_need：要不要检索（不检索则短路直接生成）；
- D3 retrieval_mode：检索策略（vector / hybrid / multi_recall）；
- D4 complexity：查询复杂度（simple / rewrite / decompose / multihop）；
- D5 generation_mode：生成模式（direct / citation / comparison）；
- confidence：置信度（全链路可观测）；
- reason：判定理由（全链路可观测、可回溯）。

路由为**纯 LLM**：按命名场景 rag_classify 懒取聊天模型（模型/参数见 service.DEFAULT_PROFILES），
一次调用输出 JSON 路由决策（枚举白名单校验）。单跳/多跳这类语义判定交给模型判断，
**不做规则正则兜底**——未配置 LLM（无 Key）或调用失败时直接抛错，避免规则误判
（如把「张三的领导是谁」这类单跳事实题误判为多跳）。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.errors import ConfigError
from app.llm.client import get_chat_model

# complexity（D4）
SIMPLE = "simple"
REWRITE = "rewrite"
DECOMPOSE = "decompose"
MULTIHOP = "multihop"

# retrieval_mode（D3）
VECTOR = "vector"
HYBRID = "hybrid"
MULTI_RECALL = "multi_recall"

# generation_mode（D5）
DIRECT = "direct"
CITATION = "citation"
COMPARISON = "comparison"

# target（D6）：查询目标语料类型，驱动检索层的定向补召回（卷名映射见 modular._TARGET_VOLUME_FILTERS）
TARGET_NONE = "none"
TARGET_PROFILE = "profile"    # 个人档案（员工权益明细）
TARGET_FAQ = "faq"            # 常见问答
TARGET_CASE = "case"          # 案例判例
TARGET_SCENE = "scene"        # 业务场景
TARGET_SOP = "sop"            # 标准作业流程
TARGET_VERSION = "version"    # 版本演进对比
TARGET_DUTY = "duty"          # 岗位职责

_VALID_TARGETS = {
    TARGET_NONE, TARGET_PROFILE, TARGET_FAQ, TARGET_CASE,
    TARGET_SCENE, TARGET_SOP, TARGET_VERSION, TARGET_DUTY,
}

_VALID_MODES = {VECTOR, HYBRID, MULTI_RECALL}
_VALID_COMPLEXITY = {SIMPLE, REWRITE, DECOMPOSE, MULTIHOP}
_VALID_GEN = {DIRECT, CITATION, COMPARISON}


@dataclass
class RouteDecision:
    """一次查询的路由决策：五个维度 + 置信度 + 理由。"""

    retrieval_need: bool = True      # D1：要不要检索
    retrieval_mode: str = VECTOR     # D3：检索策略
    complexity: str = SIMPLE         # D4：查询复杂度
    generation_mode: str = CITATION  # D5：生成模式
    target: str = TARGET_NONE        # D6：目标语料类型（定向补召回）
    confidence: float = 0.5
    reason: str = ""


class QueryClassifier(ABC):
    """查询语义路由抽象：输入问题，输出结构化路由决策。"""

    @abstractmethod
    def classify(self, query: str) -> RouteDecision:
        """返回路由决策（五维度 + 置信度 + 理由）。"""


class LLMQueryClassifier(QueryClassifier):
    """LLM 路由：按场景懒取模型，一次调用输出 JSON 路由决策（枚举白名单校验）。

    提示词遵循企业模板要点：角色隔离（你是路由引擎不是问答助手）、枚举封闭（禁止自造）、
    结构化 JSON 输出 + 置信度 + 理由、few-shot 对齐高频场景（简单事实/单跳关系/多实体对比/多跳）。
    未配置聊天模型或调用失败时抛 ConfigError——路由为纯 LLM，无规则兜底。
    """

    # 本阶段使用的模型/参数场景（qwen3.5-flash / temp=0.2 / max_tokens=500 / thinking=False，见 service.DEFAULT_PROFILES）
    scenario = "rag_classify"

    def classify(self, query: str) -> RouteDecision:
        llm = get_chat_model(self.scenario)
        if llm is None:
            raise ConfigError(
                f"modular 语义路由需要聊天模型（场景 {self.scenario}）：请配置 LLM_API_KEY。"
                "路由已改为纯 LLM，不再回退规则判定。"
            )
        messages = [
            SystemMessage(
                content=(
                    "你是 RAG 系统的语义路由引擎（Query Router）。你的唯一职责是分析用户输入，"
                    "输出一条检索路由决策 JSON。你不是问答助手，不回答用户问题。\n\n"
                    "本系统面向企业考勤、差旅、报销、人事制度问答，知识库为此领域语料。\n\n"
                    "字段语义（枚举值必须从下列值中选择，禁止自造）：\n"
                    "- retrieval_need（bool）：是否检索。寒暄/常识/通用推理 → false；事实性/领域性/需引用来源 → true。\n"
                    "- retrieval_mode（str）：vector=纯语义（口语化/概念性）；hybrid=混合检索（含领域词/编号/数字时优先）；"
                    "multi_recall=多路召回（多实体/对比/多条件）。\n"
                    "- complexity（str）：simple=单跳直接检索；rewrite=含指代/省略，需先改写；"
                    "decompose=对比/多实体/多条件，需拆分子查询；"
                    "multihop=多跳/流程/原因链/实体链（查某人领导的属性，如「张三的领导有几天年假」；"
                    "仅问「领导是谁」不算，属 simple 单跳），需迭代检索多轮。\n"
                    "- generation_mode（str）：direct=直接回答不引用来源；citation=引用来源回答（默认）；comparison=结构化对比输出。\n"
                    "- target（str）：查询最可能命中的语料类型（用于定向补召回，不影响检索策略）：\n"
                    "  profile=查某个具体员工的信息（含人名的权益/考勤/报销/档案问题）；\n"
                    "  faq=常见问答（口语化高频问题）；case=案例/判例（含「案例」「例如」「发生过」）；\n"
                    "  scene=业务场景处理（含「场景」「遇到…怎么办」）；sop=流程/表单/作业指引；\n"
                    "  version=版本对比/新旧制度差异（含「2025版」「2026版」「以前」「变更」）；\n"
                    "  duty=岗位职责/谁负责；none=一般制度条款/政策问题（默认）。\n"
                    "- confidence（float 0~1）：本次判断的把握程度。\n"
                    "- reason（str）：一句话说明分类依据。\n\n"
                    "常见组合（尽量遵循，与后续编排一致）：\n"
                    "- 寒暄/常识 → retrieval_need=false + vector/simple/direct；\n"
                    "- 单点事实 → simple + vector 或 hybrid + citation；\n"
                    "- 只问「X的{关系}是谁/叫什么」（如「张三的领导是谁」）→ simple 单跳直接检索（关系实体本身不是多跳）；\n"
                    "- 含指代/省略 → rewrite + hybrid 或 multi_recall + citation；\n"
                    "- 多跳/流程 → multihop + multi_recall + citation；\n"
                    "- 对比/多实体 → decompose + multi_recall + comparison。\n\n"
                    "输出必须严格是以下 JSON（不要输出任何其他文字）：\n"
                    '{"retrieval_need": true/false, "retrieval_mode": "vector|hybrid|multi_recall", '
                    '"complexity": "simple|rewrite|decompose|multihop", '
                    '"generation_mode": "direct|citation|comparison", '
                    '"target": "profile|faq|case|scene|sop|version|duty|none", '
                    '"confidence": 0.0~1.0, "reason": "一句话"}\n\n'
                    "当 retrieval_need=false 时，其余字段统一填 vector/simple/direct/none。\n\n"
                    "示例：\n"
                    '输入: 发票什么时候交\n'
                    '输出: {"retrieval_need": true, "retrieval_mode": "hybrid", '
                    '"complexity": "simple", "generation_mode": "citation", '
                    '"target": "none", "confidence": 0.97, "reason": "单点事实含领域词，混合检索"}\n'
                    '输入: 张三去上海出差打车费能报销吗\n'
                    '输出: {"retrieval_need": true, "retrieval_mode": "multi_recall", '
                    '"complexity": "multihop", "generation_mode": "citation", '
                    '"target": "profile", "confidence": 0.9, "reason": "查张三的部门档案再查差旅条款，多跳且定向档案卷"}\n'
                    '输入: 张三的领导是谁\n'
                    '输出: {"retrieval_need": true, "retrieval_mode": "vector", '
                    '"complexity": "simple", "generation_mode": "citation", '
                    '"target": "profile", "confidence": 0.95, "reason": "只问关系实体本身，单跳直接检索"}\n'
                    '输入: 出差和报销有什么区别\n'
                    '输出: {"retrieval_need": true, "retrieval_mode": "multi_recall", '
                    '"complexity": "decompose", "generation_mode": "comparison", '
                    '"target": "none", "confidence": 0.9, "reason": "多实体对比，需拆分子查询"}\n'
                    '输入: 那补卡流程呢（依赖上文）\n'
                    '输出: {"retrieval_need": true, "retrieval_mode": "hybrid", '
                    '"complexity": "rewrite", "generation_mode": "citation", '
                    '"target": "faq", "confidence": 0.88, "reason": "含指代且为流程问题，需改写后检索"}\n'
                    '输入: 报销发票什么时候交\n'
                    '输出: {"retrieval_need": true, "retrieval_mode": "multi_recall", '
                    '"complexity": "multihop", "generation_mode": "citation", '
                    '"target": "none", "confidence": 0.86, "reason": "多跳流程，需迭代检索"}\n'
                    '输入: 张三的领导有几天年假\n'
                    '输出: {"retrieval_need": true, "retrieval_mode": "multi_recall", '
                    '"complexity": "multihop", "generation_mode": "citation", '
                    '"target": "profile", "confidence": 0.85, "reason": "实体链多跳，先定位领导再查年假"}\n'
                    '输入: 你好，你是谁\n'
                    '输出: {"retrieval_need": false, "retrieval_mode": "vector", '
                    '"complexity": "simple", "generation_mode": "direct", '
                    '"target": "none", "confidence": 0.99, "reason": "闲聊元问题，无需检索"}'
                )
            ),
            HumanMessage(content=query),
        ]
        try:
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            return self._parse(content)
        except Exception as exc:  # noqa: BLE001 — 纯 LLM：失败直接报错，不静默回退
            raise ConfigError(f"语义路由（{self.scenario}）调用失败：{exc}") from exc

    @staticmethod
    def _parse(content: str) -> RouteDecision:
        """提取 JSON 片段并校验枚举白名单；非法/不可解析抛异常（由调用方统一报错）。"""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("LLM 路由输出中未找到 JSON")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("路由决策 JSON 必须是对象")
        mode = data.get("retrieval_mode")
        complexity = data.get("complexity")
        gen = data.get("generation_mode")
        target = data.get("target", TARGET_NONE)
        if mode not in _VALID_MODES or complexity not in _VALID_COMPLEXITY or gen not in _VALID_GEN:
            raise ValueError("路由决策枚举值非法")
        if target not in _VALID_TARGETS:
            target = TARGET_NONE  # 非法 target 降级为不过滤，不让路由失败
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return RouteDecision(
            retrieval_need=bool(data.get("retrieval_need", True)),
            retrieval_mode=mode,
            complexity=complexity,
            generation_mode=gen,
            target=target,
            confidence=max(0.0, min(1.0, confidence)),
            reason=str(data.get("reason", "") or ""),
        )


def build_classifier() -> QueryClassifier:
    """构造路由器：纯 LLM 路由（内部按场景懒取模型，无 Key/失败时报错）。"""
    return LLMQueryClassifier()
