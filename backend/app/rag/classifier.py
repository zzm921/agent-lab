"""查询语义路由（Query Router）：多维路由决策，驱动模块化检索编排。

对齐 Modular RAG 企业级架构的「前置语义分类（Query Router）」：
不再输出单一标签，而是输出一份**结构化路由决策（RouteDecision）**，包含
- D1 retrieval_need：要不要检索（不检索则短路直接生成）；
- D3 retrieval_mode：检索策略（vector / hybrid / multi_recall）；
- D4 complexity：查询复杂度（simple / rewrite / decompose）；
- D5 generation_mode：生成模式（direct / citation / comparison）；
- confidence：置信度（低置信度可走兜底路径）；
- reason：判定理由（全链路可观测、可回溯）。

- LLMQueryClassifier：按命名场景 rag_classify 懒取聊天模型（模型/参数见 service.DEFAULT_PROFILES），
  一次调用输出 JSON 路由决策（枚举白名单校验）；无模型（未配 Key）/异常回退规则路由；
- RuleQueryClassifier：无 LLM（离线/仅配 Embedding）时的确定性规则路由回退。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

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

_VALID_MODES = {VECTOR, HYBRID, MULTI_RECALL}
_VALID_COMPLEXITY = {SIMPLE, REWRITE, DECOMPOSE, MULTIHOP}
_VALID_GEN = {DIRECT, CITATION, COMPARISON}

# LLM 路由低置信度阈值：低于该值认为路由把握不足，回退确定性规则路由
_LOW_CONFIDENCE_THRESHOLD = 0.5

# 对比/多实体信号：需拆分子查询（decompose），结构化对比输出
_COMPARISON = re.compile(
    r"(对比|比较|区别|不同|分别|差异|哪几|哪两|两者|之间|有什么不同|哪个更)"
)

# 多跳/流程信号：需迭代检索（多轮召回拼出中间环节）
# 两类多跳：流程/原因链（为什么/流程/步骤/如何办理…）；实体链（先定位中间实体再查其属性，
# 如「张三的领导有几天年假」需先解析出「领导」这个中间实体再查年假）
_MULTI_HOP = re.compile(
    r"(为什么|原因|流程|步骤|先后|如何办理|怎么办|同时|以及|先.*后|导致|影响|"
    r"的(领导|上级|主管|经理|负责人|老板|下属|同事|秘书|助理))"
)

# 指代/歧义信号：依赖上下文，需改写
_DEICTIC = re.compile(r"(它|这个|那个|上述|该问题|这两种|那两种)")

# 寒暄/元问题：无需检索
_GREETING = re.compile(
    r"(你好|您好|谢谢|多谢|再见|拜拜|在吗|你是谁|你叫什么|介绍一下你自己|"
    r"你能做什么|你会什么|今天天气|吃了吗)"
)


@dataclass
class RouteDecision:
    """一次查询的路由决策：五个维度 + 置信度 + 理由。"""

    retrieval_need: bool = True      # D1：要不要检索
    retrieval_mode: str = VECTOR     # D3：检索策略
    complexity: str = SIMPLE         # D4：查询复杂度
    generation_mode: str = CITATION  # D5：生成模式
    confidence: float = 0.5
    reason: str = ""


class QueryClassifier(ABC):
    """查询语义路由抽象：输入问题，输出结构化路由决策。"""

    @abstractmethod
    def classify(self, query: str) -> RouteDecision:
        """返回路由决策（五维度 + 置信度 + 理由）。"""


class LLMQueryClassifier(QueryClassifier):
    """LLM 路由：按场景懒取模型，一次调用输出 JSON 路由决策（枚举白名单校验）；异常回退规则路由。

    提示词遵循企业模板要点：角色隔离（你是路由引擎不是问答助手）、枚举封闭（禁止自造）、
    结构化 JSON 输出 + 置信度 + 理由、few-shot 对齐高频场景（简单事实/多实体对比/闲聊）。
    """

    # 本阶段使用的模型/参数场景（qwen3.5-flash / temp=0.2 / max_tokens=500 / thinking=False，见 service.DEFAULT_PROFILES）
    scenario = "rag_classify"

    def __init__(self):
        self._fallback = RuleQueryClassifier()

    def classify(self, query: str) -> RouteDecision:
        llm = get_chat_model(self.scenario)
        if llm is None:
            return self._fallback.classify(query)
        try:
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
                        "multihop=多跳/流程/原因链/实体链（如查某人领导的属性），需迭代检索多轮。\n"
                        "- generation_mode（str）：direct=直接回答不引用来源；citation=引用来源回答（默认）；comparison=结构化对比输出。\n"
                        "- confidence（float 0~1）：本次判断的把握程度。\n"
                        "- reason（str）：一句话说明分类依据。\n\n"
                        "常见组合（尽量遵循，与后续编排一致）：\n"
                        "- 寒暄/常识 → retrieval_need=false + vector/simple/direct；\n"
                        "- 单点事实 → simple + vector 或 hybrid + citation；\n"
                        "- 含指代/省略 → rewrite + hybrid 或 multi_recall + citation；\n"
                        "- 多跳/流程 → multihop + multi_recall + citation；\n"
                        "- 对比/多实体 → decompose + multi_recall + comparison。\n\n"
                        "输出必须严格是以下 JSON（不要输出任何其他文字）：\n"
                        '{"retrieval_need": true/false, "retrieval_mode": "vector|hybrid|multi_recall", '
                        '"complexity": "simple|rewrite|decompose|multihop", '
                        '"generation_mode": "direct|citation|comparison", '
                        '"confidence": 0.0~1.0, "reason": "一句话"}\n\n'
                        "当 retrieval_need=false 时，retrieval_mode/complexity/generation_mode 统一填 vector/simple/direct。\n\n"
                        "示例：\n"
                        '输入: 发票什么时候交\n'
                        '输出: {"retrieval_need": true, "retrieval_mode": "hybrid", '
                        '"complexity": "simple", "generation_mode": "citation", '
                        '"confidence": 0.97, "reason": "单点事实含领域词，混合检索"}\n'
                        '输入: 出差和报销有什么区别\n'
                        '输出: {"retrieval_need": true, "retrieval_mode": "multi_recall", '
                        '"complexity": "decompose", "generation_mode": "comparison", '
                        '"confidence": 0.9, "reason": "多实体对比，需拆分子查询"}\n'
                        '输入: 那补卡流程呢（依赖上文）\n'
                        '输出: {"retrieval_need": true, "retrieval_mode": "hybrid", '
                        '"complexity": "rewrite", "generation_mode": "citation", '
                        '"confidence": 0.88, "reason": "含指代且为流程问题，需改写后检索"}\n'
                        '输入: 报销发票什么时候交\n'
                        '输出: {"retrieval_need": true, "retrieval_mode": "multi_recall", '
                        '"complexity": "multihop", "generation_mode": "citation", '
                        '"confidence": 0.86, "reason": "多跳流程，需迭代检索"}\n'
                        '输入: 张三的领导有几天年假\n'
                        '输出: {"retrieval_need": true, "retrieval_mode": "multi_recall", '
                        '"complexity": "multihop", "generation_mode": "citation", '
                        '"confidence": 0.85, "reason": "实体链多跳，先定位领导再查年假"}\n'
                        '输入: 你好，你是谁\n'
                        '输出: {"retrieval_need": false, "retrieval_mode": "vector", '
                        '"complexity": "simple", "generation_mode": "direct", '
                        '"confidence": 0.99, "reason": "闲聊元问题，无需检索"}'
                    )
                ),
                HumanMessage(content=query),
            ]
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            decision = self._parse(content)
            if decision.confidence < _LOW_CONFIDENCE_THRESHOLD:
                return self._fallback.classify(query)
            return decision
        except Exception:  # noqa: BLE001 — LLM 失败回退规则路由
            return self._fallback.classify(query)

    @staticmethod
    def _parse(content: str) -> RouteDecision:
        """提取 JSON 片段并校验枚举白名单；非法/不可解析抛异常交给调用方回退。"""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("LLM 路由输出中未找到 JSON")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("路由决策 JSON 必须是对象")
        mode = data.get("retrieval_mode")
        complexity = data.get("complexity")
        gen = data.get("generation_mode")
        if mode not in _VALID_MODES or complexity not in _VALID_COMPLEXITY or gen not in _VALID_GEN:
            raise ValueError("路由决策枚举值非法")
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return RouteDecision(
            retrieval_need=bool(data.get("retrieval_need", True)),
            retrieval_mode=mode,
            complexity=complexity,
            generation_mode=gen,
            confidence=max(0.0, min(1.0, confidence)),
            reason=str(data.get("reason", "") or ""),
        )


class RuleQueryClassifier(QueryClassifier):
    """确定性规则路由（无 LLM 回退），按优先级判定五维度。"""

    def classify(self, query: str) -> RouteDecision:
        # 1. 对比/多实体：需分解 + 多路召回 + 对比生成
        if _COMPARISON.search(query):
            return RouteDecision(
                retrieval_need=True,
                retrieval_mode=MULTI_RECALL,
                complexity=DECOMPOSE,
                generation_mode=COMPARISON,
                confidence=0.8,
                reason="多实体/对比，需拆分子查询",
            )
        # 2. 多跳/流程：需迭代检索（多轮召回拼出中间环节）
        if _MULTI_HOP.search(query):
            return RouteDecision(
                retrieval_need=True,
                retrieval_mode=MULTI_RECALL,
                complexity=MULTIHOP,
                generation_mode=CITATION,
                confidence=0.75,
                reason="多跳/流程，需迭代检索",
            )
        # 3. 指代/歧义：依赖上下文，需改写
        if _DEICTIC.search(query):
            return RouteDecision(
                retrieval_need=True,
                retrieval_mode=HYBRID,
                complexity=REWRITE,
                generation_mode=CITATION,
                confidence=0.7,
                reason="含指代/歧义，需改写后检索",
            )
        # 4. 寒暄/元问题：无需检索，直接生成
        if _GREETING.search(query):
            return RouteDecision(
                retrieval_need=False,
                retrieval_mode=VECTOR,
                complexity=SIMPLE,
                generation_mode=DIRECT,
                confidence=0.95,
                reason="寒暄/常识，无需检索",
            )
        # 5. 含领域关键词/数字：混合检索（语义 + 关键词精确命中）
        if _KEYWORD.search(query) or re.search(r"\d+", query):
            return RouteDecision(
                retrieval_need=True,
                retrieval_mode=HYBRID,
                complexity=SIMPLE,
                generation_mode=CITATION,
                confidence=0.6,
                reason="含领域关键词/数字，混合检索",
            )
        # 6. 默认：单点事实，纯语义检索
        return RouteDecision(
            retrieval_need=True,
            retrieval_mode=VECTOR,
            complexity=SIMPLE,
            generation_mode=CITATION,
            confidence=0.55,
            reason="单点事实，单次检索",
        )


# 领域关键词：来自 query_rewrite 的关键词表（避免循环导入，此处内联一份）
_KEYWORD = re.compile(
    r"(考勤|打卡|迟到|早退|旷工|补卡|年假|事假|病假|福利|补贴|出差|差旅|报销|"
    r"住宿|交通|餐补|发票|审批|工资|绩效|城市|一线|二线)"
)


def build_classifier() -> QueryClassifier:
    """构造路由器：有 LLM 场景配置用 LLM 路由（内部懒取），否则规则回退。"""
    return LLMQueryClassifier()
