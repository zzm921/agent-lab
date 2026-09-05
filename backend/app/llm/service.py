"""独立 LLM 服务模块：按业务场景统一管理模型与参数。

- LLMProfile：单个场景的配置（供应商 / 模型 / 生成参数 / 可覆盖的 Key 与地址）；
- LLMService：统一入口——按场景注册供应商（扩展新模型）、配置与动态调整场景参数、
  惰性构建并缓存实例；get(scenario) 返回已绑定该场景参数的模型；
- LoggedChatModel：对底层模型做日志 + 错误包装，记录每次调用的
  场景 / 模型 / 参数 / 延迟 / 是否成功 / 错误，异常统一包装为 LLMError。

新增模型只需 register_provider(name, builder)；调整某场景参数只需 update_profile。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import BaseModel, Field, PrivateAttr

from app.config import settings
from app.core.errors import ConfigError, LLMError
from app.llm.dashscope_chat import DashScopeChatModel
from app.llm.fake_model import FakeChatModel
from app.telemetry.sink import ACTIVE_SINK

logger = logging.getLogger(__name__)


class LLMProfile(BaseModel):
    """单个业务场景的 LLM 配置。"""

    scenario: str = Field(description="业务场景名，如 chat / planner / critic / rag_rewrite")
    provider: str = Field(default="dashscope", description="模型供应商（须已注册）")
    model: str = Field(default="", description="模型名；留空回退全局 settings.chat_model")
    params: dict[str, Any] = Field(default_factory=dict, description="生成参数：temperature / max_tokens / top_p / enable_thinking 等")
    api_key: str = Field(default="", description="本场景专属 API Key；留空回退全局")
    base_url: str = Field(default="", description="本场景专属服务地址；留空回退全局")


# 默认场景配置：按业务区分模型与参数（可通过 update_profile 动态调整）
DEFAULT_PROFILES: list[dict[str, Any]] = [
    {
        "scenario": "chat",  # 主对话 Agent（react / plan_execute 执行 / multi_agent）
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.7, "enable_thinking": True},
    },
    {
        "scenario": "memory_consolidate",  # 轮末记忆提取：结构化小 JSON，关闭思考以加速（实测 thinking 40s+ / 关闭 2s）
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.2, "max_tokens": 500, "enable_thinking": False},
    },
    {
        "scenario": "memory_selector",  # 主动语义召回触发判断：输出极小 JSON（need），关闭思考加速
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.0, "max_tokens": 80, "enable_thinking": False},
    },
    {
        "scenario": "planner",  # plan_execute 的任务规划/重规划：低随机性，输出精炼
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.2, "max_tokens": 300, "enable_thinking": True},
    },
    {
        "scenario": "critic",  # reflection 评审：严格、确定性
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.2, "enable_thinking": False},
    },
    {
        "scenario": "rag_rewrite",  # advanced RAG Query 重写：低随机 + 输出长度受限
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.3, "max_tokens": 200, "enable_thinking": False},
    },
    {
        "scenario": "rag_classify",  # 语义路由：输出结构化 JSON 决策（五维度），低随机 + 输出受限
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.2, "max_tokens": 500, "enable_thinking": False},
    },
    {
        "scenario": "rag_decompose",  # 查询分解：输出多行子问题，低随机
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.3, "max_tokens": 300, "enable_thinking": False},
    },
    {
        "scenario": "rag_plan",  # 多跳子查询规划：输出结构化计划 JSON（轻量决策，关闭思考以加速）
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.2, "max_tokens": 300, "enable_thinking": False},
    },
    {
        "scenario": "rag_verify",  # 多跳验证对表：输出结构化 JSON（轻量决策，关闭思考）
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.2, "max_tokens": 400, "enable_thinking": False},
    },
    {
        "scenario": "rag_next_step",  # 多跳下一跳决策：输出结构化 JSON（轻量决策，关闭思考）
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.3, "max_tokens": 300, "enable_thinking": False},
    },
    {
        "scenario": "rag_hyde",  # HyDE 假想答案文档生成：输出一段连贯说明文字
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.3, "max_tokens": 300, "enable_thinking": False},
    },
    {
        "scenario": "rag_agent_route",  # Agentic RAG 路由角色：检索必要性/生成策略 JSON（轻量决策）
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.1, "max_tokens": 250, "enable_thinking": False},
    },
    {
        "scenario": "rag_agent_plan",  # Agentic RAG 规划角色：事实清单 + 首发检索计划 JSON
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.2, "max_tokens": 500, "enable_thinking": False},
    },
    {
        "scenario": "rag_agent_grade",  # Agentic RAG 评审角色（CRAG）：逐条证据相关性 + 缺失事实 JSON
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.1, "max_tokens": 500, "enable_thinking": False},
    },
    {
        "scenario": "rag_agent_correct",  # Agentic RAG 纠错角色（CRAG）：纠错波工具调用 JSON
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.2, "max_tokens": 400, "enable_thinking": False},
    },
    {
        "scenario": "rag_agent_verify",  # Agentic RAG 校验角色（Self-RAG）：事实-证据支持度 JSON
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.1, "max_tokens": 400, "enable_thinking": False},
    },
    {
        "scenario": "rag_judge",  # L2 语义评测 judge（LLM-as-a-Judge）：输出结构化评分，低随机 + 关闭思考
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.1, "max_tokens": 300, "enable_thinking": False},
    },
    {
        "scenario": "rag_ragas",  # RAGAS 内部 LLM（全面评测语义评分）：高频小 JSON 提取，关闭思考加速
        "provider": "dashscope",
        "model": "qwen3.5-flash",
        "params": {"temperature": 0.1, "max_tokens": 1500, "enable_thinking": False},
    },
    {
        "scenario": "fake",  # 测试 / 无 Key 回退
        "provider": "fake",
        "model": "fake-chat",
        "params": {},
    },
]


def _build_dashscope(profile: LLMProfile) -> BaseChatModel:
    """DashScope 供应商：从 profile 读取，未配置项回退全局 settings。"""
    if not profile.api_key and not settings.llm_api_key:
        raise ConfigError("未配置 LLM_API_KEY（阿里云百炼 DashScope API Key），请在 backend/.env 中设置后重启服务")
    return DashScopeChatModel(
        model_name=profile.model or settings.chat_model,
        api_key=profile.api_key or settings.llm_api_key,
        base_url=profile.base_url or settings.llm_base_url,
        temperature=0.3,  # 默认；profile.params 经 bind 覆盖
        enable_thinking=settings.enable_thinking,
    )


def _build_fake(profile: LLMProfile) -> BaseChatModel:
    """Fake 供应商：确定性输出，供测试 / 离线回退。"""
    return FakeChatModel(model_name=profile.model or "fake-chat")


def _usage_tokens(obj) -> dict[str, int] | None:
    """从一次调用的结果对象提取 token 用量；无则 None。

    优先级：
    1. usage_metadata（input_tokens/output_tokens/total_tokens）——本项目 DashScope
       适配层统一写入此字段（非流式设在消息上、流式设在末块上），是记账的权威来源；
    2. response_metadata.usage / token_usage（prompt_tokens/completion_tokens/total_tokens）
       ——兼容其他供应商习惯。
    入参可为 ChatResult / ChatGenerationChunk / BaseMessage(Chunk)。
    """
    if obj is None:
        return None
    if isinstance(obj, ChatResult):
        gens = getattr(obj, "generations", None) or []
        msg = gens[0].message if gens else None
    else:
        msg = getattr(obj, "message", obj)  # ChatGenerationChunk → .message；裸消息/块原样
    if msg is None:
        return None
    meta = getattr(msg, "usage_metadata", None) or {}
    if meta.get("total_tokens") is not None or meta.get("input_tokens") is not None:
        return {
            "input": int(meta.get("input_tokens", 0) or 0),
            "output": int(meta.get("output_tokens", 0) or 0),
            "total": int(meta.get("total_tokens", 0) or 0),
        }
    rmeta = getattr(msg, "response_metadata", None) or {}
    usage = rmeta.get("usage") or rmeta.get("token_usage")
    if not usage:
        return None
    return {
        "input": int(usage.get("prompt_tokens", 0) or 0),
        "output": int(usage.get("completion_tokens", 0) or 0),
        "total": int(usage.get("total_tokens", 0) or 0),
    }


class LoggedChatModel(BaseChatModel):
    """带日志与错误包装的聊天模型：委托真实模型，记录调用过程并统一异常。"""

    scenario: str = "default"
    params: dict[str, Any] = Field(default_factory=dict)
    model_name: str = ""

    _inner: Any = PrivateAttr(default=None)

    def __init__(
        self,
        inner: BaseChatModel,
        scenario: str = "default",
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=getattr(inner, "model_name", "") or "",
            scenario=scenario,
            params=dict(params or {}),
            **kwargs,
        )
        self._inner = inner

    @property
    def _llm_type(self) -> str:
        return getattr(self._inner, "_llm_type", "logged-chat")

    def _log_start(self, method: str) -> None:
        logger.info(
            "[llm:%s] %s 开始 model=%s params=%s",
            self.scenario,
            method,
            self.model_name,
            self.params,
        )

    def _wrap_error(self, exc: Exception, method: str) -> LLMError:
        logger.error(
            "[llm:%s] %s 失败 model=%s params=%s: %s",
            self.scenario,
            method,
            self.model_name,
            self.params,
            exc,
            exc_info=True,
        )
        return LLMError(scenario=self.scenario, model=self.model_name, params=self.params, method=method, cause=exc)

    def _record(self, method: str, latency_ms: float, success: bool, tokens: dict | None = None, error: str = "") -> None:
        """把一次 LLM 调用写入当前活动的运行记录（TelemetrySink）；无 sink 时静默跳过。

        success=False 也会记录（失败是最需要观测的调用）：观测优先，不影响主流程。
        """
        sink = ACTIVE_SINK.get()
        if sink is None:
            return
        call: dict[str, Any] = {
            "scenario": self.scenario,
            "model": self.model_name,
            "method": method,
            "latency_ms": latency_ms,
            "success": success,
        }
        if tokens:
            call["tokens"] = tokens
        if error:
            call["error"] = error[:200]
        sink.record_llm(call)


    def bind(self, **kwargs: Any) -> "LoggedChatModel":
        """把附加参数绑定到底层模型并保持包装（保留日志/错误能力）。"""
        return LoggedChatModel(inner=self._inner.bind(**kwargs), scenario=self.scenario, params=self.params)

    def bind_tools(self, tools, *, tool_choice: str | None = None, **kwargs: Any) -> "LoggedChatModel":
        """把工具绑定到底层模型并保持包装。"""
        return LoggedChatModel(
            inner=self._inner.bind_tools(tools, tool_choice=tool_choice, **kwargs),
            scenario=self.scenario,
            params=self.params,
        )

    # ---- 同步调用：委托底层私有方法（_generate/_stream 契约稳定，不受公开层包装影响）----
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._log_start("invoke")
        start = time.perf_counter()
        try:
            result = self._inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:  # noqa: BLE001 — 统一包装为 LLMError
            self._record("invoke", (time.perf_counter() - start) * 1000, False, error=str(exc))
            raise self._wrap_error(exc, "invoke") from exc
        usage = _usage_tokens(result)
        self._record(
            "invoke",
            (time.perf_counter() - start) * 1000,
            True,
            tokens=usage,
        )
        logger.info("[llm:%s] invoke 完成 latency=%.3fs", self.scenario, time.perf_counter() - start)
        return result

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._log_start("stream")
        start = time.perf_counter()
        last = None
        try:
            # 公开 stream() 已按 _should_stream 分流：流式模型逐块产出，非流式回退整条消息；
            # 统一转成 ChatGenerationChunk（整条消息需转成 AIMessageChunk）。
            for chunk in self._inner.stream(messages, stop=stop, **kwargs):
                if not isinstance(chunk, BaseMessageChunk):
                    chunk = AIMessageChunk(**chunk.model_dump(exclude={"type"}))
                last = chunk
                yield ChatGenerationChunk(message=chunk)
        except Exception as exc:  # noqa: BLE001
            self._record("stream", (time.perf_counter() - start) * 1000, False, error=str(exc))
            raise self._wrap_error(exc, "stream") from exc
        self._record("stream", (time.perf_counter() - start) * 1000, True, tokens=_usage_tokens(last))
        logger.info("[llm:%s] stream 完成 latency=%.3fs", self.scenario, time.perf_counter() - start)

    # ---- 异步调用 ----
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._log_start("ainvoke")
        start = time.perf_counter()
        try:
            result = await self._inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._record("ainvoke", (time.perf_counter() - start) * 1000, False, error=str(exc))
            raise self._wrap_error(exc, "ainvoke") from exc
        usage = _usage_tokens(result)
        self._record(
            "ainvoke",
            (time.perf_counter() - start) * 1000,
            True,
            tokens=usage,
        )
        logger.info("[llm:%s] ainvoke 完成 latency=%.3fs", self.scenario, time.perf_counter() - start)
        return result

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self._log_start("astream")
        start = time.perf_counter()
        last = None
        try:
            async for chunk in self._inner.astream(messages, stop=stop, **kwargs):
                if not isinstance(chunk, BaseMessageChunk):
                    chunk = AIMessageChunk(**chunk.model_dump(exclude={"type"}))
                last = chunk
                yield ChatGenerationChunk(message=chunk)
        except Exception as exc:  # noqa: BLE001
            self._record("astream", (time.perf_counter() - start) * 1000, False, error=str(exc))
            raise self._wrap_error(exc, "astream") from exc
        self._record("astream", (time.perf_counter() - start) * 1000, True, tokens=_usage_tokens(last))
        logger.info("[llm:%s] astream 完成 latency=%.3fs", self.scenario, time.perf_counter() - start)


class LLMService:
    """LLM 统一服务：供应商注册 + 场景配置 + 实例构建与缓存。"""

    def __init__(self, profiles: Iterable[LLMProfile | dict] | None = None):
        self._providers: dict[str, Callable[[LLMProfile], BaseChatModel]] = {}
        self._profiles: dict[str, LLMProfile] = {}
        self._cache: dict[str, BaseChatModel] = {}
        self.register_provider("dashscope", _build_dashscope)
        self.register_provider("fake", _build_fake)
        for profile in profiles if profiles is not None else DEFAULT_PROFILES:
            self.set_profile(profile)

    def register_provider(self, name: str, builder: Callable[[LLMProfile], BaseChatModel]) -> None:
        """注册一个模型供应商（builder：profile → 模型实例），便于扩展新模型。"""
        self._providers[name] = builder

    def set_profile(self, profile: LLMProfile | dict) -> LLMProfile:
        """新增/覆盖一个场景配置；旧实例失效（下次 get 重建）。"""
        p = profile if isinstance(profile, LLMProfile) else LLMProfile(**profile)
        self._profiles[p.scenario] = p
        self._cache.pop(p.scenario, None)
        return p

    def update_profile(self, scenario: str, **fields: Any) -> LLMProfile:
        """动态调整某场景的配置（如 temperature / model / params），立即生效。"""
        if scenario not in self._profiles:
            raise LLMError(f"未注册的 LLM 场景：{scenario}（可用：{sorted(self._profiles)}）")
        merged = self._profiles[scenario].model_copy(update=fields)
        self._profiles[scenario] = merged
        self._cache.pop(scenario, None)
        return merged

    def get(self, scenario: str) -> BaseChatModel:
        """按场景构建（并缓存）模型实例：已绑定该场景参数，套日志/错误包装。"""
        cached = self._cache.get(scenario)
        if cached is not None:
            return cached
        profile = self._profiles.get(scenario)
        if profile is None:
            raise LLMError(f"未注册的 LLM 场景：{scenario}（可用：{sorted(self._profiles)}）")
        builder = self._providers.get(profile.provider)
        if builder is None:
            raise LLMError(f"未注册的 LLM 供应商：{profile.provider}（可用：{sorted(self._providers)}）")
        model = builder(profile)
        if profile.params:
            model = model.bind(**profile.params)
        wrapped = LoggedChatModel(inner=model, scenario=scenario, params=dict(profile.params))
        self._cache[scenario] = wrapped
        return wrapped

    def clear(self, scenario: str | None = None) -> None:
        """清空实例缓存：scenario 为空则全部重建。"""
        if scenario is None:
            self._cache.clear()
        else:
            self._cache.pop(scenario, None)

    def list(self) -> list[dict[str, Any]]:
        """返回全部场景配置（供调试 / 展示）。"""
        return [
            {
                "scenario": p.scenario,
                "provider": p.provider,
                "model": p.model or settings.chat_model,
                "params": dict(p.params),
            }
            for p in self._profiles.values()
        ]
