"""知识库检索工具：把 agentic 方案的完整检索管线封装为主循环内的单一工具。

企业级混合架构的「L2 主循环内检索」：
- naive / advanced / modular：维持前置检索（L0/L1，不进入工具集）；
- agentic：检索决策交给主 Agent —— 由主循环内的 knowledge_retrieve 工具按需触发，
  检索执行仍复用方案的完整管线（语义路由/改写/多跳/充分性校验），过程事件经 emit 实时下发，
  跨轮 seed 缓存由工具维护。

rag_block_payload 同时供前置注入（naive/advanced/modular）与本工具返回共用，保证指令一致。

检索结果预算治理（企业级「检索侧消化数据量，落盘只兜底」）：
- 块总长受 _MAX_BLOCK_CHARS 约束（低于上下文落盘阈值），每条命中按预算句末截断成
  「要点 + 来源编号」——大检索结果不再全量进上下文，落盘几乎不触发；
  需要更细依据时由模型再次检索或读取落盘文件。
"""
from __future__ import annotations

import logging
import time

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 数据校验边界：工具查询为空 / 超长时降级为未命中（安全加固，不进入检索管线）
_MAX_QUERY_CHARS = 500
# 检索结果块总字符预算：低于上下文落盘阈值（context_offload_threshold=3000），
# 保证要点化后的块通常不触发落盘（落盘仅作超大输出兜底）。
_MAX_BLOCK_CHARS = 2400


def build_sources(hits: list[dict]) -> list[str]:
    """从命中元数据生成来源清单：卷/章/节 → 文件 → 兜底「知识库」。"""
    sources = []
    for i, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        parts = [x for x in (meta.get("volume"), meta.get("chapter"), meta.get("section")) if x]
        if parts:
            label = " / ".join(parts)
        else:
            src = meta.get("source")
            label = src if src and src != "builtin" else "知识库"
        sources.append(f"[{i}] {label}")
    return sources


def _terms(text: str) -> set[str]:
    """中文相邻 2 字词集合：query 相关度粗判（与检索侧同口径，确定性零成本）。"""
    import re

    seg = re.findall(r"[\u4e00-\u9fff]+", text)
    return {s[i : i + 2] for s in seg for i in range(len(s) - 1)}


def _relevance(unit: str, q_terms: set[str]) -> float:
    """单元与查询的相关度：与查询共现的 2 字词占单元词数比（[0,1]）。"""
    if not q_terms:
        return 0.0
    ut = _terms(unit)
    if not ut:
        return 0.0
    return len(ut & q_terms) / max(1, len(ut))


def _clip_text(text: str, limit: int, query: str = "") -> str:
    """按单条预算截断检索命中：优先保留 query 相关句（不从头硬切）。

    query 提供时按相关度选取句子、原文顺序输出——关键条款/差异点即使落在
    截断点之后也能被保留；无 query / 无句结构时回退句末截断兜底。
    """
    if len(text) <= limit:
        return text
    if query:
        import re as _re

        sentences = [s for s in _re.split(r"(?<=[。！？；；])\s*|\n+", text) if s.strip()]
        if len(sentences) > 1:
            q_terms = _terms(query)
            rel = [_relevance(s, q_terms) for s in sentences] if q_terms else [0.0] * len(sentences)
            picked: list[int] = []
            used = 0
            for i in sorted(range(len(sentences)), key=lambda i: rel[i], reverse=True):
                s = sentences[i]
                if used + len(s) > limit:
                    continue
                picked.append(i)
                used += len(s) + 2
            if picked:
                picked.sort()
                out = "".join(sentences[i] for i in picked)
                if len(picked) < len(sentences):
                    out = out.rstrip() + "…"
                return out
    cut = text[:limit]
    for sep in ("。", "！", "？", "；", "\n"):
        idx = cut.rfind(sep)
        if idx >= limit // 2:
            return cut[: idx + 1] + "…"
    return cut + "…"


def _hit_budget(fixed_chars: int, n: int) -> int:
    """按总预算为每条命中分配字符：固定部分（头部/来源/指令）先行扣减，下限 120 字。

    每条命中 ≤ per 时，总块 ≤ _MAX_BLOCK_CHARS（per 上限触发时落盘作兜底）。
    """
    if n <= 0:
        return 0
    return max(120, (_MAX_BLOCK_CHARS - fixed_chars) // n)


def rag_block_payload(rag_context: dict, insufficient: bool, generation_mode: str | None, query: str = "") -> str:
    """检索结果块正文（不含用户消息前缀）；前置注入与 knowledge_retrieve 工具返回共用。

    企业级预算治理：块总长受 _MAX_BLOCK_CHARS 约束（低于上下文落盘阈值）——
    每条命中按预算保留 query 相关句（关键信息不因截断丢失）成「要点 + 来源编号」，
    大检索结果不再全量进上下文、落盘几乎不触发；需要更细依据时由模型再次检索或读取落盘文件。
    指令结构（编号/来源清单/对比表格要求）保持不变，前置与工具指令一致。
    """
    hits = rag_context["hits"]
    name = rag_context["name"]
    head = f"【知识库检索结果（{name}）】\n"
    if insufficient:
        instruction = (
            "请严格基于以上检索内容如实回答；若检索内容不足以回答用户问题，"
            "请明确说明缺失的关键信息，并礼貌地向用户追问补充，不要编造、"
            "不要依赖自身知识臆测内部人事数据。"
        )
        per = _hit_budget(len(head) + len(instruction), len(hits))
        blocks = "\n".join(f"[{i}] {_clip_text(h['text'], per, query)}" for i, h in enumerate(hits, start=1))
        return f"{head}{blocks}\n{instruction}"
    mode = generation_mode or "citation"
    if mode == "direct":
        instruction = "请直接回答用户问题，无需标注引用来源。"
        sources = ""
        numbered = False
    elif mode == "comparison":
        instruction = (
            "请基于以上检索内容，用 Markdown 对比表格结构化回答：每一行一个对比维度，"
            "每一列一个对比对象；表格内关键结论在句末标注上方编号 [1]/[2] 的来源。"
            "表格之后用一段话总结差异，末尾附「引用来源」清单（编号 → 出处）。"
        )
        sources = "\n".join(build_sources(hits))
        numbered = True
    else:  # citation
        instruction = (
            "请优先基于以上检索内容回答；每个关键事实在句末标注上方编号 [1]/[2] 的来源。"
            "回答末尾附「引用来源」清单（编号 → 出处）。"
            "若检索内容不足以回答，再结合自身知识补充，并注明哪些属于推测。"
        )
        sources = "\n".join(build_sources(hits))
        numbered = True
    per = _hit_budget(len(head) + len(instruction) + len(sources), len(hits))
    if numbered:
        blocks = "\n".join(f"[{i}] {_clip_text(h['text'], per, query)}" for i, h in enumerate(hits, start=1))
    else:
        blocks = "\n".join(_clip_text(h["text"], per, query) for h in hits)
    tail = f"来源：\n{sources}\n" if sources else ""
    return f"{head}{blocks}\n{tail}{instruction}"


def rag_status_line(
    rag_context: dict,
    insufficient: bool,
    confidence: float | None = None,
    cost: dict | None = None,
    missing_facts: list[str] | None = None,
) -> str:
    """检索状态摘要（结构化一行）：主 Agent 直接读到充分性/缺口/置信度/成本。

    层间契约的文本侧表现：与结构化元数据（holder["rag_state"]）同源，
    让外层模型不必解析长正文即可做出「追问/换词/直接作答」决策。
    """
    hits = rag_context.get("hits") or []
    state = "未命中" if not hits else ("不足" if insufficient else "充分")
    parts = [f"充分性={state}"]
    if insufficient and missing_facts:
        parts.append(f"缺失事实={'、'.join(missing_facts[:3])}")
    if confidence is not None:
        parts.append(f"置信度={confidence:.2f}")
    if cost:
        parts.append(f"调用={cost.get('calls', 0)}")
        ms = cost.get("latency_ms") or 0
        if ms:
            parts.append(f"耗时={ms / 1000:.1f}s")
    return f"【检索状态】{' | '.join(parts)}"


def make_knowledge_retrieve_tool(scheme, settings, emit, session_id, last_hits, context_holder):
    """构建 knowledge_retrieve 工具：agentic 全管线封装。

    context_holder：可变的最近会话上下文容器（{"recent": str|None}），由 runner 在前置阶段
    填充，供工具内指代消解使用（主 Agent 生成的查询可能含「他/这…」）。
    last_hits / session_id：跨轮 seed 缓存读写（上一轮最终命中供本轮过滤复用）。
    """

    @tool
    async def knowledge_retrieve(query: str) -> str:
        """检索公司内部知识库（制度、员工档案、FAQ、案例、SOP 等），返回与问题相关的依据文本与来源编号。

        当问题涉及公司制度/流程/人事/员工权益等内部知识，或需要引用知识库来源回答时调用；
        一次检索不充分可换更具体的关键词再次调用。返回内容含引用来源与生成策略，请据此组织回答；
        若返回提示「检索结果不足」，请向用户追问澄清关键信息，不要编造内部数据。
        """
        # 数据校验（安全边界）：空 / 超长查询直接降级，不进入检索管线
        if not query or not query.strip():
            return "检索查询为空，请向用户补充要查询的具体内容后再检索。"
        query = query.strip()[:_MAX_QUERY_CHARS]
        started = time.monotonic()
        holder = context_holder if context_holder is not None else {}
        context = holder.get("recent")
        prev_hits = last_hits.get(session_id)
        stream_kwargs = {"context": context}
        if prev_hits:
            stream_kwargs["seed_hits"] = prev_hits
        # 复用主循环外前置的路由决策（classify 产出）：编排器跳过 RouterAgent，
        # 省一次 LLM 且保证生成策略与前置提示一致。
        pre_route = holder.get("route")
        if pre_route:
            stream_kwargs["pre_route"] = pre_route
        rag_context = None
        generation_mode = None
        insufficient = False
        confidence = None
        cost = None
        missing_facts: list[str] = []
        try:
            async for ev in scheme.astream(query, settings.rag_top_k, **stream_kwargs):
                if emit is not None:
                    emit(ev)
                if ev["type"] == "classify":
                    generation_mode = ev.get("generation_mode") or generation_mode
                elif ev["type"] == "retrieve":
                    if ev.get("hits"):
                        rag_context = {"name": scheme.name, "hits": ev["hits"]}
                elif ev["type"] == "answerability":
                    verdict = ev.get("verdict") or {}
                    if verdict.get("answerable") is False:
                        insufficient = True
                    missing_facts = list(verdict.get("missing_facts") or [])
                    confidence = ev.get("confidence")
                    cost = ev.get("cost")
        except Exception as exc:  # noqa: BLE001 — 检索故障降级为未命中，不向 Agent 抛工具异常
            logger.exception("[rag_tool] knowledge_retrieve 执行失败 scheme=%s session=%s", getattr(scheme, "id", ""), session_id)
            return (
                f"知识库检索执行失败（{type(exc).__name__}）。请如实告知用户当前检索遇到故障，"
                "建议稍后重试或更换关键词；不要编造内部数据。"
            )
        finally:
            logger.info(
                "[rag_tool] knowledge_retrieve 完成 scheme=%s session=%s hits=%d insufficient=%s mode=%s 耗时=%.3fs",
                getattr(scheme, "id", ""), session_id,
                len((rag_context or {}).get("hits") or []), insufficient,
                generation_mode or "-", time.monotonic() - started,
            )
        # 更新跨轮 seed 缓存：本轮检索命中记录（供下一轮复用），否则清空
        if rag_context and rag_context.get("hits"):
            last_hits[session_id] = rag_context["hits"]
        else:
            last_hits.pop(session_id, None)
        if not rag_context or not rag_context.get("hits"):
            return (
                f"知识库检索未命中与“{query}”相关的内容。请如实告知用户当前检索未能获取足够信息，"
                "说明缺失的关键信息，并礼貌地向用户追问补充依据（如具体文件、部门名称等）；"
                "不要编造、不要依赖自身知识臆测内部数据。"
            )
        block = rag_block_payload(rag_context, insufficient, generation_mode, query)
        # 层间结构化契约（旁路透传）：正文供模型作答，结构化元数据供外层程序化消费
        holder["rag_state"] = {
            "query": query,
            "verdict": {
                "answerable": not insufficient,
                "missing_facts": missing_facts,
                "recommendation": "answer" if not insufficient else "clarify",
            },
            "confidence": confidence,
            "cost": cost,
            "hits": len(rag_context["hits"]),
        }
        return f"{block}\n{rag_status_line(rag_context, insufficient, confidence, cost, missing_facts)}"

    return knowledge_retrieve
