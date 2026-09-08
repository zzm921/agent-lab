"""Agent 评测 · 架构级 L2 约束（架构不变量）。

任务层（task_set.jsonl）对所有 agent 一致；本模块是「架构特定」的不变量——
不管用例怎么变，对应架构必须满足的协议特征：

- plan_execute：必须产出 plan created（子任务数 ∈ [2,5]，与 _PLAN_PROMPT 对齐）且以 plan done 收尾；
- reflection：  必须产出 reflect(draft) 与 critique 事件，评审结论必须以 PASS 通过；
- multi_agent： 必须委派 compute / analyze 两个 worker（agent_event dispatch）；
- react：       线性思考-行动循环，无常设不变量（轨迹约束下沉到用例级 sequence / must_call）。

这些不变量与具体任务无关，任何该模式用例都须满足——防止架构协议悄悄退化
（如 plan 事件丢失、评审永远不通过、编排者不再委派 worker）。
"""
from __future__ import annotations

from typing import Any

# 与 app/agents/modes/plan_execute.py 的 _PLAN_PROMPT「2-5 个有序子步骤」保持一致
PLAN_STEP_RANGE = (2, 5)
# 与 reflection 的 max_iter 语义一致的评审通过判定（PASS 开头）
MULTI_AGENT_WORKERS = ("compute", "analyze")


def _critique_texts(events: list[dict[str, Any]]) -> list[str]:
    """critique 事件为增量 delta 分片；相邻 critique 增量归并为同一轮评审文本，跨轮分段。"""
    texts: list[str] = []
    cur = ""
    for ev in events:
        if ev.get("type") == "critique":
            cur += str(ev.get("delta") or "")
        elif cur:
            texts.append(cur)
            cur = ""
    if cur:
        texts.append(cur)
    return texts


def check_arch_spec(mode: str, events: list[dict[str, Any]]) -> list[str]:
    """返回违规明细（空列表 = 通过）。"""
    if mode == "plan_execute":
        created = [ev for ev in events if ev.get("type") == "plan" and ev.get("status") == "created"]
        if not created:
            return ["缺少 plan created 事件（计划层未产出）"]
        items = created[0].get("items") or []
        if not (PLAN_STEP_RANGE[0] <= len(items) <= PLAN_STEP_RANGE[1]):
            return [f"计划子任务数 {len(items)} 超出 [{PLAN_STEP_RANGE[0]}, {PLAN_STEP_RANGE[1]}]"]
        if not any(ev.get("type") == "plan" and ev.get("status") == "done" for ev in events):
            return ["缺少 plan done 事件（计划未正常走完）"]
        return []

    if mode == "reflection":
        if not any(ev.get("type") == "reflect" and ev.get("stage") == "draft" for ev in events):
            return ["缺少 reflect(draft) 事件（生成阶段未产出草稿）"]
        # 可能经历多轮「评审→修订」：以最后一轮评审结论为准（修订场景首轮可能 FAIL）
        texts = _critique_texts(events)
        last = texts[-1] if texts else ""
        head = (last or "").strip().splitlines()[0] if (last or "").strip() else ""
        if not head.startswith("【PASS】") and not head.startswith("PASS"):
            return [f"最终评审未通过：critique 缺失或未以 PASS 开头（critique={last[:30]!r}）"]
        return []

    if mode == "multi_agent":
        dispatched = {
            ev.get("worker")
            for ev in events
            if ev.get("type") == "agent_event" and ev.get("status") == "dispatch"
        }
        missing = sorted(set(MULTI_AGENT_WORKERS) - dispatched)
        if missing:
            return [f"编排者未委派 worker：{missing}"]
        return []

    return []  # react：无常设架构不变量
