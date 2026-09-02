"""输入 Guardrail：越狱 / Prompt 注入 高置信度特征过滤，命中即礼貌拒绝。"""
from __future__ import annotations

from dataclasses import dataclass

from app.security.patterns import INPUT_BLOCK_PATTERNS

DEFAULT_REFUSAL = (
    "抱歉，我不能响应这条请求。检测到疑似越狱或提示注入的指令内容，"
    "为保障安全已停止处理。如有正常需求，请换一种方式描述。"
)


@dataclass
class GuardVerdict:
    rejected: bool
    reason: str = ""
    matched: str = ""


class InputGuard:
    """输入护栏：规则式过滤，命中高置信度注入特征即拒绝（无 LLM 依赖，可离线单测）。"""

    def __init__(self, enabled: bool = True, refusal: str = DEFAULT_REFUSAL):
        self.enabled = enabled
        self.refusal = refusal

    def check(self, text: str) -> GuardVerdict:
        if not self.enabled or not text:
            return GuardVerdict(rejected=False)
        for pat in INPUT_BLOCK_PATTERNS:
            m = pat.search(text)
            if m:
                return GuardVerdict(
                    rejected=True,
                    reason="检测到越狱/提示注入特征",
                    matched=m.group(0),
                )
        return GuardVerdict(rejected=False)
