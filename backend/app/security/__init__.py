"""安全防护能力：输入/输出 Guardrail、来源可信分级（Prompt 注入防御）、敏感数据脱敏。

对应 content/security.md 的三层 Guardrails（输入 → 工具 → 输出）：
- 输入：InputGuard 拦截越狱/提示注入高置信度特征；
- 工具：wrap_untrusted 给不可信外部内容打标记，隔离指令与数据（来源可信分级）；
- 输出：mask_sensitive 流式脱敏 + scan_output 敏感数据泄露阻断提示。
"""
from app.security.input_guard import DEFAULT_REFUSAL, GuardVerdict, InputGuard
from app.security.output_guard import OutputVerdict, StreamMasker, mask_sensitive, scan_output
from app.security.wrap import UNTRUSTED_TOOLS, is_untrusted_tool, wrap_untrusted

__all__ = [
    "DEFAULT_REFUSAL",
    "GuardVerdict",
    "InputGuard",
    "OutputVerdict",
    "StreamMasker",
    "UNTRUSTED_TOOLS",
    "is_untrusted_tool",
    "mask_sensitive",
    "scan_output",
    "wrap_untrusted",
]
