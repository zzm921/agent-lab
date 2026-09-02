"""输出 Guardrail：敏感数据脱敏 + 违规内容阻断扫描。

- mask_sensitive：对最终输出做敏感数据脱敏（手机号/身份证/银行卡/密钥明文）；
- scan_output：全文扫描，命中敏感数据泄露规则时返回阻断结论（流式已发，事后提示）；
- StreamMasker：流式脱敏的滚动缓冲——只提交「完整分词」前缀（以空白为安全边界），
  尾部保留未完成的 token，避免把密钥/号码在正则匹配完成前提前发出。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.security.patterns import OUTPUT_BLOCK_PATTERNS, SENSITIVE_MASK_RULES


@dataclass
class OutputVerdict:
    blocked: bool
    reason: str = ""
    matched: str = ""


def mask_sensitive(text: str) -> str:
    """按规则顺序对文本中的敏感数据做脱敏替换。"""
    out = text or ""
    for pat, repl in SENSITIVE_MASK_RULES:
        out = pat.sub(repl, out)
    return out


def scan_output(text: str) -> OutputVerdict:
    """全文扫描：命中密钥明文等敏感数据泄露规则时返回阻断结论。"""
    for pat in OUTPUT_BLOCK_PATTERNS:
        m = pat.search(text or "")
        if m:
            return OutputVerdict(
                blocked=True,
                reason="检测到敏感数据（密钥/凭据）泄露",
                matched=m.group(0),
            )
    return OutputVerdict(blocked=False)


class StreamMasker:
    """流式脱敏：滚动缓冲 + 安全边界提交。

    只在「完整分词」处提交前缀（以空白/换行为边界），保证任一凭据 token 在被
    完整接收前不会提前发出；流结束调用 flush 冲刷剩余缓冲。
    """

    _SEPS = ("\n", "\t", " ")

    def __init__(self) -> None:
        self.buf = ""

    def push(self, text: str) -> str:
        """接收新增量，返回需要发射的已脱敏前缀（可能为空）。"""
        self.buf += text
        idx = -1
        for sep in self._SEPS:
            idx = max(idx, self.buf.rfind(sep))
        if idx <= 0:
            return ""
        commit, self.buf = self.buf[: idx + 1], self.buf[idx + 1 :]
        return mask_sensitive(commit)

    def flush(self) -> str:
        """冲刷剩余缓冲并返回脱敏结果。"""
        out = mask_sensitive(self.buf)
        self.buf = ""
        return out
