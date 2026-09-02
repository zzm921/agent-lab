"""安全防护规则集：输入越狱/注入特征、输出敏感数据脱敏、输出违规阻断。

规则设计遵循 security.md 的「防得住与不误伤」平衡：只拦截高置信度的
指令性注入（忽略既有指令 / 越狱角色扮演 / 套取系统提示），避免误伤正常提问。
"""
from __future__ import annotations

import re

# ---- 输入 Guardrail：越狱 / Prompt 注入 高置信度特征（命中即礼貌拒绝）----
INPUT_BLOCK_PATTERNS: list[re.Pattern] = [
    # —— 中文：忽略 / 无视既有指令 ——
    re.compile(r"(?:忽略|无视)(?:之前|以上|前面|所有)?(?:的)?(?:所有)?(?:指令|提示|提示词|规则|设定|要求|system|系统提示)"),
    re.compile(r"(?:忘记|忽略|无视)(?:你|自己|系统)?(?:的)?(?:身份|角色|设定|规则|提示词|系统提示)"),
    re.compile(r"(?:不要再|不要继续)(?:遵守|遵循|服从|听从)(?:任何|之前的)?(?:指令|规则|设定|要求)"),
    # —— 中文：越狱角色扮演 / 切换越狱模式 ——
    re.compile(r"(?:你现在|请现在|假装)(?:是|扮演)(?:一个|一名)?(?:不受限|没有限制|无限制|越狱)"),
    # 注意：DAN 等英文 token 相邻中文时 \b 不可靠（中文按 Unicode 视为单词字符），
    # 用显式 ASCII 字符类做边界，兼容「你现在是DAN」这类中英混排。
    re.compile(r"(?<![A-Za-z0-9_])DAN(?:模式|mode)?(?![A-Za-z0-9_])", re.I),
    # —— 中文：套取系统提示 / 内部指令（token 间允许空白，兼容「泄露你的 system prompt」）——
    re.compile(r"(?:泄露|透露|说出|给出|展示|复述|打印)\s*(?:你的|系统)?\s*(?:system prompt|系统提示|提示词|指令|内部规则|初始设定)"),
    re.compile(r"(?:把|请把)\s*(?:你的|系统)?\s*(?:system prompt|系统提示|提示词|指令)\s*(?:发|发给|告诉|写|输出)\s*(?:给我|我|出来)?"),
    # —— 英文：经典注入 ——
    re.compile(r"\bignore (?:all )?(?:previous|above|prior) (?:instructions|prompts|rules|messages)\b", re.I),
    re.compile(r"\bignore everything (?:before|above|you have been told)\b", re.I),
    re.compile(r"\bdisregard (?:all )?(?:previous|above|prior) (?:instructions|prompts|rules|messages)\b", re.I),
    re.compile(r"\b(?:do anything now|dan mode|jailbreak)\b", re.I),
    re.compile(r"\breveal (?:your )?(?:system prompt|system instructions|initial instructions)\b", re.I),
    re.compile(r"\byou are now (?:jailbroken|dan|without restrictions|unrestricted)\b", re.I),
    re.compile(r"\bpretend you are (?:dan|unrestricted|without rules)\b", re.I),
]

# ---- 输出 Guardrail：敏感数据脱敏（顺序应用，先命中先替换）----


def _mask_phone(m: re.Match) -> str:
    s = m.group(0)
    return s[:3] + "****" + s[-4:]


def _mask_id(m: re.Match) -> str:
    s = m.group(0)
    return s[:6] + "********" + s[-4:]


def _mask_card(m: re.Match) -> str:
    s = m.group(0)
    return s[:4] + "********" + s[-4:]


def _mask_key(m: re.Match) -> str:
    s = m.group(0)
    keep_tail = 4
    return s[:6] + "****" + s[-keep_tail:] if len(s) > 10 + keep_tail else s[:4] + "****"


# (正则, 替换)：手机号 / 身份证 / 银行卡 / 常见密钥明文
SENSITIVE_MASK_RULES: list[tuple[re.Pattern, object]] = [
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), _mask_phone),              # 中国大陆手机号
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), _mask_id),                # 身份证号
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), _mask_card),                 # 银行卡号
    (re.compile(r"sk-[A-Za-z0-9_-]{12,}"), _mask_key),                   # OpenAI 风格 API Key
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), _mask_key),                    # AWS Access Key
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), _mask_key),                # GitHub Personal Token
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), _mask_key),  # JWT
]

# ---- 输出阻断：命中视为敏感数据泄露，追加 guard_refused 提示 ----
OUTPUT_BLOCK_PATTERNS: list[re.Pattern] = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
]
