"""实测定向补召回多样性截断后的整链路效果（r21/r24/r25 挑战题）。"""
import sys

sys.path.insert(0, ".")
from eval.real_full import _build_real_scheme  # noqa: E402

QUERIES = [
    ("r21", "李雪可以申请远程办公吗？"),
    ("r24", "2026版和2024版在免费补卡上有什么区别？"),
    ("r25", "2026版的差旅餐补和以前相比有什么变化？"),
]

scheme = _build_real_scheme(5)
for cid, q in QUERIES:
    r = scheme.retrieve_full(q, 5)
    print(f"\n[{cid}] {q}")
    for h in r.hits:
        vol = (h.get("metadata") or {}).get("volume") or "?"
        print(f"   {vol[:22]:24} | {h.get('text', '')[:36]!r}")
