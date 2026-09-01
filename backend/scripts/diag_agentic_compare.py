"""挑战题对比诊断：modular vs agentic 在路由易误判题（r21/r24/r25）上的检索行为对照。

用法（backend 目录）：
  python scripts/diag_agentic_compare.py                 # 默认 r21/r24/r25、top_k=5
  python scripts/diag_agentic_compare.py --ids r21 r24   # 指定用例
  python scripts/diag_agentic_compare.py --top-k 3

对照维度：命中卷分布、evidence 覆盖（金标 real_eval_set.jsonl）、agent 轨迹
（事件/工具/纠错/token）——验证 agentic 的「首轮偏差可当场换路」是否改善
modular 定向路判错即整路白跑的问题（需 LLM_API_KEY + EMBEDDING_API_KEY + 已建库）。
"""
import argparse
import sys

sys.path.insert(0, ".")

from eval.real_full import _build_real_scheme, _load_cases  # noqa: E402

DEFAULT_IDS = ["r21", "r24", "r25"]


def _norm(s: str) -> str:
    return "".join(s.split())


def _volume_dist(hits: list[dict]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for h in hits:
        vol = (h.get("metadata") or {}).get("volume") or "?"
        counter[vol] = counter.get(vol, 0) + 1
    return counter


def _covered(evidence: list[str], hits: list[dict]) -> list[str]:
    """evidence 覆盖口径（norm 包含判定，与 real_full 一致）。"""
    norm_hits = [_norm(h.get("text", "")) for h in hits]
    return [ev for ev in evidence if _norm(ev) and any(_norm(ev) in t for t in norm_hits)]


def main() -> int:
    parser = argparse.ArgumentParser(description="modular vs agentic 挑战题对比诊断")
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS, help="评测用例 id（real_eval_set.jsonl）")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = {c["id"]: c for c in _load_cases()}
    missing = [cid for cid in args.ids if cid not in cases]
    if missing:
        print(f"用例不存在：{missing}（可选 {sorted(cases)}）")
        return 1

    schemes = [("modular", _build_real_scheme(args.top_k, "modular")),
               ("agentic", _build_real_scheme(args.top_k, "agentic"))]
    for cid in args.ids:
        case = cases[cid]
        ev_total = len(case.get("evidence", []))
        print(f"\n===== [{cid}] {case['query']} =====")
        for name, scheme in schemes:
            r = scheme.retrieve_full(case["query"], args.top_k)
            dist = _volume_dist(r.hits)
            covered = _covered(case.get("evidence", []), r.hits)
            print(f"\n[{name}] 命中 {len(r.hits)} 条 | evidence 覆盖 {len(covered)}/{ev_total}")
            print("   卷分布: " + "、".join(f"{v}({c})" for v, c in sorted(dist.items(), key=lambda kv: -kv[1])))
            if r.trace is not None:
                t = r.trace
                print(
                    f"   agent: {t['total_events']} 事件（工具执行 {t['total_tool_exec']}）"
                    f"工具={t['tool_calls']} 纠错={t['corrections']} "
                    f"token={t['tokens']['prompt']}+{t['tokens']['completion']}"
                )
                for s in t["steps"]:
                    line = f"     - #{s['seq']} [{s['role']}/{s['action']}] 「{(s['params'] or {}).get('query') or ''}」→ {s['hits']} 条"
                    if (s["params"] or {}).get("volume"):
                        line += f" 卷={(s['params'])['volume']}"
                    if s.get("note"):
                        line += f"（{s['note']}）"
                    print(line)
            for h in r.hits[:5]:
                vol = (h.get("metadata") or {}).get("volume") or "?"
                print(f"   {vol[:22]:24} | {h.get('text', '')[:36]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
