"""临时诊断：hook Verifier/Corrector，记录每轮输入输出，定位"反复确认张三缺失但最终正确输出"根因。"""
import sys

sys.path.insert(0, ".")

from eval.real_full import _build_real_scheme  # noqa: E402

QUERY = "张三和李雪的部门有多少人，部门领导是谁"


def _snip(text: str, n: int = 60) -> str:
    t = (text or "").replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


def main() -> int:
    scheme = _build_real_scheme(5, "agentic")
    orch = scheme.orchestrator  # RetrieveResult 构建时持有 orchestrator
    verifier = orch.verifier
    corrector = orch.corrector
    grader = orch.grader

    orig_verify = verifier.run
    orig_correct = corrector.run
    orig_grade = grader.run

    def hook_verify(query, facts, hits, state, use_llm=True, confirmed_facts=None):
        print("\n[VERIFY] =========================================")
        print(f"  facts={facts}")
        print(f"  confirmed_facts={confirmed_facts}")
        print(f"  evidence({len(hits)}):")
        for i, h in enumerate(hits):
            vol = (h.get("metadata") or {}).get("volume") or "?"
            print(f"    [{i}] {vol} | {_snip(h.get('text'), 90)}")
        out = orig_verify(query, facts, hits, state, use_llm, confirmed_facts)
        print(f"  -> answerable={out.answerable} missing={out.missing_facts}")
        print(f"  -> thought={_snip(out.thought, 150)}")
        return out

    def hook_correct(query, missing_facts, executed, catalog, state, use_llm=True, prior_hits=None):
        print("\n[CORRECT] ========================================")
        print(f"  missing_facts={missing_facts}")
        print(f"  prior_hits({len(prior_hits) if prior_hits else 0}):")
        for i, h in enumerate(prior_hits or []):
            vol = (h.get("metadata") or {}).get("volume") or "?"
            print(f"    [{i}] {vol} | {_snip(h.get('text'), 80)}")
        out = orig_correct(query, missing_facts, executed, catalog, state, use_llm, prior_hits)
        for c in out.calls[:6]:
            print(f"  -> call: {c.action} 「{c.query}」 vol={c.volume}")
        return out

    def hook_grade(query, hits, state, use_llm=True, prior_hits=None):
        print("\n[GRADE] =========================================")
        print(f"  candidates({len(hits)}):")
        for i, h in enumerate(hits):
            vol = (h.get("metadata") or {}).get("volume") or "?"
            txt = h.get("text") or ""
            print(f"    [{i}] {vol} len={len(txt)} | {_snip(txt, 80)}")
            if "关键人员" in txt or "权益明细" in txt:
                print(f"        FULL: {txt[:600]}")
        out = orig_grade(query, hits, state, use_llm, prior_hits)
        print(f"  -> keep={out.keep} missing={out.missing_facts}")
        print(f"  -> thought={_snip(out.thought, 150)}")
        return out

    verifier.run = hook_verify
    corrector.run = hook_correct
    grader.run = hook_grade
    try:
        r = scheme.retrieve_full(QUERY, 5)
    finally:
        verifier.run = orig_verify
        corrector.run = orig_correct
        grader.run = orig_grade

    answ = r.answerability or {}
    print("\n===== 结果 =====")
    print(f"answerable={answ.get('answerable')} missing={answ.get('missing_facts')}")
    if r.trace:
        t = r.trace
        print(f"工具={t['tool_calls']} 纠错={t['corrections']} token={t['tokens']}")
        for s in t["steps"]:
            print(f"  #{s['seq']} [{s['role']}/{s['action']}] note={s.get('note') or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
