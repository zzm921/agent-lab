"""临时诊断：agentic 单条真实查询（张三和李雪的部门有多少人，领导是谁），打印完整 trace。"""
import sys

sys.path.insert(0, ".")

from eval.real_full import _build_real_scheme  # noqa: E402

QUERY = "张三和李雪的部门有多少人，领导是谁"


def main() -> int:
    scheme = _build_real_scheme(5, "agentic")
    r = scheme.retrieve_full(QUERY, 5)
    answ = r.answerability or {}
    print(f"\n===== 查询：{QUERY} =====")
    print(f"answerable={answ.get('answerable')}  recommendation={answ.get('recommendation')}")
    print(f"missing_facts={answ.get('missing_facts')}")
    if r.trace is not None:
        t = r.trace
        print(
            f"agent: {t['total_events']} 事件（工具执行 {t['total_tool_exec']}）"
            f"工具={t['tool_calls']} 纠错={t['corrections']} "
            f"token={t['tokens']['prompt']}+{t['tokens']['completion']} "
            f"role_calls={t['role_llm_calls']}"
        )
        for s in t["steps"]:
            q = (s["params"] or {}).get("query") or ""
            v = (s["params"] or {}).get("volume") or ""
            line = f"  #{s['seq']} [{s['role']}/{s['action']}] 「{q}」→ {s['hits']} 条"
            if v:
                line += f" 卷={v}"
            if s.get("note"):
                line += f"（{s['note']}）"
            if s.get("thought"):
                th = s["thought"].replace("\n", " ")[:120]
                line += f"\n        thought: {th}"
            print(line)
    print("\n--- 最终命中 ---")
    for h in r.hits[:8]:
        vol = (h.get("metadata") or {}).get("volume") or "?"
        print(f"  {vol[:22]:24} | {h.get('text', '')[:70]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
