"""金标测试集标注：按 evidence 子串在本地分块产物中定位 relevant chunk_id。

- 匹配口径：normalize（去除全部空白字符）后包含判定，容忍表格行/子块切分的空白差异；
- 每题 relevant = 全部 evidence 命中 chunk_id 的并集（单 evidence 命中数上限 8，防超宽串刷量）；
- 校验：非库外题 relevant 必须非空、每个 evidence 至少命中 1 块，否则打印失败明细并以非零退出。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
CHUNKS = BASE / "real_chunks.json"
CASES = BASE / "real_eval_set.jsonl"
MAX_HITS_PER_EVIDENCE = 8


def _norm(s: str) -> str:
    return "".join(s.split())


def main() -> int:
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    norm_texts = {c["chunk_id"]: _norm(c["text"]) for c in chunks}
    cases = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]

    failures: list[str] = []
    for case in cases:
        relevant: list[str] = []
        for ev in case.get("evidence", []):
            nev = _norm(ev)
            hits = [cid for cid, t in norm_texts.items() if nev in t]
            if not hits:
                failures.append(f"{case['id']}: evidence 未命中 -> {ev[:40]}...")
                continue
            relevant.extend(hits[:MAX_HITS_PER_EVIDENCE])
        case["relevant"] = sorted(dict.fromkeys(relevant))
        if case["answerable"] and not case["relevant"]:
            failures.append(f"{case['id']}: 无 relevant chunk")

    with CASES.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    annotated = sum(1 for c in cases if c["relevant"])
    print(f"标注完成: {annotated}/{len(cases)} 题有 relevant；chunks={len(chunks)}")
    for c in cases:
        print(f"  {c['id']}: {len(c['relevant'])} 块")
    if failures:
        print("失败明细:")
        for msg in failures:
            print(" ", msg)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
