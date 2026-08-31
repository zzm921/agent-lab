"""临时：对比 top_k=3 与 top_k=5 报告的 per-case 差异。"""
import json

base = json.load(open("eval/reports/real_full.json", encoding="utf-8"))
top5 = json.load(open("eval/reports/real_full_top5.json", encoding="utf-8"))
b = {c["id"]: c for c in base["cases"]}
t = {c["id"]: c for c in top5["cases"]}

print("== per-case recall 变化 ==")
for cid in b:
    rb, rt = b[cid].get("recall"), t[cid].get("recall")
    if rb != rt:
        print(
            f"  {cid} [{b[cid]['difficulty']}] {b[cid]['query'][:24]}"
            f"  k3: recall={rb} ev={b[cid].get('covered_evidence')}/{len(b[cid].get('evidence', []))}"
            f"  k5: recall={rt} ev={t[cid].get('covered_evidence')}/{len(t[cid].get('evidence', []))}"
            f"  行为 k3={b[cid]['behavior']} k5={t[cid]['behavior']}"
        )
