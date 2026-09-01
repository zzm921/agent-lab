"""对比 v2/v3 评测报告中 challenge 用例的逐题指标（诊断定向补召回有效性）。"""
import json

v2 = json.load(open("eval/reports/real_full_v2_k5.json", encoding="utf-8"))
v3 = json.load(open("eval/reports/real_full_v3_k5.json", encoding="utf-8"))
r2 = {(x["id"], x["branch"]): x for x in v2["cases"]}
r3 = {(x["id"], x["branch"]): x for x in v3["cases"]}

print("id         v2 recall/mrr   v3 recall/mrr")
for k in sorted(r3):
    if r3[k].get("difficulty") != "challenge":
        continue
    a, b = r2.get(k, {}), r3[k]
    print(
        f"{k[0]:10} {a.get('recall')}/{a.get('mrr')}        "
        f"{b.get('recall')}/{b.get('mrr')}    query={b.get('query', '')[:24]}"
    )
