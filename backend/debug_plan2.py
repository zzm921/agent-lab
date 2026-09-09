"""临时调试2（跑完删除）。"""
import json

text = '{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}, {"desc": "步骤二", "deps": ["t1"]}'
print("len", len(text), "last5", repr(text[-5:]))
for suffix in ("]", "}]"):
    s = text + suffix
    try:
        p = json.loads(s)
        print("OK", repr(suffix), json.dumps(p, ensure_ascii=False)[:60])
    except Exception as e:
        print("FAIL", repr(suffix), type(e).__name__, e)
