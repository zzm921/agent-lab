"""临时调试3（跑完删除）。"""
import json

text = '{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}, {"desc": "步骤二", "deps": ["t1"]}'
print(repr(text))
s = text + "}]"
print("FULL:", repr(s))
print("first150:", s[:150])
try:
    print(json.loads(s))
except Exception as e:
    print("ERR", e)
