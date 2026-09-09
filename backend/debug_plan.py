"""临时调试 plan_execute 截断解析（跑完删除）。"""
import json

from app.agents.modes.plan_execute import _extract_json, _parse_plan_json, _parse_plan_output

text = '{"direct": false, "tasks": [{"desc": "步骤一", "deps": []}, {"desc": "步骤二", "deps": ["t1"]}'
print("raw len", len(text), "last3", repr(text[-3:]))
print("extract:", _extract_json(text))
payload = _extract_json(text)
if payload:
    print("plan_json:", _parse_plan_json(payload))
    print("payload:", json.dumps(payload, ensure_ascii=False))
direct, items = _parse_plan_output(text)
print("direct:", direct, "items:", items)
