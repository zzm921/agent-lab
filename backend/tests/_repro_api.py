"""通过运行中的后端（127.0.0.1:8000）真实流式调用，复现 save_note 失败。"""
import json
import requests

url = "http://127.0.0.1:8000/api/stream"
payload = {
    "session_id": "debug-mcp",
    "message": "请用 save_note 帮我记一条便签，标题是「张三的笔记」，内容是「我叫张三」",
    "mode": "react",
    "enabled_capabilities": ["mcp-notes:save_note", "mcp-notes:list_notes"],
    "prompt_strategy": "standard",
    "approval_policy": "never",
}

print("== 发送请求 ==")
try:
    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        print("HTTP", resp.status_code)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            try:
                ev = json.loads(data)
            except Exception:
                continue
            t = ev.get("type")
            if t in ("tool_start", "tool_end", "error", "done", "message"):
                print("EVT:", json.dumps(ev, ensure_ascii=False)[:400])
except Exception as e:
    print("请求异常:", type(e).__name__, e)
