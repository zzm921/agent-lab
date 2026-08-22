"""临时验证：真实后端 /api/stream 链路调用 mcp-notes 工具（验证后删除）。"""
import json
import urllib.request

body = {
    "session_id": "verify-mcp",
    "message": "请使用 save_note 保存一条便签，标题为「验证」，内容为「MCP链路打通」；然后再调用 list_notes 列出便签。",
    "mode": "react",
    "enabled_capabilities": ["mcp-notes:save_note", "mcp-notes:list_notes"],
    "prompt_strategy": "standard",
    "approval_policy": "never",
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/stream",
    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    for line in resp:
        line = line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:])
        t = ev.get("type")
        if t in ("tool_start", "tool_end", "done", "error"):
            print("EVENT", json.dumps(ev, ensure_ascii=False)[:400])
        if t in ("done", "error"):
            break
