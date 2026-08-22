"""临时复现：测试 /api/stream 在 approval_policy=always 时是否发出 approval_request。"""
import json
import sys
import urllib.request

payload = {
    "session_id": "approval-debug",
    "message": "帮我计算 1+1 的结果",
    "mode": "react",
    "enabled_capabilities": ["calculator"],
    "prompt_strategy": "standard",
    "approval_policy": "always",
}
req = urllib.request.Request(
    "http://localhost:8000/api/stream",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

seen = []
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            for part in line.split("\n"):
                part = part.strip()
                if part.startswith("data:"):
                    try:
                        ev = json.loads(part[5:])
                    except Exception:
                        continue
                    t = ev.get("type")
                    seen.append(t)
                    if t == "approval_request":
                        print(">>> approval_request:", json.dumps(ev, ensure_ascii=False)[:300])
                    elif t in ("done", "error", "tool_start", "tool_end"):
                        print(">>>", t, json.dumps(ev, ensure_ascii=False)[:200])
                    elif t in ("thinking", "message"):
                        print(">>>", t, ev.get("delta", "")[:60])
except Exception as exc:
    print("REQ ERROR:", exc)

print("=== event types seen ===")
from collections import Counter

print(Counter(seen))
