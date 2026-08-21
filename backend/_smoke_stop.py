"""临时冒烟脚本：验证 /api/stop 能立即停止正在执行的后端任务，节省 token。"""
import asyncio
import json
import time

import httpx

BASE = "http://127.0.0.1:8001"


async def main():
    session_id = "smoke_stop_" + str(int(time.time() * 1000))
    task = "连续帮我计算：1+1、2+2、3+3、4+4、5+5、6+6、7+7、8+8、9+9、10+10、11+11、12+12"

    async with httpx.AsyncClient(timeout=None) as client:
        # 1) 启动流式执行（后台消费）
        async def consume():
            events = []
            async with client.stream(
                "POST",
                f"{BASE}/api/stream",
                json={
                    "session_id": session_id,
                    "message": task,
                    "mode": "react",
                    "enabled_capabilities": ["calculator", "time_now"],
                    "prompt_strategy": "standard",
                    "approval_policy": "never",
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[5:].strip()))
            return events

        consumer = asyncio.create_task(consume())

        # 2) 等待一小段，确保已进入执行（出现工具事件或流式输出）
        await asyncio.sleep(3.5)
        tool_started = False
        # 3) 触发后端停止
        start = time.time()
        r = await client.post(f"{BASE}/api/stop", json={"session_id": session_id})
        print(f"POST /api/stop -> {r.status_code} {r.json()}")

        # 4) 等待消费完成（连接关闭）
        await asyncio.wait_for(consumer, timeout=10)

    events = consumer.result()
    elapsed = time.time() - start
    print(f"停止后约 {elapsed:.2f}s 连接关闭")
    print(f"共收到 {len(events)} 个事件")
    types = [e.get("type") for e in events]
    print("事件类型序列:", types)
    done = [e for e in events if e.get("type") == "done"]
    errs = [e for e in events if e.get("type") == "error"]
    msgs = "".join(e.get("delta", "") for e in events if e.get("type") == "message")
    print("done 事件:", done)
    print("error 事件:", errs)
    print("最终答案是否输出:", bool(msgs.strip()))
    print("末尾 200 字符:", msgs[-200:])

    # 判定：连接被关闭（后端被取消），且没有产出最终答案
    ok = done or errs or (len(events) > 0)
    assert not msgs.strip(), "停止失败：仍输出了完整最终答案"
    print("\nSMOKE_RESULT: OK" if ok else "\nSMOKE_RESULT: FAIL")


if __name__ == "__main__":
    asyncio.run(main())
