"""临时诊断：验证流式/非流式 LLM 调用末块是否携带 usage_metadata（token 记账依据）。"""
import asyncio
from dashscope import Generation
from app.config import settings
from app.llm.service import LLMService, _usage_tokens

svc = LLMService()
model = svc.get("chat")
print("model:", type(model).__name__, getattr(model, "model_name", ""))


async def raw_stream_test():
    """直接看 dashscope 流式原始块：每块 choices 数、finish_reason、usage。"""
    from app.llm.dashscope_chat import DashScopeChatModel

    ds = DashScopeChatModel(
        model_name="qwen3.5-flash",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "",
        temperature=0.3,
        enable_thinking=True,
    )
    payload = ds._payload(
        [{"role": "user", "content": "你好，用一句话回复"}],
        stream=True,
    )
    it = iter(Generation.call(**payload))
    idx = 0
    while True:
        try:
            resp = next(it)
        except StopIteration:
            break
        status = getattr(resp, "status_code", None)
        output = getattr(resp, "output", None)
        choices = output.get("choices") if isinstance(output, dict) else []
        fin = choices[0].get("finish_reason") if choices else None
        msg = (choices[0].get("message") or {}) if choices else {}
        usage = getattr(resp, "usage", None)
        content_len = len(_ct(msg.get("content")) or "")
        print(f"[{idx}] status={status} choices={len(choices)} finish={fin} usage={usage} content_len={content_len}")
        idx += 1
    print("total chunks:", idx)


def _ct(x):
    return x if isinstance(x, str) else ""


async def stream_test():
    from app.telemetry.sink import ACTIVE_SINK
    from app.telemetry.store import RunStore
    import tempfile, os

    tmp = tempfile.mkdtemp()
    store = RunStore(os.path.join(tmp, "telemetry"), ttl_days=7, max_runs=100)
    sink = store.new_run("s1", "cid:x", {"message": "diag"})
    token = ACTIVE_SINK.set(sink)
    try:
        last = None
        async for c in model.astream([{"role": "user", "content": "你好，用一句话回复"}]):
            last = c
        print("astream last type:", type(last).__name__)
        print("astream last usage_metadata:", getattr(last, "usage_metadata", None))
        print("astream _usage_tokens:", _usage_tokens(last))
        meta = sink.close()
        print("recorded stats.tokens:", meta["stats"]["tokens"], "llm_calls:", meta["stats"]["llm_calls"])
    finally:
        ACTIVE_SINK.reset(token)


async def nonstream_test():
    import time

    t0 = time.perf_counter()
    result = await model._agenerate([{"role": "user", "content": "你好，用一句话回复"}])
    msg = result.generations[0].message
    print("agenerate usage_metadata:", getattr(msg, "usage_metadata", None))
    print("agenerate llm_output:", result.llm_output)
    print("agenerate _usage_tokens:", _usage_tokens(result), f"({(time.perf_counter()-t0)*1000:.0f}ms)")


asyncio.run(raw_stream_test())
def raw_ds_stream():
    from app.llm.dashscope_chat import DashScopeChatModel

    ds = DashScopeChatModel(
        model_name="qwen3.5-flash",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "",
        temperature=0.3,
        enable_thinking=True,
    )
    for i, gen in enumerate(ds._stream([{"role": "user", "content": "你好，用一句话回复"}])):
        m = gen.message
        print(
            f"[{i}] content_len={len(getattr(m, 'content', '') or '')} "
            f"usage_metadata={getattr(m, 'usage_metadata', None)} "
            f"response_metadata={getattr(m, 'response_metadata', None)}"
        )


raw_ds_stream()

import asyncio


async def inner_stream_probe():
    """直接探 _inner.astream（bind 后）与 model.astream（公开）的块结构与末块 usage。"""
    print("_inner type:", type(model._inner).__name__)
    print("_inner._llm_type:", getattr(getattr(model._inner, "_llm_type", None), "__name__", getattr(model._inner, "_llm_type", None)))
    print("=== _inner.astream ===")
    last = None
    n = 0
    async for chunk in model._inner.astream([{"role": "user", "content": "你好，用一句话回复"}]):
        n += 1
        last = chunk
    print("chunks:", n, "last type:", type(last).__name__)
    print("last usage_metadata:", getattr(last, "usage_metadata", None))

    print("=== model.astream (public) ===")
    last = None
    n = 0
    async for chunk in model.astream([{"role": "user", "content": "你好，用一句话回复"}]):
        n += 1
        last = chunk
    print("chunks:", n, "last type:", type(last).__name__)
    print("last usage_metadata:", getattr(last, "usage_metadata", None))
    print("_should_stream:", model._should_stream(async_api=True))


asyncio.run(inner_stream_probe())
asyncio.run(nonstream_test())
