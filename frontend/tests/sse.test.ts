import { afterEach, describe, expect, it, vi } from 'vitest'
import { streamEvents } from '../src/services/sse'

function makeResponse(chunks: (string | Uint8Array)[]): Response {
  const parts = chunks.map((c) => (typeof c === 'string' ? new TextEncoder().encode(c) : c))
  const stream = new ReadableStream({
    start(controller) {
      parts.forEach((p) => controller.enqueue(p))
      controller.close()
    },
  })
  return new Response(stream, { status: 200 })
}

async function collect(url: string, body: unknown) {
  const events = []
  for await (const ev of streamEvents(url, body)) events.push(ev)
  return events
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('streamEvents（SSE 解析器）', () => {
  it('解析单条事件', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(makeResponse(['data: {"type":"meta","session_id":"s1","mode":"react","capabilities":[]}\n\n'])),
    )
    const events = await collect('/api/stream', {})
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'meta', session_id: 's1', mode: 'react' })
  })

  it('一个 chunk 内包含多个事件', async () => {
    const body = 'data: {"type":"thinking","delta":"a"}\n\ndata: {"type":"thinking","delta":"b"}\n\n'
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(makeResponse([body])))
    const events = await collect('/api/stream', {})
    expect(events.map((e) => e.type)).toEqual(['thinking', 'thinking'])
  })

  it('事件跨 chunk 分段仍可正确拼接', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(makeResponse(['data: {"type":"mess', 'age","delta":"hi"}\n\n'])),
    )
    const events = await collect('/api/stream', {})
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'message', delta: 'hi' })
  })

  it('兼容 \\r\\n 换行', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(makeResponse(['data: {"type":"done","summary":"ok","stats":{}}\r\n\r\n'])),
    )
    const events = await collect('/api/stream', {})
    expect(events[0]).toMatchObject({ type: 'done', summary: 'ok' })
  })

  it('多次 data: 行合并为一条事件', async () => {
    const body = 'data: {"type":"error"\ndata: ,"message":"bad"}\n\n'
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(makeResponse([body])))
    const events = await collect('/api/stream', {})
    expect(events[0]).toMatchObject({ type: 'error', message: 'bad' })
  })

  it('非 data 行（注释/空行）被忽略', async () => {
    const body = ': keepalive\n\ndata: {"type":"thinking","delta":"x"}\n\n'
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(makeResponse([body])))
    const events = await collect('/api/stream', {})
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'thinking' })
  })

  it('HTTP 错误抛出包含状态码的异常', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('no key', { status: 500 })))
    await expect(collect('/api/stream', {})).rejects.toThrow(/500/)
  })
})
