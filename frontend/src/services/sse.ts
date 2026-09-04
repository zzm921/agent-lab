/** SSE 流式解析器：以 fetch + ReadableStream 读取 POST 响应，解析为结构化事件流。
 * 兼容 \r\n 与 \n 换行、跨 chunk 分片、单事件多行 data:。
 */
import type { AgentEvent } from '../types/agent'

/** 客户端设备指纹：localStorage 持久化，用于后端按「一台电脑」隔离常驻记忆与统计每日对话配额 */
const CLIENT_ID_KEY = 'agent_lab_client_id'

export function getClientId(): string {
  try {
    let id = localStorage.getItem(CLIENT_ID_KEY)
    if (!id) {
      id =
        typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
      localStorage.setItem(CLIENT_ID_KEY, id)
    }
    return id
  } catch {
    return ''
  }
}

/** 获取当前客户端的每日对话配额使用情况（GET /api/quota，带设备指纹头） */
export async function fetchQuota(): Promise<{ enabled: boolean; limit: number; remaining: number }> {
  const clientId = getClientId()
  const resp = await fetch('/api/quota', {
    method: 'GET',
    headers: clientId ? { 'X-Client-Id': clientId } : {},
  })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

/** 找到第一个事件块结束位置（空行 \n\n 或 \r\n\r\n），无则返回 -1 */
function findBlockEnd(buf: string): number {
  const a = buf.indexOf('\n\n')
  const b = buf.indexOf('\r\n\r\n')
  if (a === -1) return b
  if (b === -1) return a
  return Math.min(a, b)
}

/** 事件块结束符长度 */
function blockEndLength(buf: string, end: number): number {
  return buf.startsWith('\r\n\r\n', end) ? 4 : 2
}

/** 提取一个事件块中的所有 data: 行内容 */
function parseDataLines(block: string): string[] {
  const lines: string[] = []
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith('data:')) {
      lines.push(line.slice(5).replace(/^ /, ''))
    }
  }
  return lines
}

function isAgentEvent(value: unknown): value is AgentEvent {
  return (
    value !== null &&
    typeof value === 'object' &&
    typeof (value as Record<string, unknown>).type === 'string'
  )
}

/** 发起 POST 并逐条 yield 解析后的 SSE 事件；HTTP 非 2xx 时抛出异常。 */
export async function* streamEvents(
  url: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<AgentEvent> {
  const clientId = getClientId()
  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(clientId ? { 'X-Client-Id': clientId } : {}),
    },
    body: JSON.stringify(body),
    signal,
  })
  if (!resp.ok) {
    let detail = ''
    try {
      const text = await resp.text()
      const parsed = JSON.parse(text)
      if (parsed && typeof parsed.detail === 'string') detail = parsed.detail
      else detail = text.slice(0, 200)
    } catch {
      /* 忽略读取失败 */
    }
    throw new Error(`HTTP ${resp.status}${detail ? ` · ${detail}` : ''}`)
  }
  if (!resp.body) throw new Error('响应没有可读的流')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let end: number
      while ((end = findBlockEnd(buffer)) !== -1) {
        const block = buffer.slice(0, end)
        buffer = buffer.slice(end + blockEndLength(buffer, end))
        const datas = parseDataLines(block)
        if (!datas.length) continue
        const joined = datas.join('\n')
        let parsed: unknown
        try {
          parsed = JSON.parse(joined)
        } catch {
          // 多行 data 无法整体解析时逐行尝试
          for (const d of datas) {
            try {
              const ev: unknown = JSON.parse(d)
              if (isAgentEvent(ev)) yield ev
            } catch {
              /* 忽略无法解析的行 */
            }
          }
          continue
        }
        if (isAgentEvent(parsed)) yield parsed
      }
    }
  } finally {
    reader.releaseLock()
  }
}
