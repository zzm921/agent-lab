/** 多会话管理：会话列表 / 新建 / 切换 / 重命名 / 删除。
 *
 * 会话归属后端统一分配 session_id 并持久化（checkpointer 文件落盘 + 元数据），
 * 前端只负责「当前会话」的记忆（localStorage）与列表渲染；
 * 切换会话后由 useChatStream 以该 session_id 继续对话（后端按 id 恢复历史）。
 */
import { ref } from 'vue'
import { getClientId } from '../services/sse'
import type { AgentEvent } from '../types/agent'

/** 会话元数据（与后端 sessions.jsonl 字段一致） */
export interface SessionMeta {
  session_id: string
  title: string
  client_key: string
  created_at: number
  last_active: number
  message_count: number
}

const SESSION_KEY = 'agent-lab.current-session'

export function useSessions() {
  const sessions = ref<SessionMeta[]>([])
  const currentId = ref(localStorage.getItem(SESSION_KEY) ?? '')
  const sessionsLoading = ref(false)
  const sessionsError = ref('')

  /** 记住当前会话（刷新/重启后恢复） */
  function persistCurrent() {
    if (currentId.value) localStorage.setItem(SESSION_KEY, currentId.value)
    else localStorage.removeItem(SESSION_KEY)
  }

  async function load() {
    sessionsLoading.value = true
    sessionsError.value = ''
    try {
      const resp = await fetch('/api/sessions', { headers: clientHeaders() })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail ?? '加载会话列表失败')
      sessions.value = data.sessions ?? []
    } catch (e) {
      sessionsError.value = e instanceof Error ? e.message : String(e)
    } finally {
      sessionsLoading.value = false
    }
  }

  /** 新建会话：后端分配 id；自动切换为当前会话 */
  async function create(): Promise<string> {
    const resp = await fetch('/api/sessions', { method: 'POST', headers: clientHeaders() })
    const data = await resp.json()
    if (!resp.ok) throw new Error(data.detail ?? '新建会话失败')
    currentId.value = data.session_id
    persistCurrent()
    await load()
    return data.session_id
  }

  /** 切换会话：更新当前 id 并记忆 */
  function switchTo(sessionId: string) {
    currentId.value = sessionId
    persistCurrent()
  }

  /** 重命名会话 */
  async function rename(sessionId: string, title: string): Promise<void> {
    const resp = await fetch(`/api/sessions/${sessionId}`, {
      method: 'PATCH',
      headers: { ...clientHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    })
    const data = await resp.json()
    if (!resp.ok) throw new Error(data.detail ?? '重命名失败')
    await load()
  }

  /** 删除会话：后端联动清理 checkpoint 与记忆；若删的是当前会话则新建一个空会话 */
  async function remove(sessionId: string): Promise<void> {
    const resp = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE', headers: clientHeaders() })
    const data = await resp.json()
    if (!resp.ok) throw new Error(data.detail ?? '删除会话失败')
    if (currentId.value === sessionId) {
      currentId.value = ''
      localStorage.removeItem(SESSION_KEY)
      await create() // 删除当前会话后立即新建空会话，保持可聊状态
    } else {
      await load()
    }
  }

  /** 拉取某会话已落盘的历史事件流（切换会话时由 useChatStream.replay 重放） */
  async function loadEvents(sessionId: string): Promise<AgentEvent[]> {
    try {
      const resp = await fetch(`/api/sessions/${sessionId}/events`, { headers: clientHeaders() })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail ?? '加载会话历史失败')
      return (data.events as AgentEvent[]) ?? []
    } catch {
      return [] // 历史读取失败不阻断切换：按空会话继续
    }
  }

  return {
    sessions,
    currentId,
    sessionsLoading,
    sessionsError,
    load,
    create,
    switchTo,
    rename,
    remove,
    loadEvents,
  }
}

function clientHeaders(): Record<string, string> {
  const headers: Record<string, string> = {}
  const cid = getClientId() // 与 SSE 流同一设备指纹，保证会话隔离与对话隔离一致
  if (cid) headers['X-Client-Id'] = cid
  return headers
}
