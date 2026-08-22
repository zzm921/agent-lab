/** 能力池：加载能力列表、开关热插拔、示例一键填入。 */
import { computed, ref } from 'vue'
import type { Capability } from '../types/agent'

export function useCapabilities() {
  const caps = ref<Capability[]>([])
  const enabled = ref<string[]>(['calculator', 'time_now'])
  const loading = ref(false)
  const loadError = ref<string | null>(null)
  /** 故障注入配置：tool_id → 注入类型（如 timeout/http_500/business/http_400） */
  const faults = ref<Record<string, string>>({})
  /** 故障注入类型目录：类型 → 重试分类（retryable=工具层直接重试，permanent=交给模型思考后重试） */
  const faultTypes = ref<Record<string, string>>({})
  /** MCP 服务开关：默认关闭，页面点选开启（后端 POST /api/mcp 连接并发现工具） */
  const mcpEnabled = ref(false)
  /** 示例提示：{cap, nonce}，HomeView watch 后填入输入框并展示提示条 */
  const exampleHint = ref<{ cap: Capability; nonce: number } | null>(null)

  async function loadFaultTypes() {
    try {
      const res = await fetch('/api/faults/types')
      if (!res.ok) return
      const data = await res.json()
      faultTypes.value = data.types ?? {}
    } catch {
      /* 后端未启动或未配置模型时忽略 */
    }
  }

  async function loadFaults() {
    try {
      const res = await fetch('/api/faults')
      if (!res.ok) return
      const data = await res.json()
      faults.value = data.faults ?? {}
    } catch {
      /* 后端未启动或未配置模型时忽略 */
    }
  }

  async function setFault(id: string, mode: string) {
    try {
      const res = await fetch('/api/fault', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tool: id, mode }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      faults.value = data.faults ?? faults.value
    } catch (e) {
      loadError.value = e instanceof Error ? e.message : String(e)
    }
  }

  function faultMode(id: string): string {
    return faults.value[id] ?? 'off'
  }

  /** 读取 MCP 开关状态（默认关闭，后端单例维护） */
  async function loadMcp() {
    try {
      const res = await fetch('/api/mcp')
      if (!res.ok) return
      const data = await res.json()
      mcpEnabled.value = Boolean(data.enabled)
    } catch {
      /* 后端未启动时忽略 */
    }
  }

  /** 页面点选开启/关闭 MCP：成功后重新拉能力列表（MCP 工具出现/消失） */
  async function setMcpEnabled(v: boolean) {
    try {
      const res = await fetch('/api/mcp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: v }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      mcpEnabled.value = v
      await load()
    } catch (e) {
      loadError.value = e instanceof Error ? e.message : String(e)
    }
  }

  async function load() {
    loading.value = true
    loadError.value = null
    try {
      const res = await fetch('/api/capabilities')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      caps.value = Array.isArray(data.capabilities) ? data.capabilities : []
      // 清理已不存在的启用项
      const ids = new Set(caps.value.map((c) => c.id))
      enabled.value = enabled.value.filter((id) => ids.has(id))
    } catch (e) {
      loadError.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
    await loadFaultTypes()
    await loadFaults()
    await loadMcp()
  }

  function toggle(id: string) {
    if (enabled.value.includes(id)) {
      enabled.value = enabled.value.filter((x) => x !== id)
    } else {
      enabled.value = [...enabled.value, id]
    }
  }

  function isEnabled(id: string) {
    return enabled.value.includes(id)
  }

  function ensureEnabled(id: string) {
    if (!enabled.value.includes(id)) enabled.value = [...enabled.value, id]
  }

  /** 示例按钮：可用则自动启用该能力并产出示例提示 */
  function applyExample(cap: Capability) {
    if (cap.availability !== 'available') return false
    ensureEnabled(cap.id)
    exampleHint.value = { cap, nonce: Date.now() }
    return true
  }

  function clearHint() {
    exampleHint.value = null
  }

  const enabledCapabilities = computed(() => caps.value.filter((c) => enabled.value.includes(c.id)))
  const availableCount = computed(() => caps.value.filter((c) => c.availability === 'available').length)
  /** 内置能力 / MCP 能力分组：直观呈现有无 MCP 的差异 */
  const builtinCaps = computed(() => caps.value.filter((c) => c.source !== 'mcp'))
  const mcpCaps = computed(() => caps.value.filter((c) => c.source === 'mcp'))

  return {
    caps,
    enabled,
    loading,
    loadError,
    faults,
    faultTypes,
    mcpEnabled,
    exampleHint,
    enabledCapabilities,
    availableCount,
    builtinCaps,
    mcpCaps,
    load,
    loadMcp,
    setMcpEnabled,
    setFault,
    faultMode,
    toggle,
    isEnabled,
    ensureEnabled,
    applyExample,
    clearHint,
  }
}
