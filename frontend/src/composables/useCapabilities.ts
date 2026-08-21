/** 能力池：加载能力列表、开关热插拔、示例一键填入。 */
import { computed, ref } from 'vue'
import type { Capability } from '../types/agent'

export function useCapabilities() {
  const caps = ref<Capability[]>([])
  const enabled = ref<string[]>(['calculator', 'time_now'])
  const loading = ref(false)
  const loadError = ref<string | null>(null)
  /** 故障注入配置：tool_id → 'error' | 'timeout'（验证熔断机制用） */
  const faults = ref<Record<string, string>>({})
  /** 示例提示：{cap, nonce}，HomeView watch 后填入输入框并展示提示条 */
  const exampleHint = ref<{ cap: Capability; nonce: number } | null>(null)

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
    await loadFaults()
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

  return {
    caps,
    enabled,
    loading,
    loadError,
    faults,
    exampleHint,
    enabledCapabilities,
    availableCount,
    load,
    setFault,
    faultMode,
    toggle,
    isEnabled,
    ensureEnabled,
    applyExample,
    clearHint,
  }
}
