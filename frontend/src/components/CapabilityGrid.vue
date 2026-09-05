<script setup lang="ts">
import CapabilityCard from './CapabilityCard.vue'
import type { Capability } from '../types/agent'

defineProps<{
  caps: Capability[]
  enabledIds: string[]
  faults?: Record<string, string>
  faultTypes?: Record<string, string>
  loading?: boolean
  error?: string | null
  compact?: boolean
}>()
const emit = defineEmits<{ toggle: [id: string]; example: [cap: Capability]; fault: [id: string, mode: string] }>()
function onFault(id: string, mode: string) {
  emit('fault', id, mode)
}
</script>

<template>
  <div>
    <div v-if="loading" class="py-10 text-center text-sm text-slate-500">能力加载中…</div>
    <div v-else-if="error" class="py-10 text-center text-sm text-rose-400">
      能力加载失败：{{ error }}<span class="block text-xs text-slate-500">请确认后端运行于 http://localhost:8000</span>
    </div>
    <div v-else-if="!caps.length" class="py-10 text-center text-sm text-slate-500">暂无能力（/api/capabilities 未返回数据）</div>
    <div
      v-else
      class="grid gap-2"
      :class="compact ? 'grid-cols-2' : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3'"
    >
      <CapabilityCard
        v-for="cap in caps"
        :key="cap.id"
        :cap="cap"
        :enabled="enabledIds.includes(cap.id)"
        :fault="faults?.[cap.id]"
        :fault-types="faultTypes"
        :compact="compact"
        @toggle="emit('toggle', $event)"
        @example="emit('example', $event)"
        @fault="onFault"
      />
    </div>
  </div>
</template>
