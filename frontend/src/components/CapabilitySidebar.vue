<script setup lang="ts">
import CapabilityGrid from './CapabilityGrid.vue'
import ModeSelector from './ModeSelector.vue'
import PromptStrategyPicker from './PromptStrategyPicker.vue'
import type { ApprovalPolicy, Capability, ModeId, PromptStrategy } from '../types/agent'

defineProps<{
  caps: Capability[]
  enabledIds: string[]
  faults?: Record<string, string>
  faultTypes?: Record<string, string>
  loading?: boolean
  error?: string | null
  mode: ModeId
  strategy: PromptStrategy
  policy: ApprovalPolicy
  open?: boolean
}>()

const emit = defineEmits<{
  toggle: [id: string]
  example: [cap: Capability]
  fault: [id: string, mode: string]
  'update:mode': [v: ModeId]
  'update:strategy': [v: PromptStrategy]
  'update:policy': [v: ApprovalPolicy]
  close: []
}>()

const POLICIES: ApprovalPolicy[] = ['always', 'never']

function onFault(id: string, mode: string) {
  emit('fault', id, mode)
}
</script>

<template>
  <aside
    class="fixed inset-y-0 left-0 z-30 flex w-80 flex-col border-r border-slate-800 bg-slate-950 md:static md:inset-auto"
    :class="open ? 'translate-x-0' : '-translate-x-full md:translate-x-0'"
  >
    <div class="flex items-center justify-between border-b border-slate-800 px-4 py-3">
      <div>
        <h2 class="text-sm font-semibold text-white">能力选配</h2>
        <p class="text-[11px] text-slate-500">已启用 {{ enabledIds.length }} 项能力</p>
      </div>
      <button
        type="button"
        class="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white md:hidden"
        @click="emit('close')"
      >
        <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <div class="flex-1 space-y-6 overflow-y-auto px-4 py-4">
      <section class="space-y-3">
        <h3 class="text-xs font-semibold text-slate-300">推理模式</h3>
        <ModeSelector :model-value="mode" @update:model-value="emit('update:mode', $event)" />
      </section>

      <section class="space-y-3 border-t border-slate-800 pt-4">
        <h3 class="text-xs font-semibold text-slate-300">审批策略</h3>
        <div class="flex overflow-hidden rounded-lg border border-slate-700">
          <button
            v-for="p in POLICIES"
            :key="p"
            type="button"
            class="flex-1 px-2.5 py-1.5 text-xs transition"
            :class="p === policy ? 'bg-indigo-500 text-white' : 'bg-slate-800 text-slate-400 hover:text-white'"
            :title="p === 'always' ? '工具执行前弹出审批' : '工具自动执行不审批'"
            @click="emit('update:policy', p)"
          >
            {{ p === 'always' ? '执行前审批' : '自动执行' }}
          </button>
        </div>
      </section>

      <section class="space-y-3 border-t border-slate-800 pt-4">
        <PromptStrategyPicker :model-value="strategy" @update:model-value="emit('update:strategy', $event)" />
      </section>

      <section class="border-t border-slate-800 pt-4">
        <div class="mb-2 flex items-center justify-between">
          <h3 class="text-xs font-semibold text-slate-300">能力与故障注入</h3>
          <span class="text-[10px] text-slate-500" title="瞬时错误→工具层直接重试；参数/业务错误→交给模型思考后重试">瞬时重试 / 交模型</span>
        </div>
        <CapabilityGrid
          :caps="caps"
          :enabled-ids="enabledIds"
          :faults="faults"
          :fault-types="faultTypes"
          :loading="loading"
          :error="error"
          compact
          @toggle="emit('toggle', $event)"
          @example="emit('example', $event)"
          @fault="onFault"
        />
      </section>
    </div>
  </aside>
</template>
