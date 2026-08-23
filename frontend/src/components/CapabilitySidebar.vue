<script setup lang="ts">
import { computed } from 'vue'
import CapabilityGrid from './CapabilityGrid.vue'
import ModeSelector from './ModeSelector.vue'
import PromptStrategyPicker from './PromptStrategyPicker.vue'
import RagSchemeSelector from './RagSchemeSelector.vue'
import type { ApprovalPolicy, Capability, ModeId, PromptStrategy, RagScheme, RagSchemeId } from '../types/agent'

const props = defineProps<{
  caps: Capability[]
  enabledIds: string[]
  faults?: Record<string, string>
  faultTypes?: Record<string, string>
  loading?: boolean
  error?: string | null
  mode: ModeId
  strategy: PromptStrategy
  policy: ApprovalPolicy
  ragScheme: RagSchemeId
  ragSchemes: RagScheme[]
  open?: boolean
  mcpEnabled?: boolean
  mcpCaps?: Capability[]
}>()

const emit = defineEmits<{
  toggle: [id: string]
  example: [cap: Capability]
  fault: [id: string, mode: string]
  'toggle-mcp': [v: boolean]
  'update:mode': [v: ModeId]
  'update:strategy': [v: PromptStrategy]
  'update:policy': [v: ApprovalPolicy]
  'update:rag-scheme': [v: RagSchemeId]
  close: []
}>()

const POLICIES: ApprovalPolicy[] = ['always', 'never']

/** 仅当 rag 能力可用时展示方案选择器（体验「换方案看提升」） */
const ragAvailable = computed(() => props.caps.some((c) => c.id === 'rag' && c.availability === 'available'))

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

      <!-- RAG 方案：仅 rag 能力可用时展示，直观对比各方案检索与回答差异 -->
      <section v-if="ragAvailable" class="space-y-3 border-t border-slate-800 pt-4">
        <div class="mb-1 flex items-center justify-between">
          <h3 class="text-xs font-semibold text-slate-300">RAG 方案</h3>
          <span class="text-[10px] text-slate-500" title="同一语料、不同检索策略；切换后检索卡片与回答随方案变化">对比检索差异</span>
        </div>
        <RagSchemeSelector
          :model-value="ragScheme"
          :schemes="ragSchemes"
          @update:model-value="emit('update:rag-scheme', $event)"
        />
      </section>

      <!-- MCP 服务：默认关闭，页面点选开启；直观对比有无 MCP -->
      <section class="border-t border-slate-800 pt-4">
        <div class="mb-2 flex items-center justify-between">
          <h3 class="text-xs font-semibold text-slate-300">MCP 服务</h3>
          <button
            type="button"
            role="switch"
            :aria-checked="mcpEnabled"
            :title="mcpEnabled ? '关闭 MCP 服务（移除 MCP 能力）' : '开启 MCP 服务（连接 mcp-notes 便签 server）'"
            class="flex h-5 w-9 shrink-0 items-center rounded-full p-0.5 transition"
            :class="mcpEnabled ? 'bg-fuchsia-500' : 'bg-slate-700'"
            @click="emit('toggle-mcp', !mcpEnabled)"
          >
            <span class="h-4 w-4 rounded-full bg-white transition" :class="mcpEnabled ? 'translate-x-4' : ''"></span>
          </button>
        </div>

        <div v-if="!mcpEnabled" class="rounded-xl border border-dashed border-slate-700/70 px-3 py-3 text-center">
          <p class="text-[11px] text-slate-500">MCP 未启用 — 仅使用内置能力</p>
          <p class="mt-1 text-[10px] text-slate-600">开启后接入 mcp-notes 便签服务，可扩展 4 个 MCP 工具</p>
        </div>

        <div v-else class="space-y-2">
          <p class="text-[11px] text-fuchsia-300/90">已连接 mcp-notes · 发现 {{ mcpCaps?.length ?? 0 }} 个工具</p>
          <CapabilityGrid
            :caps="mcpCaps ?? []"
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
        </div>
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
