<script setup lang="ts">
import type { ToolCallEntry } from '../composables/useChatStream'

defineProps<{ entry: ToolCallEntry }>()

const STATUS: Record<ToolCallEntry['status'], { label: string; cls: string }> = {
  running: { label: '执行中', cls: 'border-amber-500/40 bg-amber-500/10 text-amber-300' },
  success: { label: '成功', cls: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300' },
  failed: { label: '失败', cls: 'border-rose-500/40 bg-rose-500/10 text-rose-300' },
  rejected: { label: '已拒绝', cls: 'border-slate-500/40 bg-slate-500/10 text-slate-400' },
}
</script>

<template>
  <div class="rounded-xl border px-3 py-2 text-xs" :class="STATUS[entry.status].cls">
    <div class="flex items-center justify-between gap-2">
      <span class="flex min-w-0 items-center gap-1.5">
        <!-- 执行中加载动画 -->
        <svg
          v-if="entry.status === 'running'"
          class="h-3.5 w-3.5 shrink-0 animate-spin text-amber-300"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-90" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
        </svg>
        <span class="truncate font-mono font-semibold">{{ entry.tool }}</span>
      </span>
      <span class="shrink-0">{{ STATUS[entry.status].label }}</span>
    </div>
    <pre class="mt-1.5 max-h-40 overflow-auto rounded-lg bg-black/30 p-2 text-[11px] text-slate-300">{{ JSON.stringify(entry.args, null, 2) }}</pre>
    <div v-if="entry.result" class="mt-1.5 break-words text-slate-300">{{ entry.result }}</div>
  </div>
</template>
