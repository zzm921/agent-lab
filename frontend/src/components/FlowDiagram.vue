<script setup lang="ts">
import { computed } from 'vue'
import type { ModeMeta } from '../data/techModules'

const props = defineProps<{ mode: ModeMeta }>()

const KIND_STYLE: Record<string, string> = {
  start: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300',
  llm: 'border-indigo-500/50 bg-indigo-500/10 text-indigo-200',
  tool: 'border-amber-500/50 bg-amber-500/10 text-amber-200',
  plan: 'border-cyan-500/50 bg-cyan-500/10 text-cyan-200',
  decision: 'border-fuchsia-500/50 bg-fuchsia-500/10 text-fuchsia-200',
  worker: 'border-sky-500/50 bg-sky-500/10 text-sky-200',
  reflect: 'border-rose-500/50 bg-rose-500/10 text-rose-200',
  revise: 'border-orange-500/50 bg-orange-500/10 text-orange-200',
  end: 'border-slate-500/50 bg-slate-500/10 text-slate-300',
}

/** 回边（指向更上方节点）作为循环标注展示 */
const loopEdges = computed(() => {
  const index = new Map(props.mode.nodes.map((n, i) => [n.id, i]))
  return props.mode.edges.filter((e) => (index.get(e.from) ?? 0) > (index.get(e.to) ?? 0))
})
</script>

<template>
  <div class="flex flex-col items-center">
    <div v-for="(n, i) in mode.nodes" :key="n.id" class="flex w-full flex-col items-center">
      <div
        class="w-full rounded-xl border px-3 py-2 text-center text-sm"
        :class="KIND_STYLE[n.kind] ?? KIND_STYLE.end"
      >
        <span v-if="n.kind === 'decision'" class="mr-1">◇</span>
        <span class="font-medium">{{ n.label }}</span>
        <span v-if="n.note" class="mt-0.5 block text-[11px] opacity-80">{{ n.note }}</span>
      </div>
      <div v-if="i < mode.nodes.length - 1" class="flex h-7 flex-col items-center">
        <span class="h-3 w-px bg-slate-600"></span>
        <span class="text-[10px] leading-none text-slate-500">▼</span>
      </div>
    </div>

    <div v-for="e in loopEdges" :key="e.from + '>' + e.to" class="mt-2 w-full rounded-lg border border-dashed border-slate-700 px-2 py-1 text-center text-[11px] text-slate-400">
      ↻ {{ e.label ?? '循环回边' }}（{{ e.from }} → {{ e.to }}）
    </div>
  </div>
</template>
