<script setup lang="ts">
import type { TodoItem } from '../types/agent'

const props = defineProps<{ items: TodoItem[]; currentStep?: number; status?: string }>()

function badgeClass(t: TodoItem, i: number) {
  if (t.status === 'done') return 'bg-emerald-500 text-slate-900'
  if (t.status === 'failed') return 'bg-rose-500 text-white'
  if (i === 0 || i === (props.currentStep ?? 0)) return 'bg-indigo-400 text-slate-900'
  return 'bg-slate-700 text-slate-300'
}
</script>

<template>
  <ol class="space-y-0">
    <li v-for="(t, i) in items" :key="t.id" class="relative flex gap-3 pb-4 last:pb-0">
      <div class="flex flex-col items-center">
        <span
          class="grid h-6 w-6 shrink-0 place-items-center rounded-full text-[11px] font-semibold"
          :class="badgeClass(t, i)"
        >
          <span v-if="t.status === 'done'">✓</span>
          <span v-else-if="t.status === 'failed'">✗</span>
          <span v-else>{{ i + 1 }}</span>
        </span>
        <span v-if="i < items.length - 1" class="w-px flex-1 bg-slate-700"></span>
      </div>
      <div class="pt-0.5">
        <p
          class="text-sm"
          :class="
            t.status === 'done'
              ? 'text-slate-500 line-through'
              : t.status === 'failed'
                ? 'text-rose-300'
                : i === (currentStep ?? 0)
                  ? 'text-white'
                  : 'text-slate-400'
          "
        >
          {{ t.desc }}
        </p>
        <span v-if="t.deps?.length" class="text-[11px] text-slate-500">依赖 {{ t.deps.join('、') }}</span>
        <span v-if="t.status === 'done'" class="ml-2 text-[11px] text-emerald-400">已完成</span>
        <span v-else-if="t.status === 'failed'" class="ml-2 text-[11px] text-rose-400">失败</span>
        <span v-else-if="i === (currentStep ?? 0) && status === 'running'" class="ml-2 text-[11px] text-indigo-300">执行中</span>
      </div>
    </li>
  </ol>
</template>
