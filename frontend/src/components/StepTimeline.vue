<script setup lang="ts">
defineProps<{ steps: string[]; currentStep?: number; status?: string }>()
</script>

<template>
  <ol class="space-y-0">
    <li v-for="(s, i) in steps" :key="i" class="relative flex gap-3 pb-4 last:pb-0">
      <div class="flex flex-col items-center">
        <span
          class="grid h-6 w-6 shrink-0 place-items-center rounded-full text-[11px] font-semibold"
          :class="i === currentStep ? 'bg-indigo-400 text-slate-900' : 'bg-slate-700 text-slate-300'"
        >
          {{ i + 1 }}
        </span>
        <span v-if="i < steps.length - 1" class="w-px flex-1 bg-slate-700"></span>
      </div>
      <div class="pt-0.5">
        <p
          class="text-sm"
          :class="i === currentStep ? 'text-white' : (currentStep ?? -1) >= 0 && i < currentStep ? 'text-slate-500 line-through' : 'text-slate-400'"
        >
          {{ s }}
        </p>
        <span v-if="i === currentStep && status === 'running'" class="text-[11px] text-indigo-300">执行中…</span>
        <span v-else-if="i === currentStep && status === 'done'" class="text-[11px] text-emerald-400">已完成</span>
      </div>
    </li>
  </ol>
</template>
