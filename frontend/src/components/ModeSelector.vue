<script setup lang="ts">
import type { ModeId } from '../types/agent'

defineProps<{ modelValue: ModeId }>()
const emit = defineEmits<{ 'update:modelValue': [v: ModeId] }>()

const OPTIONS: { id: ModeId; name: string; desc: string }[] = [
  { id: 'react', name: 'ReAct', desc: '思考-行动-观察' },
  { id: 'plan_execute', name: '计划执行', desc: '拆解计划逐步执行' },
  { id: 'reflection', name: '反思修订', desc: '草稿→批评→修订' },
  { id: 'multi_agent', name: '多智能体', desc: 'Orchestrator 分派' },
]
</script>

<template>
  <div class="grid grid-cols-2 gap-2">
    <button
      v-for="o in OPTIONS"
      :key="o.id"
      type="button"
      class="rounded-xl border px-3 py-2 text-left transition"
      :class="
        o.id === modelValue
          ? 'border-indigo-400 bg-indigo-500/20 text-white'
          : 'border-slate-700 bg-slate-800/50 text-slate-300 hover:border-slate-500'
      "
      @click="emit('update:modelValue', o.id)"
    >
      <span class="block text-sm font-medium">{{ o.name }}</span>
      <span class="block text-[11px] text-slate-400">{{ o.desc }}</span>
    </button>
  </div>
</template>
