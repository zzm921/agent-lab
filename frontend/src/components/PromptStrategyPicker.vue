<script setup lang="ts">
import type { PromptStrategy } from '../types/agent'

defineProps<{ modelValue: PromptStrategy }>()
const emit = defineEmits<{ 'update:modelValue': [v: PromptStrategy] }>()

const OPTIONS: { id: PromptStrategy; name: string; desc: string }[] = [
  { id: 'standard', name: 'Standard', desc: '直接回答' },
  { id: 'few_shot', name: 'Few-Shot', desc: '示例引导' },
  { id: 'cot', name: 'CoT', desc: '逐步思考' },
]
</script>

<template>
  <div class="flex flex-wrap items-center gap-2">
    <span class="text-xs text-slate-400">提示词策略</span>
    <button
      v-for="o in OPTIONS"
      :key="o.id"
      type="button"
      class="rounded-lg border px-2.5 py-1 text-xs transition"
      :class="
        o.id === modelValue
          ? 'border-fuchsia-400 bg-fuchsia-500/20 text-white'
          : 'border-slate-700 bg-slate-800/50 text-slate-400 hover:border-slate-500 hover:text-slate-200'
      "
      :title="o.desc"
      @click="emit('update:modelValue', o.id)"
    >
      {{ o.name }}
    </button>
  </div>
</template>
