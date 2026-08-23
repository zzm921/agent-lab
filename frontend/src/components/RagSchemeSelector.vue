<script setup lang="ts">
import type { RagScheme, RagSchemeId } from '../types/agent'

const props = defineProps<{
  modelValue: RagSchemeId
  schemes: RagScheme[]
}>()
const emit = defineEmits<{ 'update:modelValue': [v: RagSchemeId] }>()

/** 后端未返回方案目录时兜底展示的方案 */
const FALLBACK: RagScheme[] = [
  { id: 'naive', name: '朴素 RAG', description: '固定切块 + 纯稠密向量检索', collection: '', count: 0 },
  { id: 'advanced', name: '高级 RAG', description: '语义分块 + 混合检索 + Query重写 + Rerank', collection: '', count: 0 },
]

function options(): RagScheme[] {
  return props.schemes.length ? props.schemes : FALLBACK
}
</script>

<template>
  <div class="flex flex-wrap gap-2">
    <button
      v-for="o in options()"
      :key="o.id"
      type="button"
      class="rounded-xl border px-3 py-2 text-left transition"
      :class="
        o.id === modelValue
          ? 'border-emerald-400 bg-emerald-500/20 text-white'
          : 'border-slate-700 bg-slate-800/50 text-slate-300 hover:border-slate-500'
      "
      @click="emit('update:modelValue', o.id as RagSchemeId)"
    >
      <span class="block text-sm font-medium">{{ o.name }}</span>
      <span class="block text-[11px] text-slate-400">{{ o.description }}</span>
    </button>
  </div>
</template>
