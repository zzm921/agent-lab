<script setup lang="ts">
import type { RagScheme, RagSchemeId } from '../types/agent'

const props = defineProps<{
  modelValue: RagSchemeId
  schemes: RagScheme[]
}>()
const emit = defineEmits<{ 'update:modelValue': [v: RagSchemeId] }>()
</script>

<template>
  <div class="grid grid-cols-2 gap-2">
    <button
      v-for="o in schemes"
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
