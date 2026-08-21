<script setup lang="ts">
import type { Capability } from '../types/agent'

defineProps<{ cap: Capability | null }>()
const emit = defineEmits<{ close: [] }>()
</script>

<template>
  <Transition name="fade">
    <div
      v-if="cap"
      class="flex items-center justify-between gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm"
    >
      <span class="text-emerald-300">
        「{{ cap.name }}」示例已填入，可直接发送体验（该能力已自动启用）
      </span>
      <button type="button" class="shrink-0 text-slate-400 transition hover:text-white" @click="emit('close')">✕</button>
    </div>
  </Transition>
</template>
