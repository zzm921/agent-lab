<script setup lang="ts">
import type { ErrorInfo } from '../composables/useChatStream'

defineProps<{ error: ErrorInfo | null }>()
const emit = defineEmits<{ retry: [] }>()
</script>

<template>
  <div v-if="error" class="rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-sm">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <p class="font-medium text-rose-300">{{ error.message }}</p>
        <p v-if="error.detail" class="mt-1 break-all font-mono text-xs text-rose-400/80">{{ error.detail }}</p>
        <p v-if="error.detail?.includes('500') || /未配置|API Key|api_key/i.test(error.message + (error.detail ?? ''))" class="mt-1 text-xs text-amber-400">
          提示：请确认后端已配置阿里云百炼（DashScope）API Key，否则无法运行 Agent 对话。
        </p>
      </div>
      <button
        type="button"
        class="shrink-0 rounded-lg border border-rose-500/50 px-2.5 py-1 text-xs text-rose-300 transition hover:bg-rose-500/10"
        @click="emit('retry')"
      >
        重试
      </button>
    </div>
  </div>
</template>
