<script setup lang="ts">
defineProps<{ guard: { reason: string; matched?: string } | null }>()
</script>

<template>
  <div v-if="guard" class="rounded-xl border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
    <p class="font-medium text-amber-300">
      <span class="mr-1">🛡</span>{{ guard.reason }}，本次请求已被安全护栏拦截
    </p>
    <p v-if="guard.matched" class="mt-1 break-all font-mono text-xs text-amber-400/80">
      命中特征：{{ guard.matched }}
    </p>
  </div>
</template>
