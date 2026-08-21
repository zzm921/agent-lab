<script setup lang="ts">
import { onMounted, ref } from 'vue'

interface Health {
  status: string
  model: string
  mcp_configured: boolean
  embedding_configured: boolean
}

const health = ref<Health | null>(null)

async function fetchHealth() {
  try {
    const res = await fetch('/api/health')
    if (res.ok) health.value = await res.json()
  } catch {
    health.value = null
  }
}
onMounted(fetchHealth)
</script>

<template>
  <header class="sticky top-0 z-40 border-b border-slate-800/80 bg-slate-950/85 backdrop-blur">
    <div class="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3">
      <RouterLink to="/" class="flex items-center gap-3">
        <div class="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-lg font-bold text-white">A</div>
        <div>
          <h1 class="text-base leading-tight font-semibold text-white">Agent Lab</h1>
          <p class="text-xs text-slate-400">Agent 实验室</p>
        </div>
      </RouterLink>

      <div class="hidden items-center gap-2 text-xs md:flex">
        <span :class="health ? 'text-emerald-400' : 'text-rose-400'">
          {{ health ? '后端在线' : '后端未连接' }}
        </span>
        <span v-if="health" class="rounded bg-slate-800 px-2 py-0.5 text-slate-300">{{ health.model }}</span>
        <span v-if="health" class="rounded bg-slate-800 px-2 py-0.5" :class="health.mcp_configured ? 'text-emerald-400' : 'text-slate-500'">
          MCP {{ health.mcp_configured ? '已配置' : '未配置' }}
        </span>
        <span v-if="health" class="rounded bg-slate-800 px-2 py-0.5" :class="health.embedding_configured ? 'text-emerald-400' : 'text-amber-400'">
          Embedding {{ health.embedding_configured ? '已配置' : '未配置' }}
        </span>
      </div>
    </div>
  </header>
</template>
