<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { MODES } from '../data/techModules'

interface Health {
  status: string
  model: string
  mcp_configured: boolean
  embedding_configured: boolean
}

const health = ref<Health | null>(null)
const showModes = ref(false)

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
          <h1 class="text-base leading-tight font-semibold text-white">AI Agent 平台</h1>
          <p class="text-xs text-slate-400">能力热插拔 · MCP · 多模式推理</p>
        </div>
      </RouterLink>

      <nav class="flex items-center gap-1 text-sm">
        <RouterLink to="/" class="rounded-lg px-3 py-1.5 text-slate-300 hover:bg-slate-800 hover:text-white">首页</RouterLink>
        <RouterLink to="/compare" class="rounded-lg px-3 py-1.5 text-slate-300 hover:bg-slate-800 hover:text-white">模式对比</RouterLink>
        <div class="relative" @mouseenter="showModes = true" @mouseleave="showModes = false">
          <button type="button" class="rounded-lg px-3 py-1.5 text-slate-300 hover:bg-slate-800 hover:text-white">模式详情 ▾</button>
          <div v-if="showModes" class="absolute right-0 mt-1 w-44 rounded-xl border border-slate-700 bg-slate-900 p-1 shadow-xl">
            <RouterLink v-for="m in MODES" :key="m.id" :to="`/module/${m.id}`" class="block rounded-lg px-3 py-2 text-slate-300 hover:bg-slate-800 hover:text-white">
              <span class="block text-sm">{{ m.name }}</span>
              <span class="block text-[11px] text-slate-500">{{ m.tagline }}</span>
            </RouterLink>
          </div>
        </div>
      </nav>

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
