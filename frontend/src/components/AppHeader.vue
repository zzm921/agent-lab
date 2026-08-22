<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { theme, toggleTheme } from '../composables/useTheme'

interface Health {
  status: string
  model: string
  mcp_configured: boolean
  embedding_configured: boolean
}

const health = ref<Health | null>(null)
const route = useRoute()

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
    <div class="grid w-full grid-cols-[1fr_auto_1fr] items-center gap-4 px-4 py-3">
      <RouterLink to="/" class="flex items-center justify-self-start gap-3">
          <div class="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-lg font-bold text-white">A</div>
          <div>
            <h1 class="text-base leading-tight font-semibold text-white">Agent Lab</h1>
            <p class="text-xs text-slate-400">Agent实验室</p>
          </div>
        </RouterLink>

        <nav class="hidden items-center justify-self-center gap-1 md:flex">
          <RouterLink
            to="/"
            class="rounded-lg px-3 py-1.5 text-sm transition"
            :class="route.path === '/' ? 'bg-white/10 text-white' : 'text-slate-400 hover:bg-white/5 hover:text-white'"
          >
            <span class="flex items-center gap-1.5">
              <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM9.75 6A2.25 2.25 0 0 1 12 3.75h2.25A2.25 2.25 0 0 1 16.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H12a2.25 2.25 0 0 1-2.25-2.25V6ZM15.75 6A2.25 2.25 0 0 1 18 3.75h2.25A2.25 2.25 0 0 1 22.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H18a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM9.75 15.75A2.25 2.25 0 0 1 12 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H12a2.25 2.25 0 0 1-2.25-2.25v-2.25ZM15.75 15.75A2.25 2.25 0 0 1 18 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H18a2.25 2.25 0 0 1-2.25-2.25v-2.25Z" />
              </svg>
              能力地图
            </span>
          </RouterLink>
          <RouterLink
            to="/lab"
            class="rounded-lg px-3 py-1.5 text-sm transition"
            :class="route.path === '/lab' ? 'bg-white/10 text-white' : 'text-slate-400 hover:bg-white/5 hover:text-white'"
          >
            <span class="flex items-center gap-1.5">
              <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714a2.25 2.25 0 0 0 .659 1.591L18.75 14.5M13.5 3.104c.251.023.501.05.75.082M19.5 14.5l-1.8 6.533a2.25 2.25 0 0 1-2.174 1.665H8.474a2.25 2.25 0 0 1-2.174-1.665L4.5 14.5M19.5 14.5h-15" />
              </svg>
              实验室
            </span>
          </RouterLink>
        </nav>

      <div class="hidden items-center justify-self-end gap-2 text-xs md:flex">
        <button
          type="button"
          class="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-800 hover:text-white"
          :title="theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'"
          @click="toggleTheme"
        >
          <svg v-if="theme === 'dark'" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 3v2.25m6.364.386-1.591 1.591M21 12h-2.25m-.386 6.364-1.591-1.591M12 18.75V21m-4.773-4.227-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0Z" />
          </svg>
          <svg v-else class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21.752 15.002A9.72 9.72 0 0 1 18 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 0 0 3 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 0 0 9.002-5.998Z" />
          </svg>
        </button>
        <span :class="health ? 'text-emerald-400' : 'text-rose-400'">
          {{ health ? '后端在线' : '后端未连接' }}
        </span>
        <span v-if="health" class="rounded bg-slate-800 px-2 py-0.5 text-slate-300">{{ health.model }}</span>
        <span v-if="health" class="rounded bg-slate-800 px-2 py-0.5" :class="health.mcp_configured ? 'text-emerald-400' : 'text-slate-500'">
          MCP {{ health.mcp_configured ? '已连接' : '未配置' }}
        </span>
        <span v-if="health" class="rounded bg-slate-800 px-2 py-0.5" :class="health.embedding_configured ? 'text-emerald-400' : 'text-amber-400'">
          Embedding {{ health.embedding_configured ? '已配置' : '未配置' }}
        </span>
      </div>
    </div>
  </header>
</template>
