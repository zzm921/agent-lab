<script setup lang="ts">
import { JOURNEY_STAGES } from '../data/capabilityData'

const ICONS: Record<string, string> = {
  shield:
    'M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z',
  compass:
    'M16.712 4.33a9.027 9.027 0 0 1 1.652 1.306c.51.51.944 1.064 1.306 1.652M16.712 4.33l-3.448 4.138m3.448-4.138a9.014 9.014 0 0 0-9.424 0M19.67 7.288l-4.138 3.448m4.138-3.448a9.014 9.014 0 0 1 0 9.424m-4.138-5.976a3.736 3.736 0 0 0-.88-1.388 3.737 3.737 0 0 0-1.388-.88m2.268 2.268a3.765 3.765 0 0 1-2.268 2.268m0-4.536a3.765 3.765 0 0 1-2.268 2.268m0 4.536a3.737 3.737 0 0 1-1.388-.88 3.737 3.737 0 0 1-.88-1.388m0 2.268a9.015 9.015 0 0 1-4.418-1.157M4.33 16.712a9.027 9.027 0 0 1-1.306-1.652m1.306 1.652 4.138-3.448m-4.138 3.448a9.014 9.014 0 0 0 9.424 0M4.67 7.288l4.138 3.448m-4.138-3.448a9.014 9.014 0 0 0 0 9.424m0-9.424a9.027 9.027 0 0 1 1.306-1.652m0 0a9.014 9.014 0 0 1 9.424 0',
  cpu: 'M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 0 0 2.25-2.25V6.75a2.25 2.25 0 0 0-2.25-2.25H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Zm.75-12h9v9h-9v-9Z',
  zap: 'M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z',
  terminal:
    'm6.75 7.5 3 2.25-3 2.25m4.5 0h3m-9 8.25h13.5A2.25 2.25 0 0 0 21 18V6a2.25 2.25 0 0 0-2.25-2.25H5.25A2.25 2.25 0 0 0 3 6v12a2.25 2.25 0 0 0 2.25 2.25Z',
  database:
    'M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 3.75v3.75m-16.5-3.75v3.75',
  brain:
    'M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z',
  check:
    'M9 12.75 11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 0 1-1.043 3.296 3.745 3.745 0 0 1-3.296 1.043A3.745 3.745 0 0 1 12 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 0 1-3.296-1.043 3.745 3.745 0 0 1-1.043-3.296A3.745 3.745 0 0 1 3 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 0 1 1.043-3.296 3.746 3.746 0 0 1 3.296-1.043A3.746 3.746 0 0 1 12 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 0 1 3.296 1.043 3.745 3.745 0 0 1 1.043 3.296A3.745 3.745 0 0 1 21 12Z',
  'chart-bar':
    'M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z',
  sparkles:
    'M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z',
}

function iconPath(name: string) {
  return ICONS[name] || ICONS.sparkles
}

/** 真实技术栈徽标（替代原分层里的虚构技术：Pinia/Monaco/PostgreSQL/Redis） */
const TECH_STACK_BADGES = [
  'Vue 3',
  'TypeScript',
  'Vite',
  'Tailwind CSS',
  'FastAPI',
  'LangGraph',
  'LangChain',
  'DashScope Qwen',
  'MCP',
  'Qdrant',
  'Elasticsearch',
  'SSE',
]

function pad(n: number) {
  return String(n).padStart(2, '0')
}
</script>

<template>
  <section class="pt-16">
    <div class="mb-8">
      <h2 class="text-2xl font-semibold text-white">一次对话的完整旅程</h2>
      <p class="mt-1 text-sm text-slate-400">
        每条消息背后，平台按 {{ JOURNEY_STAGES.length }} 个站点依次运转——从提问到观测，每一步都有真实实现
      </p>
    </div>

    <div class="relative">
      <!-- 竖向连接线 -->
      <div
        class="pointer-events-none absolute bottom-10 left-6 top-10 w-px bg-white/10"
      ></div>

      <div class="space-y-6">
        <div v-for="(stop, i) in JOURNEY_STAGES" :key="stop.id" class="relative">
          <!-- 节点图标 -->
          <div
            class="absolute left-0 top-0 z-10 grid h-12 w-12 place-items-center rounded-2xl border border-white/10"
            :style="{ background: `${stop.color}20`, color: stop.color }"
          >
            <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" :d="iconPath(stop.icon)" />
            </svg>
          </div>

          <!-- 内容卡片 -->
          <div class="ml-16 rounded-2xl border border-white/10 bg-white/[0.03] p-5">
            <div class="flex items-center gap-2.5">
              <span class="text-xs font-bold tabular-nums" :style="{ color: stop.color }">{{ pad(i + 1) }}</span>
              <h4 class="text-base font-semibold text-white">{{ stop.name }}</h4>
            </div>

            <!-- 单通道阶段 -->
            <template v-if="stop.desc">
              <p class="mt-2 text-xs leading-relaxed text-slate-400">{{ stop.desc }}</p>
              <div v-if="stop.techs?.length" class="mt-3 flex flex-wrap gap-1.5">
                <span
                  v-for="t in stop.techs"
                  :key="t"
                  class="rounded-md border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-slate-300"
                >
                  {{ t }}
                </span>
              </div>
            </template>

            <!-- 双通道阶段（工具执行 / 检索增强） -->
            <div v-else-if="stop.channels?.length" class="mt-3 grid gap-3 sm:grid-cols-2">
              <div
                v-for="ch in stop.channels"
                :key="ch.id"
                class="rounded-xl border p-4"
                :style="{ borderColor: `${ch.color}33`, background: `${ch.color}0d` }"
              >
                <div class="flex items-center gap-2">
                  <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" :style="{ color: ch.color }">
                    <path stroke-linecap="round" stroke-linejoin="round" :d="iconPath(ch.icon)" />
                  </svg>
                  <span class="text-sm font-medium text-white">{{ ch.name }}</span>
                </div>
                <p class="mt-2 text-xs leading-relaxed text-slate-400">{{ ch.desc }}</p>
                <div class="mt-2.5 flex flex-wrap gap-1.5">
                  <span
                    v-for="t in ch.techs"
                    :key="t"
                    class="rounded bg-white/5 px-1.5 py-0.5 text-[10px]"
                    :style="{ color: ch.color }"
                  >
                    {{ t }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 真实技术栈徽标带 -->
    <div class="mt-10 flex flex-wrap items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.03] p-4">
      <span class="px-1 text-xs font-medium text-slate-500">真实技术栈</span>
      <span
        v-for="t in TECH_STACK_BADGES"
        :key="t"
        class="rounded-md bg-white/5 px-2.5 py-1 text-xs text-slate-300"
      >
        {{ t }}
      </span>
    </div>
  </section>
</template>
