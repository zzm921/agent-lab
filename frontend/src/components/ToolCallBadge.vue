<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import type { ToolCallEntry } from '../composables/useChatStream'

const props = defineProps<{ entry: ToolCallEntry }>()

const STATUS: Record<ToolCallEntry['status'], { label: string; cls: string }> = {
  running: { label: '执行中', cls: 'border-amber-500/40 bg-amber-500/10 text-amber-300' },
  success: { label: '成功', cls: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300' },
  failed: { label: '失败', cls: 'border-rose-500/40 bg-rose-500/10 text-rose-300' },
  rejected: { label: '已拒绝', cls: 'border-slate-500/40 bg-slate-500/10 text-slate-400' },
}

// —— 重试实时倒计时：从 retryDelay（实际睡眠，含抖动）递减，直观展示退避等待 ——
const remaining = ref(0)
let timer: number | undefined

function stopTimer() {
  if (timer) {
    clearInterval(timer)
    timer = undefined
  }
}

watch(
  () => [props.entry.retryCount, props.entry.retryDelay, props.entry.status] as const,
  ([count, delay, status]) => {
    stopTimer()
    if (count && delay != null && status === 'running') {
      remaining.value = delay
      timer = window.setInterval(() => {
        remaining.value = Math.max(0, remaining.value - 0.1)
        if (remaining.value <= 0) stopTimer()
      }, 100)
    } else {
      remaining.value = 0
    }
  },
  { immediate: true },
)

onUnmounted(stopTimer)

// —— 指数退避序列：由当前纯指数值反推首项，得到 0.5 → 1 → 2 → … 曲线供阶梯条展示 ——
const backoffSeq = computed<number[]>(() => {
  const attempt = props.entry.retryCount ?? 0
  const max = props.entry.retryMax ?? 0
  const baseDelay = props.entry.retryBaseDelay ?? props.entry.retryDelay ?? 0
  if (!attempt || !max || !baseDelay) return []
  const base = baseDelay / 2 ** (attempt - 1)
  return Array.from({ length: max }, (_, i) => base * 2 ** i)
})
const currentAttempt = computed(() => props.entry.retryCount ?? 0)
</script>

<template>
  <div class="rounded-xl border px-3 py-2 text-xs" :class="STATUS[entry.status].cls">
    <div class="flex items-center justify-between gap-2">
      <span class="flex min-w-0 items-center gap-1.5">
        <!-- 执行中加载动画 -->
        <svg
          v-if="entry.status === 'running'"
          class="h-3.5 w-3.5 shrink-0 animate-spin text-amber-300"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-90" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
        </svg>
        <span class="truncate font-mono font-semibold">{{ entry.tool }}</span>
      </span>
      <span class="shrink-0">{{ STATUS[entry.status].label }}</span>
      <!-- 重试徽标：n/m + 指数退避秒数 -->
      <span
        v-if="entry.retryCount"
        class="shrink-0 rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-amber-300"
        :title="entry.retryReason"
      >
        重试 {{ entry.retryCount }}/{{ entry.retryMax }}
        <template v-if="entry.status === 'running'">· 退避 {{ entry.retryBaseDelay?.toFixed(1) }}s</template>
      </span>
    </div>

    <!-- 重试进行中：下一次重试的实时倒计时 -->
    <div v-if="entry.retryCount && entry.status === 'running' && remaining > 0" class="mt-2">
      <div class="flex items-center justify-between font-mono text-[11px] text-amber-300/90">
        <span>{{ remaining.toFixed(1) }}s 后第 {{ currentAttempt }}/{{ entry.retryMax }} 次重试</span>
        <span title="实际等待 = 指数退避值 × (±50% 抖动)">抖动后 {{ entry.retryDelay?.toFixed(2) }}s</span>
      </div>
      <div class="mt-1 h-1.5 overflow-hidden rounded-full bg-black/40">
        <div
          class="h-full rounded-full bg-amber-400/80 transition-[width] duration-100 ease-linear"
          :style="{ width: `${(remaining / (entry.retryDelay ?? 1)) * 100}%` }"
        ></div>
      </div>
    </div>

    <!-- 指数退避序列阶梯条：间隔逐次翻倍，当前尝试高亮 -->
    <div v-if="backoffSeq.length && entry.status === 'running'" class="mt-2 flex items-end gap-1.5">
      <div v-for="(v, i) in backoffSeq" :key="i" class="flex flex-1 flex-col items-center gap-0.5">
        <span class="font-mono text-[10px]" :class="i + 1 === currentAttempt ? 'text-amber-300' : 'text-slate-500'">
          {{ v.toFixed(1) }}s
        </span>
        <div
          class="w-full rounded-t"
          :class="
            i + 1 === currentAttempt
              ? 'bg-amber-400/90'
              : i + 1 < currentAttempt
                ? 'bg-amber-500/40'
                : 'bg-slate-700'
          "
          :style="{ height: `${8 + (i + 1) * 6}px` }"
        ></div>
        <span class="font-mono text-[9px] text-slate-500">#{{ i + 1 }}</span>
      </div>
    </div>

    <pre class="mt-1.5 max-h-40 overflow-auto rounded-lg bg-black/30 p-2 text-[11px] text-slate-300">{{ JSON.stringify(entry.args, null, 2) }}</pre>
    <div v-if="entry.result" class="mt-1.5 break-words text-slate-300">{{ entry.result }}</div>
    <div v-if="entry.retryCount && entry.status === 'failed'" class="mt-1 text-[11px] text-amber-300/80">
      自动重试已耗尽，已把结构化错误返回给模型重新思考
    </div>
  </div>
</template>
