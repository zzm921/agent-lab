<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import LiveStage from './LiveStage.vue'
import TaskInput from './TaskInput.vue'
import { fetchQuota } from '../services/sse'
import type { ApprovalPolicy, Capability, ChatStream, ModeId, PromptStrategy } from '../types/agent'

const props = defineProps<{
  stream: ChatStream
  task: string
  mode: ModeId
  strategy: PromptStrategy
  policy: ApprovalPolicy
  sending: boolean
  enabledCapabilities: Capability[]
  filesOpen?: boolean
}>()

const emit = defineEmits<{
  'update:task': [v: string]
  send: []
  'toggle-files': [v: boolean]
}>()

// 流水线内容滚动容器：接近底部时随内容自动下滚，保证始终可见最新步骤
const stageEl = ref<HTMLElement | null>(null)
watch(
  [() => props.stream.steps, () => props.stream.done],
  async () => {
    await nextTick()
    const el = stageEl.value
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120
    if (nearBottom) el.scrollTop = el.scrollHeight
  },
  { deep: true },
)

const modeName = computed(() => {
  const map: Record<ModeId, string> = {
    react: 'ReAct',
    plan_execute: '计划执行',
    reflection: '反思修订',
    multi_agent: '多智能体',
  }
  return map[props.mode]
})

const strategyName = computed(() => {
  const map: Record<PromptStrategy, string> = {
    standard: 'Standard',
    few_shot: 'Few-Shot',
    cot: 'CoT',
  }
  return map[props.strategy]
})

function onSend() {
  if (!props.task.trim() || props.sending) return
  emit('send')
}

// 每日对话配额：展示「今日剩余次数」，进入页面与每次对话结束后刷新
const quota = ref<{ enabled: boolean; limit: number; remaining: number } | null>(null)

async function refreshQuota() {
  try {
    quota.value = await fetchQuota()
  } catch {
    quota.value = null // 接口不可用时静默隐藏
  }
}

onMounted(refreshQuota)
watch(
  () => props.stream.status,
  (s) => {
    if (s === 'done' || s === 'error') refreshQuota()
  },
)
</script>

<template>
  <div class="flex h-full flex-col">
    <div class="flex flex-col gap-2 border-b border-slate-800 px-4 py-2.5 md:flex-row md:items-center md:justify-between">
      <div class="flex flex-wrap items-center gap-2 text-xs">
        <span class="rounded bg-slate-800 px-2 py-0.5 text-slate-300">{{ modeName }}</span>
        <span class="rounded bg-slate-800 px-2 py-0.5 text-slate-300">{{ strategyName }}</span>
        <span
          class="rounded px-2 py-0.5"
          :class="policy === 'always' ? 'bg-amber-500/15 text-amber-300' : 'bg-slate-800 text-slate-400'"
        >
          {{ policy === 'always' ? '执行前审批' : '自动执行' }}
        </span>
      </div>
      <div class="flex items-center gap-2">
        <div v-if="enabledCapabilities.length" class="flex flex-wrap items-center gap-1.5">
          <span class="text-[11px] text-slate-500">已装配</span>
          <span
            v-for="c in enabledCapabilities.slice(0, 4)"
            :key="c.id"
            class="rounded bg-indigo-500/15 px-1.5 py-0.5 text-[11px] text-indigo-300"
          >
            {{ c.name }}
          </span>
          <span v-if="enabledCapabilities.length > 4" class="text-[11px] text-slate-500">
            +{{ enabledCapabilities.length - 4 }}
          </span>
        </div>
        <div v-else class="text-[11px] text-slate-500">未装配任何能力</div>

        <button
          type="button"
          class="flex shrink-0 items-center gap-1 rounded-lg border px-2 py-1 text-[11px] transition"
          :class="
            filesOpen
              ? 'border-indigo-500/50 bg-indigo-500/15 text-indigo-300'
              : 'border-slate-700 text-slate-400 hover:bg-slate-800 hover:text-white'
          "
          title="沙箱文件（Agent 在沙箱中写出的产物，可下载）"
          @click="emit('toggle-files', !filesOpen)"
        >
          <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
          </svg>
          沙箱文件
        </button>
      </div>
    </div>

    <div ref="stageEl" class="flex-1 overflow-y-auto p-4">
      <LiveStage :stream="stream" />
    </div>

    <div class="border-t border-slate-800 p-4">
      <div class="mb-2 flex items-center justify-between text-[11px] text-slate-500">
        <span v-if="quota?.enabled">
          今日剩余
          <span :class="quota.remaining > 0 ? 'font-semibold text-indigo-300' : 'font-semibold text-rose-400'">
            {{ quota.remaining }}
          </span>
          / {{ quota.limit }} 次对话
        </span>
        <span v-else-if="quota">每日对话次数不限</span>
        <span v-else>&nbsp;</span>
        <span class="hidden sm:inline">按电脑/IP 计数，每日 0 点重置</span>
      </div>
      <TaskInput
        :model-value="task"
        placeholder="输入任务，例如：帮我计算 (137×0.85−20)÷3 等于多少"
        @update:model-value="emit('update:task', $event)"
        @submit="onSend"
      />
      <div class="mt-3 flex gap-2">
        <button
          type="button"
          :disabled="sending || !task.trim()"
          class="flex-1 rounded-xl bg-gradient-to-r from-indigo-500 to-fuchsia-500 py-2.5 text-sm font-semibold text-white transition enabled:hover:opacity-90 disabled:opacity-40"
          @click="onSend"
        >
          {{ sending ? '执行中…' : '发送任务' }}
        </button>
        <button
          v-if="sending"
          type="button"
          class="rounded-xl border border-rose-500/50 px-4 py-2.5 text-sm font-semibold text-rose-300 transition enabled:hover:bg-rose-500/10"
          @click="props.stream.stop()"
        >
          停止执行
        </button>
      </div>
    </div>
  </div>
</template>
