<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import LiveStage from './LiveStage.vue'
import TaskInput from './TaskInput.vue'
import type { ApprovalPolicy, Capability, ChatStream, ModeId, PromptStrategy } from '../types/agent'

const props = defineProps<{
  stream: ChatStream
  task: string
  mode: ModeId
  strategy: PromptStrategy
  policy: ApprovalPolicy
  sending: boolean
  enabledCapabilities: Capability[]
}>()

const emit = defineEmits<{
  'update:task': [v: string]
  send: []
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
    </div>

    <div ref="stageEl" class="flex-1 overflow-y-auto p-4">
      <LiveStage :stream="stream" />
    </div>

    <div class="border-t border-slate-800 p-4">
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
