<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { MODES, modeMeta } from '../data/techModules'
import { useChatStream } from '../composables/useChatStream'
import type { ApprovalPolicy, PromptStrategy } from '../types/agent'
import LiveStage from '../components/LiveStage.vue'
import StepTimeline from '../components/StepTimeline.vue'
import FlowDiagram from '../components/FlowDiagram.vue'
import CodeSnippet from '../components/CodeSnippet.vue'
import PrinciplePanel from '../components/PrinciplePanel.vue'
import TaskInput from '../components/TaskInput.vue'

const route = useRoute()
const meta = computed(() => modeMeta(String(route.params.id)) ?? MODES[0])
const stream = useChatStream()

const task = ref('')
const strategy = ref<PromptStrategy>('standard')
const policy = ref<ApprovalPolicy>('always')

const sending = computed(() => stream.status === 'streaming' || stream.status === 'waiting_approval')

function run() {
  const msg = task.value.trim()
  if (!msg || sending.value) return
  void stream.send({
    message: msg,
    mode: meta.value.id,
    enabled: ['calculator', 'time_now'],
    strategy: strategy.value,
    policy: policy.value,
  })
}

function useExample() {
  task.value = meta.value.defaultPrompt
  run()
}

onMounted(() => {
  task.value = meta.value.defaultPrompt
})
</script>

<template>
  <div class="mx-auto max-w-7xl px-4 py-8">
    <header class="mb-6">
      <div class="flex items-center gap-2 text-xs text-slate-500">
        <RouterLink to="/" class="transition hover:text-indigo-300">首页</RouterLink>
        <span>/</span>
        <span class="text-slate-300">{{ meta.name }}</span>
      </div>
      <h2 class="mt-2 text-2xl font-bold text-white">{{ meta.name }}</h2>
      <p class="mt-1 text-sm text-slate-400">{{ meta.tagline }} · {{ meta.description }}</p>
      <div class="mt-3 flex flex-wrap gap-2">
        <span
          v-for="t in meta.tags"
          :key="t"
          class="rounded-full border border-slate-700 bg-slate-800/50 px-2.5 py-0.5 text-xs text-slate-300"
        >
          {{ t }}
        </span>
      </div>
    </header>

    <div class="grid gap-6 lg:grid-cols-3">
      <!-- 左：实时执行 + 源码 -->
      <div class="space-y-4 lg:col-span-2">
        <div class="rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
          <div class="mb-2 flex items-center justify-between">
            <h3 class="text-sm font-semibold text-white">实时执行</h3>
            <div class="flex items-center gap-2 text-xs">
              <select
                v-model="strategy"
                class="rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 outline-none focus:border-indigo-400"
              >
                <option value="standard">策略：standard</option>
                <option value="few_shot">策略：few_shot</option>
                <option value="cot">策略：cot</option>
              </select>
              <select
                v-model="policy"
                class="rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 outline-none focus:border-indigo-400"
              >
                <option value="always">审批：always</option>
                <option value="never">审批：never</option>
              </select>
            </div>
          </div>
          <TaskInput v-model="task" :placeholder="meta.defaultPrompt" @submit="run" />
          <div class="mt-3 flex gap-2">
            <button
              type="button"
              :disabled="sending || !task.trim()"
              class="rounded-xl bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-4 py-2 text-sm font-semibold text-white transition enabled:hover:opacity-90 disabled:opacity-40"
              @click="run"
            >
              {{ sending ? '执行中…' : '运行演示' }}
            </button>
            <button
              type="button"
              :disabled="sending"
              class="rounded-xl border border-slate-700 px-4 py-2 text-sm text-slate-300 transition hover:border-slate-500 disabled:opacity-40"
              @click="useExample"
            >
              运行默认示例
            </button>
            <button
              v-if="sending"
              type="button"
              class="rounded-xl border border-rose-500/50 px-4 py-2 text-sm font-semibold text-rose-300 transition hover:bg-rose-500/10"
              @click="stream.stop()"
            >
              停止执行
            </button>
          </div>
        </div>

        <LiveStage :stream="stream" />
        <CodeSnippet :code-key="meta.id" />
      </div>

      <!-- 右：流程图 + 步骤 + 原理 -->
      <div class="space-y-6">
        <div class="rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
          <h3 class="mb-3 text-sm font-semibold text-white">模式流程图</h3>
          <FlowDiagram :mode="meta" />
        </div>
        <div class="rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
          <h3 class="mb-3 text-sm font-semibold text-white">执行步骤</h3>
          <StepTimeline :steps="meta.steps" />
        </div>
        <PrinciplePanel :topic="meta.id" />
      </div>
    </div>
  </div>
</template>
