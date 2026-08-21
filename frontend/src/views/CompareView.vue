<script setup lang="ts">
import { computed, ref } from 'vue'
import { useDemoRunner } from '../composables/useDemoRunner'
import type { ApprovalPolicy, ModeId, PromptStrategy } from '../types/agent'
import { MODES } from '../data/techModules'
import LiveStage from '../components/LiveStage.vue'
import TaskInput from '../components/TaskInput.vue'

const { runners, running, start, stopAll } = useDemoRunner()

const task = ref('')
const selected = ref<ModeId[]>(['react', 'plan_execute'])
const strategy = ref<PromptStrategy>('standard')
const policy = ref<ApprovalPolicy>('always')

const PRESETS = [
  '帮我计算 (137×0.85−20)÷3 等于多少',
  '现在几点？今天是几号？',
  '搜索一下 Qwen3 的发布时间',
  '写一句介绍 AI Agent 的话，并反思修订',
]

const canStart = computed(() => task.value.trim().length > 0 && selected.value.length >= 2)

function toggleMode(m: ModeId) {
  if (selected.value.includes(m)) {
    if (selected.value.length > 2) {
      selected.value = selected.value.filter((x) => x !== m)
    }
  } else {
    selected.value = [...selected.value, m]
  }
}

function startRun() {
  if (!canStart.value || running.value) return
  void start({
    task: task.value,
    modes: selected.value,
    enabled: ['calculator', 'time_now'],
    strategy: strategy.value,
    policy: policy.value,
  })
  task.value = '' // 发送后清空输入框
}

function fillPreset(t: string) {
  task.value = t
}
</script>

<template>
  <div class="mx-auto max-w-7xl px-4 py-8">
    <header class="mb-6">
      <h2 class="text-2xl font-bold text-white">多模式对比</h2>
      <p class="mt-2 text-sm text-slate-400">同一任务在多种模式下并行运行，并排对比各自的推理过程与最终结果。</p>
    </header>

    <section class="mb-6 rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
      <div class="grid gap-4 lg:grid-cols-2">
        <div>
          <label class="mb-1.5 block text-xs text-slate-400">任务</label>
          <TaskInput v-model="task" @submit="startRun" />
          <div class="mt-2 flex flex-wrap gap-2">
            <button
              v-for="t in PRESETS"
              :key="t"
              type="button"
              class="rounded-lg border border-slate-700 px-2.5 py-1 text-xs text-slate-300 transition hover:border-indigo-400 hover:text-white"
              @click="fillPreset(t)"
            >
              预设
            </button>
          </div>
        </div>
        <div>
          <label class="mb-1.5 block text-xs text-slate-400">选择 2-3 种模式</label>
          <div class="flex flex-wrap gap-2">
            <button
              v-for="m in MODES"
              :key="m.id"
              type="button"
              class="rounded-xl border px-3 py-2 text-sm transition"
              :class="
                selected.includes(m.id)
                  ? 'border-indigo-400 bg-indigo-500/20 text-white'
                  : 'border-slate-700 bg-slate-800/50 text-slate-400 hover:border-slate-500 hover:text-slate-200'
              "
              @click="toggleMode(m.id)"
            >
              {{ m.name }}
              <span class="ml-1 text-[11px] text-slate-500">{{ m.tagline }}</span>
            </button>
          </div>
          <div class="mt-3 flex flex-wrap items-center gap-3">
            <label class="text-xs text-slate-400">
              策略
              <select
                v-model="strategy"
                class="ml-1 rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 outline-none focus:border-indigo-400"
              >
                <option value="standard">standard</option>
                <option value="few_shot">few_shot</option>
                <option value="cot">cot</option>
              </select>
            </label>
            <label class="text-xs text-slate-400">
              审批
              <select
                v-model="policy"
                class="ml-1 rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 outline-none focus:border-indigo-400"
              >
                <option value="always">always</option>
                <option value="never">never</option>
              </select>
            </label>
          </div>
        </div>
      </div>

      <div class="mt-4 flex items-center gap-3">
        <button
          type="button"
          :disabled="!canStart || running"
          class="rounded-xl bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-5 py-2.5 text-sm font-semibold text-white transition enabled:hover:opacity-90 disabled:opacity-40"
          @click="startRun"
        >
          {{ running ? '运行中…' : '并行运行' }}
        </button>
        <button
          type="button"
          :disabled="!running"
          class="rounded-xl border border-rose-500/50 px-5 py-2.5 text-sm text-rose-300 transition enabled:hover:bg-rose-500/10 disabled:opacity-40"
          @click="stopAll"
        >
          停止
        </button>
        <span class="text-xs text-slate-500">已选 {{ selected.length }} 种模式</span>
      </div>
    </section>

    <section class="grid gap-5 xl:grid-cols-2 2xl:grid-cols-3">
      <LiveStage v-for="r in runners" :key="r.mode" :stream="r.stream" />
    </section>
    <p v-if="!runners.length" class="py-10 text-center text-sm text-slate-500">
      选择 2-3 种模式并点击「并行运行」开始对比
    </p>
  </div>
</template>
