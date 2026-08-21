<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useCapabilities } from '../composables/useCapabilities'
import { useChatStream } from '../composables/useChatStream'
import type { Capability, ModeId, PromptStrategy, ApprovalPolicy } from '../types/agent'
import CapabilitySidebar from '../components/CapabilitySidebar.vue'
import ChatPanel from '../components/ChatPanel.vue'
import ExampleFillHint from '../components/ExampleFillHint.vue'
import SandboxFilesPanel from '../components/SandboxFilesPanel.vue'

const {
  caps,
  enabled,
  loading,
  loadError,
  exampleHint,
  enabledCapabilities,
  load,
  toggle,
  applyExample,
  clearHint,
} = useCapabilities()
const stream = useChatStream()

const task = ref('')
const mode = ref<ModeId>('react')
const strategy = ref<PromptStrategy>('standard')
const policy = ref<ApprovalPolicy>('always')
const sidebarOpen = ref(false)
const filesOpen = ref(false)
const filesRefreshKey = ref(0)

onMounted(() => {
  load()
})

watch(exampleHint, (h) => {
  if (h) {
    task.value = h.cap.example
    window.setTimeout(() => {
      if (exampleHint.value?.nonce === h.nonce) clearHint()
    }, 6000)
  }
})

const sending = ref(false)
watch(
  () => stream.status,
  (s) => {
    sending.value = s === 'streaming' || s === 'waiting_approval'
  },
  { immediate: true },
)

// run_command 每次执行结束（成功/失败）后刷新沙箱文件列表，便于立即下载产物
let lastCmdEndCount = 0
watch(
  () =>
    stream.steps.filter(
      (s) => s.kind === 'tool' && s.tool === 'run_command' && s.status !== 'running',
    ).length,
  (n) => {
    if (n > lastCmdEndCount) {
      lastCmdEndCount = n
      filesRefreshKey.value++
    }
  },
)

function onExample(cap: Capability) {
  applyExample(cap)
}

function send() {
  if (!task.value.trim() || sending.value) return
  clearHint()
  void stream.send({
    message: task.value,
    mode: mode.value,
    enabled: [...enabled.value],
    strategy: strategy.value,
    policy: policy.value,
  })
  task.value = '' // 发送后清空输入框
}
</script>

<template>
  <div class="flex h-full">
    <CapabilitySidebar
      :caps="caps"
      :enabled-ids="enabled"
      :loading="loading"
      :error="loadError"
      :mode="mode"
      :strategy="strategy"
      :policy="policy"
      :open="sidebarOpen"
      @toggle="toggle"
      @example="onExample"
      @update:mode="mode = $event"
      @update:strategy="strategy = $event"
      @update:policy="policy = $event"
      @close="sidebarOpen = false"
    />

    <div class="flex flex-1 flex-col">
      <div class="flex items-center gap-3 border-b border-slate-800 px-4 py-2 md:hidden">
        <button
          type="button"
          class="rounded-lg border border-slate-700 p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
          @click="sidebarOpen = true"
        >
          <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
          </svg>
        </button>
        <span class="text-sm font-medium text-white">能力选配</span>
      </div>

      <ChatPanel
        :stream="stream"
        v-model:task="task"
        :mode="mode"
        :strategy="strategy"
        :policy="policy"
        :sending="sending"
        :enabled-capabilities="enabledCapabilities"
        :files-open="filesOpen"
        @send="send"
        @toggle-files="filesOpen = $event"
      />

      <div class="pointer-events-none absolute left-4 top-14 z-10 md:left-80 md:top-4">
        <div class="pointer-events-auto inline-block">
          <ExampleFillHint :cap="exampleHint?.cap ?? null" @close="clearHint" />
        </div>
      </div>
    </div>

    <div
      v-if="sidebarOpen"
      class="fixed inset-0 z-20 bg-black/50 md:hidden"
      @click="sidebarOpen = false"
    ></div>

    <div
      v-if="filesOpen"
      class="fixed inset-0 z-20 bg-black/50"
      @click="filesOpen = false"
    ></div>

    <SandboxFilesPanel
      :open="filesOpen"
      :refresh-key="filesRefreshKey"
      @close="filesOpen = false"
    />
  </div>
</template>
