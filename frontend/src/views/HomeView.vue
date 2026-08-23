<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useCapabilities } from '../composables/useCapabilities'
import { useChatStream } from '../composables/useChatStream'
import type { Capability, ModeId, PromptStrategy, ApprovalPolicy, RagSchemeId } from '../types/agent'
import CapabilitySidebar from '../components/CapabilitySidebar.vue'
import ChatPanel from '../components/ChatPanel.vue'
import ExampleFillHint from '../components/ExampleFillHint.vue'
import SandboxFilesPanel from '../components/SandboxFilesPanel.vue'

const route = useRoute()
const {
  enabled,
  loading,
  loadError,
  faults,
  faultTypes,
  mcpEnabled,
  ragSchemes,
  exampleHint,
  enabledCapabilities,
  builtinCaps,
  mcpCaps,
  load,
  setFault,
  setMcpEnabled,
  toggle,
  ensureEnabled,
  applyExample,
  clearHint,
} = useCapabilities()
const stream = useChatStream()

const task = ref('')
const validModes: ModeId[] = ['react', 'plan_execute', 'reflection', 'multi_agent']
const validStrategies: PromptStrategy[] = ['standard', 'few_shot', 'cot']
const validPolicies: ApprovalPolicy[] = ['always', 'never']
const validRagSchemes: RagSchemeId[] = ['naive', 'advanced']
const mode = ref<ModeId>(validModes.includes(route.query.mode as ModeId) ? (route.query.mode as ModeId) : 'react')
const strategy = ref<PromptStrategy>(validStrategies.includes(route.query.strategy as PromptStrategy) ? (route.query.strategy as PromptStrategy) : 'standard')
const policy = ref<ApprovalPolicy>(validPolicies.includes(route.query.policy as ApprovalPolicy) ? (route.query.policy as ApprovalPolicy) : 'always')
const ragScheme = ref<RagSchemeId>(validRagSchemes.includes(route.query.rag_scheme as RagSchemeId) ? (route.query.rag_scheme as RagSchemeId) : 'naive')
const sidebarOpen = ref(false)
const filesOpen = ref(false)
const filesRefreshKey = ref(0)

onMounted(async () => {
  await load()
  // 能力开关：?tools=a,b
  const toolsParam = route.query.tools
  if (toolsParam) {
    String(toolsParam)
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean)
      .forEach(ensureEnabled)
  }
  // 故障注入：?faults=calculator:timeout,web_search:http_500
  const faultsParam = route.query.faults
  if (faultsParam) {
    for (const pair of String(faultsParam).split(',')) {
      const [tool, type] = pair.split(':')
      if (tool && type) await setFault(tool.trim(), type.trim())
    }
  } else {
    // 未指定故障参数：清空历史遗留的全局故障，避免污染其他会话的审批与工具行为
    for (const tool of Object.keys(faults.value)) {
      await setFault(tool, 'off')
    }
  }
  // 预设任务：?prompt=...
  if (route.query.prompt) {
    task.value = String(route.query.prompt)
  }
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
    ragScheme: ragScheme.value,
  })
  task.value = '' // 发送后清空输入框
}
</script>

<template>
  <div class="flex h-full w-full">
    <CapabilitySidebar
      :caps="builtinCaps"
      :enabled-ids="enabled"
      :faults="faults"
      :fault-types="faultTypes"
      :loading="loading"
      :error="loadError"
      :mode="mode"
      :strategy="strategy"
      :policy="policy"
      :rag-scheme="ragScheme"
      :rag-schemes="ragSchemes"
      :open="sidebarOpen"
      :mcp-enabled="mcpEnabled"
      :mcp-caps="mcpCaps"
      @toggle="toggle"
      @example="onExample"
      @fault="setFault"
      @toggle-mcp="setMcpEnabled"
      @update:mode="mode = $event"
      @update:strategy="strategy = $event"
      @update:policy="policy = $event"
      @update:rag-scheme="ragScheme = $event"
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
