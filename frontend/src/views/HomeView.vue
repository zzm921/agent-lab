<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useCapabilities } from '../composables/useCapabilities'
import { useChatStream } from '../composables/useChatStream'
import { useSessions } from '../composables/useSessions'
import { LAB_PRESET_STORAGE_KEY } from '../data/capabilityData'
import type { Capability, ModeId, PromptStrategy, ApprovalPolicy, RagSchemeId } from '../types/agent'
import CapabilitySidebar from '../components/CapabilitySidebar.vue'
import ChatPanel from '../components/ChatPanel.vue'
import ExampleFillHint from '../components/ExampleFillHint.vue'
import MemoryPanel from '../components/MemoryPanel.vue'
import RunRecordsPanel from '../components/RunRecordsPanel.vue'
import SandboxFilesPanel from '../components/SandboxFilesPanel.vue'
import SessionSidebar from '../components/SessionSidebar.vue'

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
/** 多会话管理：会话列表 / 当前会话（后端分配 id，localStorage 恢复） */
const sessions = useSessions()

const task = ref('')
const validModes: ModeId[] = ['react', 'plan_execute', 'reflection', 'multi_agent']
const validStrategies: PromptStrategy[] = ['standard', 'few_shot', 'cot']
const validPolicies: ApprovalPolicy[] = ['always', 'never']
const validRagSchemes: RagSchemeId[] = ['naive', 'advanced', 'modular', 'agentic']
const mode = ref<ModeId>(validModes.includes(route.query.mode as ModeId) ? (route.query.mode as ModeId) : 'react')
const strategy = ref<PromptStrategy>(validStrategies.includes(route.query.strategy as PromptStrategy) ? (route.query.strategy as PromptStrategy) : 'standard')
const policy = ref<ApprovalPolicy>(validPolicies.includes(route.query.policy as ApprovalPolicy) ? (route.query.policy as ApprovalPolicy) : 'always')
const hasRagScheme = validRagSchemes.includes(route.query.rag_scheme as RagSchemeId)
const ragScheme = ref<RagSchemeId>(hasRagScheme ? (route.query.rag_scheme as RagSchemeId) : 'naive')
/** 知识库检索开关：默认关闭，需在能力选配中手动开启；若 URL 已携带 rag_scheme（如从能力卡片带方案进入）则自动开启 */
const ragEnabled = ref(hasRagScheme)
/** 长期记忆开关：默认开启（常驻注入 + 主动召回 + 轮末巩固） */
const memoryEnabled = ref(true)
/** 记忆管理面板开关 */
const memoryOpen = ref(false)
/** 运行记录面板开关（可观测性演示） */
const runsOpen = ref(false)
/** 「每轮压缩」演示：保留最近 N 轮对话原文，更早历史每轮被压缩；0 关闭（用系统默认阈值） */
const keepRounds = ref(3)
const sidebarOpen = ref(false)
/** 会话侧边栏开关（移动端抽屉） */
const sessionsRailOpen = ref(false)
const filesOpen = ref(false)
const filesRefreshKey = ref(0)
/** 输入框下方的快捷 Prompt：跳转卡片配置的 prompts 列表（content 驱动） */
const presetPrompts = ref<string[]>([])

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
  // 预设任务：优先从 sessionStorage 读取跳转卡片携带的 prompts（避免长文本进 URL）
  // 读取顺序：sessionStorage(jump) → ?prompts=（分享/兜底） → 旧版 ?prompt=
  const jump = route.query.jump
  let jumpedPrompts: string[] = []
  if (jump) {
    try {
      const stored = JSON.parse(sessionStorage.getItem(LAB_PRESET_STORAGE_KEY) ?? 'null')
      if (stored && stored.nonce === String(jump) && Array.isArray(stored.prompts)) {
        jumpedPrompts = stored.prompts.filter((p: unknown): p is string => typeof p === 'string')
      }
    } catch {
      /* sessionStorage 异常时忽略，走 URL 兜底 */
    }
  }
  const promptsParam = route.query.prompts
  if (jumpedPrompts.length) {
    presetPrompts.value = jumpedPrompts
    // 自动填入第一条，可直接发送
    task.value = presetPrompts.value[0]
  } else if (promptsParam) {
    presetPrompts.value = String(promptsParam)
      .split('\n')
      .map((p) => p.trim())
      .filter(Boolean)
    if (presetPrompts.value.length) {
      task.value = presetPrompts.value[0]
    }
  } else if (route.query.prompt) {
    task.value = String(route.query.prompt)
    presetPrompts.value = [task.value]
  }
  // 会话恢复：加载会话列表；localStorage 中的上次会话已失效（TTL 清理/删除）时新建
  await sessions.load()
  const restored = sessions.sessions.value.some((s) => s.session_id === sessions.currentId.value)
  if (!restored) await sessions.create()
  stream.sessionId = sessions.currentId.value
  // 恢复历史：刷新/重启后按已落盘事件流重放，看到上次会话的完整流水线
  stream.replay(await sessions.loadEvents(sessions.currentId.value))
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
    sending.value = s === 'streaming' || s === 'waiting_approval' || s === 'waiting_user'
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

async function send() {
  if (!task.value.trim() || sending.value) return
  clearHint()
  // 兜底：极少数情况（后端会话初始化失败）下先补建会话，保证本轮落在某个持久化会话内
  if (!sessions.currentId.value) {
    try {
      await sessions.create()
      stream.sessionId = sessions.currentId.value
    } catch {
      /* 仍失败则让后端自行分配（仅内存兜底） */
    }
  }
  void stream.send({
    message: task.value,
    mode: mode.value,
    enabled: [...enabled.value],
    strategy: strategy.value,
    policy: policy.value,
    ragScheme: ragScheme.value,
    ragEnabled: ragEnabled.value,
    memoryEnabled: memoryEnabled.value,
    contextKeepRounds: keepRounds.value,
    sessionId: sessions.currentId.value || undefined,
  })
  task.value = '' // 发送后清空输入框
}

/** 清空面板瞬态状态（切换/新建会话时调用，历史保留在后端，按 session_id 恢复） */
function resetPanel() {
  stream.done = null
  stream.error = null
  stream.guard = null
  stream.approval = null
  stream.askUser = null
  stream.elapsed = 0
  stream.steps.splice(0, stream.steps.length)
}

async function newSession() {
  stream.stop()
  resetPanel()
  try {
    await sessions.create()
    stream.sessionId = sessions.currentId.value
  } catch (e) {
    sessions.sessionsError.value = e instanceof Error ? e.message : String(e)
  }
}

async function selectSession(id: string) {
  if (sessions.currentId.value === id) return
  stream.stop()
  resetPanel()
  sessions.switchTo(id)
  stream.sessionId = id
  // 切换会话：按该会话已落盘的事件流重放历史
  stream.replay(await sessions.loadEvents(id))
}

async function renameSession(id: string, title: string) {
  try {
    await sessions.rename(id, title)
  } catch (e) {
    sessions.sessionsError.value = e instanceof Error ? e.message : String(e)
  }
}

async function removeSession(id: string) {
  const wasCurrent = sessions.currentId.value === id
  if (wasCurrent) {
    stream.stop()
    resetPanel()
  }
  try {
    await sessions.remove(id)
    // 删除当前会话后 useSessions 已自动新建空会话，同步流状态
    if (wasCurrent) stream.sessionId = sessions.currentId.value
  } catch (e) {
    sessions.sessionsError.value = e instanceof Error ? e.message : String(e)
  }
}
</script>

<template>
  <div class="flex h-full w-full">
    <!-- 运行记录：可观测性演示入口（右上角常驻，显眼） -->
    <button
      type="button"
      class="fixed right-4 top-4 z-40 flex items-center gap-1.5 rounded-xl border border-indigo-500/40 bg-slate-900/90 px-3 py-2 text-xs font-semibold text-indigo-200 shadow-lg backdrop-blur transition hover:border-indigo-400 hover:text-white"
      title="查看本次及历史对话的运行记录（SSE 事件流 + LLM 调用明细 + 聚合统计）"
      @click="runsOpen = true"
    >
      <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M4 7V5a1 1 0 011-1h2m10 0h2a1 1 0 011 1v2m0 10v2a1 1 0 01-1 1h-2m-10 0H5a1 1 0 01-1-1v-2m2-6h4v4H7v-4zm6 0h4v4h-4v-4z" />
      </svg>
      运行记录
    </button>

    <SessionSidebar
      :sessions="sessions.sessions.value"
      :current-id="sessions.currentId.value"
      :loading="sessions.sessionsLoading.value"
      :error="sessions.sessionsError.value"
      :open="sessionsRailOpen"
      @create="newSession"
      @select="selectSession"
      @rename="renameSession"
      @remove="removeSession"
      @close="sessionsRailOpen = false"
    />

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
      :rag-enabled="ragEnabled"
      :keep-rounds="keepRounds"
      :open="sidebarOpen"
      :mcp-enabled="mcpEnabled"
      :mcp-caps="mcpCaps"
      :memory-enabled="memoryEnabled"
      @toggle="toggle"
      @example="onExample"
      @fault="setFault"
      @toggle-mcp="setMcpEnabled"
      @update:mode="mode = $event"
      @update:strategy="strategy = $event"
      @update:policy="policy = $event"
      @update:rag-scheme="ragScheme = $event"
      @update:rag-enabled="ragEnabled = $event"
      @update:keep-rounds="keepRounds = $event"
      @update:memory-enabled="memoryEnabled = $event"
      @open-memory="memoryOpen = true"
      @close="sidebarOpen = false"
    />

    <div class="flex flex-1 flex-col">
      <div class="flex items-center gap-3 border-b border-slate-800 px-4 py-2 md:hidden">
        <button
          type="button"
          class="rounded-lg border border-slate-700 p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
          title="会话列表"
          @click="sessionsRailOpen = true"
        >
          <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M8 10h8m-8 4h5M9 4h6a2 2 0 012 2v1m-9 0V6a2 2 0 012-2m0 0a2 2 0 012 2m-2 2a2 2 0 100 4m3-2a2 2 0 110 4m0 0a2 2 0 00-1.76 1.05M5 20a5 5 0 0110 0" />
          </svg>
        </button>
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
        :content-prompts="presetPrompts"
        :files-open="filesOpen"
        @send="send"
        @toggle-files="filesOpen = $event"
      />

      <div class="pointer-events-none absolute left-4 top-14 z-10 md:left-[40rem] md:top-4">
        <div class="pointer-events-auto inline-block">
          <ExampleFillHint :cap="exampleHint?.cap ?? null" @close="clearHint" />
        </div>
      </div>
    </div>

    <div
      v-if="sessionsRailOpen"
      class="fixed inset-0 z-30 bg-black/50 md:hidden"
      @click="sessionsRailOpen = false"
    ></div>

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

    <MemoryPanel
      :open="memoryOpen"
      :session-id="stream.sessionId"
      @close="memoryOpen = false"
    />

    <RunRecordsPanel
      :open="runsOpen"
      :session-id="stream.sessionId"
      @close="runsOpen = false"
    />
  </div>
</template>
