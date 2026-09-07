<script setup lang="ts">
/** 长期记忆管理面板：查看（会话/全局）、删除、手动写入（用户掌控权）。
 * 全局（常驻）记忆按设备指纹隔离：请求带 X-Client-Id，后端据此各存一份、互不可见。
 */
import { reactive, ref, watch } from 'vue'
import { getClientId } from '../services/sse'

const props = defineProps<{
  open: boolean
  sessionId: string
}>()

const emit = defineEmits<{ close: [] }>()

interface MemItem {
  id: string
  kind: string
  text: string
  importance: number
  created_at?: number
  last_access_at?: number
  access_count?: number
  scope: string
}

interface AuditItem {
  ts?: string
  ns?: string
  scope?: string
  action?: string
  kind?: string
  importance?: number
  text?: string
}

const tab = ref<'memory' | 'audit' | 'dream'>('memory')
const loading = ref(false)
const error = ref('')
const sessionItems = ref<MemItem[]>([])
const globalItems = ref<MemItem[]>([])
const auditItems = ref<AuditItem[]>([])
/** 记忆梦游：手动触发的显式巩固（提取→匹配→裁决→落库），分步展示中间结果 */
const dreaming = ref(false)
const dreamReport = ref<DreamReport | null>(null)
const tidyReport = ref<TidyReport | null>(null)
const dreamError = ref('')

interface DreamStep {
  name: string
  detail: string
}
interface DreamItem {
  text: string
  kind?: string
  importance?: number
  scope?: string
  action?: string
  reason?: string
  sim?: number | null
  zone?: string
  existing?: string | null
}
interface DreamSummary {
  transcript_messages: number
  extracted: number
  kept: number
  dropped: number
  written: number
  added: number
  merged: number
  conflicted: number
  before: { session: number; global: number }
  after: { session: number; global: number }
}
interface DreamReport {
  steps: DreamStep[]
  summary: DreamSummary
  extracted: DreamItem[]
  filtered: DreamItem[]
  matches: DreamItem[]
  judgments: DreamItem[]
  written: DreamItem[]
}

/** 整理模式（对齐 Claude Dreaming）报告：读取现有记忆 → 归并/替换/挖模式 → 直接落库 */
interface TidyAction {
  action: string
  scope: string
  ids: string[]
  text: string
  kind?: string
  importance?: number
  reason?: string
  executed?: boolean
  error?: string
}
interface TidyReport {
  steps: DreamStep[]
  summary: {
    before: { session: number; global: number }
    after: { session: number; global: number }
    merge: number
    replace: number
    pattern: number
  }
  actions: TidyAction[]
}

/** 整理动作标签 */
const TIDY_ACTION_LABEL: Record<string, string> = {
  merge: '合并重复',
  replace: '替换过时',
  pattern: '挖新模式',
}

const TIDY_ACTION_CLS: Record<string, string> = {
  merge: 'bg-sky-500/15 text-sky-300',
  replace: 'bg-amber-500/15 text-amber-300',
  pattern: 'bg-violet-500/15 text-violet-300',
}

const form = reactive({
  text: '',
  kind: 'fact',
  importance: 0.7,
  scope: 'session',
})

const KIND_LABEL: Record<string, string> = {
  fact: '事实',
  preference: '偏好',
  episodic: '事件',
  procedural: '经验',
}

const ACTION_LABEL: Record<string, string> = {
  add: '新增',
  update: '更新',
  delete: '删除',
}

/** 梦游动作标签：add/merge/conflict/error */
const DREAM_ACTION_LABEL: Record<string, string> = {
  add: '新增',
  merge: '合并',
  conflict: '改口',
  error: '失败',
}

const DREAM_ACTION_CLS: Record<string, string> = {
  add: 'bg-emerald-500/15 text-emerald-300',
  merge: 'bg-sky-500/15 text-sky-300',
  conflict: 'bg-amber-500/15 text-amber-300',
  error: 'bg-rose-500/15 text-rose-300',
}

/** 匹配区带标签：none（无匹配）/ low（低相似）/ ambiguous（模糊带）/ high（高相似） */
const ZONE_LABEL: Record<string, string> = {
  none: '无匹配',
  low: '低相似',
  ambiguous: '模糊带',
  high: '高相似',
}

const ACTION_CLS: Record<string, string> = {
  add: 'bg-emerald-500/15 text-emerald-300',
  update: 'bg-amber-500/15 text-amber-300',
  delete: 'bg-rose-500/15 text-rose-300',
}

const SCOPE_LABEL: Record<string, string> = {
  session: '会话',
  global: '全局',
}

function fmtDate(ts?: number) {
  if (!ts) return '?'
  const d = new Date(ts * 1000)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

/** 请求头：附带设备指纹 X-Client-Id，供后端按「一台电脑」隔离常驻记忆 */
function clientHeaders(extra?: Record<string, string>): Record<string, string> {
  const id = getClientId()
  return { ...(id ? { 'X-Client-Id': id } : {}), ...extra }
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const [s, g] = await Promise.all([
      fetch(`/api/memory?scope=session&session_id=${encodeURIComponent(props.sessionId)}`, {
        headers: clientHeaders(),
      }).then((r) => r.json()),
      fetch('/api/memory?scope=global', { headers: clientHeaders() }).then((r) => r.json()),
    ])
    sessionItems.value = s.items ?? []
    globalItems.value = g.items ?? []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

async function del(item: MemItem) {
  if (!window.confirm(`删除这条记忆？\n${item.text}`)) return
  await fetch(`/api/memory/${item.id}?scope=${item.scope}&session_id=${encodeURIComponent(props.sessionId)}`, {
    method: 'DELETE',
    headers: clientHeaders(),
  })
  await load()
}

async function write() {
  const text = form.text.trim()
  if (!text) return
  const body = {
    text,
    kind: form.kind,
    importance: form.importance,
    scope: form.scope,
    session_id: form.scope === 'session' ? props.sessionId : '',
  }
  const resp = await fetch('/api/memory', {
    method: 'POST',
    headers: clientHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    error.value = (await resp.json()).detail ?? '写入失败'
    return
  }
  form.text = ''
  await load()
}

async function loadAudit() {
  try {
    const r = await fetch('/api/memory/audit?limit=100', { headers: clientHeaders() }).then((res) => res.json())
    auditItems.value = r.items ?? []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}

/** 触发记忆梦游：对话提炼（extract）或整理现有记忆（tidy），展示每步中间结果 */
const dreamMode = ref<'extract' | 'tidy'>('extract')
async function dream() {
  dreaming.value = true
  dreamError.value = ''
  dreamReport.value = null
  tidyReport.value = null
  try {
    const resp = await fetch(`/api/memory/dream?session_id=${encodeURIComponent(props.sessionId)}&mode=${dreamMode.value}`, {
      method: 'POST',
      headers: clientHeaders(),
    })
    const data = await resp.json()
    if (!resp.ok) {
      dreamError.value = data.detail ?? '梦游失败'
      return
    }
    if (data.mode === 'tidy') tidyReport.value = data
    else dreamReport.value = data
  } catch (e) {
    dreamError.value = e instanceof Error ? e.message : String(e)
  } finally {
    dreaming.value = false
  }
}

function switchTab(next: 'memory' | 'audit' | 'dream') {
  tab.value = next
  if (next === 'audit' && !auditItems.value.length) void loadAudit()
}

// 组件常驻 DOM（内部 v-if 控制显示），需监听 open 变化：打开面板即自动加载数据
watch(
  () => props.open,
  (v) => {
    if (!v) return
    void load()
    if (tab.value === 'audit' && !auditItems.value.length) void loadAudit()
  },
  { immediate: true },
)
</script>

<template>
  <div v-if="open" class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" @click.self="emit('close')">
    <div class="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl">
      <div class="flex items-center justify-between border-b border-slate-800 px-5 py-3">
        <div>
          <h2 class="text-sm font-semibold text-white">记忆管理</h2>
          <p class="text-[11px] text-slate-500">查看 / 删除 / 手动写入长期记忆（会话 + 全局常驻）</p>
        </div>
        <button type="button" class="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white" @click="emit('close')">
          <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div class="flex gap-1 border-b border-slate-800 px-5 pt-2">
        <button
          type="button"
          class="rounded-t-lg px-3 py-1.5 text-xs transition"
          :class="tab === 'memory' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'"
          @click="switchTab('memory')"
        >
          记忆
        </button>
        <button
          type="button"
          class="rounded-t-lg px-3 py-1.5 text-xs transition"
          :class="tab === 'audit' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'"
          @click="switchTab('audit')"
        >
          审计流水
        </button>
        <button
          type="button"
          class="rounded-t-lg px-3 py-1.5 text-xs transition"
          :class="tab === 'dream' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'"
          @click="switchTab('dream')"
        >
          记忆梦游
        </button>
      </div>

      <div class="flex-1 space-y-5 overflow-y-auto px-5 py-4">
        <template v-if="tab === 'memory'">
          <p v-if="error" class="rounded-lg bg-rose-500/10 px-3 py-2 text-xs text-rose-300">{{ error }}</p>

          <!-- 手动写入 -->
          <section class="rounded-xl border border-slate-800 p-3">
            <h3 class="mb-2 text-xs font-semibold text-amber-300">手动写入</h3>
            <div class="space-y-2">
              <input
                v-model="form.text"
                type="text"
                placeholder="要记住的事实，如：用户喜欢深色主题"
                class="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white outline-none focus:border-amber-500/50"
                @keyup.enter="write"
              />
              <div class="flex flex-wrap items-center gap-2">
                <select v-model="form.kind" class="rounded-lg border border-slate-700 bg-slate-800 px-2 py-1.5 text-xs text-white outline-none">
                  <option value="fact">事实</option>
                  <option value="preference">偏好</option>
                  <option value="episodic">事件</option>
                  <option value="procedural">经验</option>
                </select>
                <select v-model="form.scope" class="rounded-lg border border-slate-700 bg-slate-800 px-2 py-1.5 text-xs text-white outline-none">
                  <option value="session">会话记忆</option>
                  <option value="global">全局常驻</option>
                </select>
                <label class="flex items-center gap-1.5 text-[11px] text-slate-400">
                  重要度
                  <input v-model.number="form.importance" type="range" min="0" max="1" step="0.05" class="w-24" />
                  <span class="w-8 text-right text-slate-200">{{ form.importance.toFixed(2) }}</span>
                </label>
                <button
                  type="button"
                  class="ml-auto rounded-lg bg-amber-500 px-3 py-1.5 text-xs font-semibold text-slate-900 hover:bg-amber-400"
                  @click="write"
                >
                  写入
                </button>
              </div>
            </div>
          </section>

          <!-- 全局常驻记忆 -->
          <section>
            <h3 class="mb-2 text-xs font-semibold text-slate-300">全局常驻记忆（会话启动注入 system）</h3>
            <div v-if="globalItems.length" class="space-y-1.5">
              <div
                v-for="it in globalItems"
                :key="it.id"
                class="flex items-start justify-between gap-2 rounded-lg border border-slate-800 bg-slate-800/40 px-3 py-2"
              >
                <div class="min-w-0">
                  <p class="text-sm text-slate-200">{{ it.text }}</p>
                  <p class="mt-0.5 text-[10px] text-slate-500">
                    <span class="text-amber-300/90">{{ KIND_LABEL[it.kind] ?? it.kind }}</span>
                    · 重要度 {{ it.importance.toFixed(1) }} · 记录于 {{ fmtDate(it.created_at) }} · 访问 {{ it.access_count ?? 0 }} 次
                  </p>
                </div>
                <button type="button" class="shrink-0 rounded p-1 text-slate-500 hover:bg-rose-500/10 hover:text-rose-300" title="删除" @click="del(it)">
                  <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M6 7h12M9 7V5a1 1 0 011-1h4a1 1 0 011 1v2m1 0v12a1 1 0 01-1 1H8a1 1 0 01-1-1V7m4 4v6m4-6v6" />
                  </svg>
                </button>
              </div>
            </div>
            <p v-else class="rounded-xl border border-dashed border-slate-700/70 px-3 py-4 text-center text-[11px] text-slate-600">
              暂无全局常驻记忆 — 可用上方表单写入，或在对话中让助手记住（轮末自动巩固提取）
            </p>
          </section>

          <!-- 会话记忆 -->
          <section>
            <h3 class="mb-2 text-xs font-semibold text-slate-300">会话记忆（{{ sessionId.slice(0, 8) }}…）</h3>
            <div v-if="sessionItems.length" class="space-y-1.5">
              <div
                v-for="it in sessionItems"
                :key="it.id"
                class="flex items-start justify-between gap-2 rounded-lg border border-slate-800 bg-slate-800/40 px-3 py-2"
              >
                <div class="min-w-0">
                  <p class="text-sm text-slate-200">{{ it.text }}</p>
                  <p class="mt-0.5 text-[10px] text-slate-500">
                    <span class="text-amber-300/90">{{ KIND_LABEL[it.kind] ?? it.kind }}</span>
                    · 重要度 {{ it.importance.toFixed(1) }} · 记录于 {{ fmtDate(it.created_at) }} · 访问 {{ it.access_count ?? 0 }} 次
                  </p>
                </div>
                <button type="button" class="shrink-0 rounded p-1 text-slate-500 hover:bg-rose-500/10 hover:text-rose-300" title="删除" @click="del(it)">
                  <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M6 7h12M9 7V5a1 1 0 011-1h4a1 1 0 011 1v2m1 0v12a1 1 0 01-1 1H8a1 1 0 01-1-1V7m4 4v6m4-6v6" />
                  </svg>
                </button>
              </div>
            </div>
            <p v-else class="rounded-xl border border-dashed border-slate-700/70 px-3 py-4 text-center text-[11px] text-slate-600">
              本会话暂无记忆 — 对话后由轮末巩固自动提取
            </p>
          </section>

          <p v-if="loading" class="py-2 text-center text-[11px] text-slate-500">加载中…</p>
          <button
            type="button"
            class="w-full rounded-lg border border-slate-700 py-1.5 text-xs text-slate-400 hover:bg-slate-800 hover:text-white"
            @click="load"
          >
            刷新
          </button>
        </template>

        <!-- 审计流水：所有新增/更新/删除操作，最新在前 -->
        <template v-else-if="tab === 'audit'">
          <p v-if="error" class="rounded-lg bg-rose-500/10 px-3 py-2 text-xs text-rose-300">{{ error }}</p>
          <section>
            <h3 class="mb-2 text-xs font-semibold text-slate-300">记忆操作审计（新增 / 更新 / 删除，最新在前）</h3>
            <div v-if="auditItems.length" class="space-y-1.5">
              <div
                v-for="(a, i) in auditItems"
                :key="i"
                class="rounded-lg border border-slate-800 bg-slate-800/40 px-3 py-2"
              >
                <p class="text-sm text-slate-200">{{ a.text }}</p>
                <p class="mt-0.5 text-[10px] text-slate-500">
                  <span class="rounded px-1" :class="ACTION_CLS[a.action ?? ''] ?? 'bg-slate-600/30 text-slate-300'">
                    {{ ACTION_LABEL[a.action ?? ''] ?? a.action }}
                  </span>
                  <span class="ml-1" :class="a.scope === 'global' ? 'text-amber-300/90' : 'text-slate-400'">
                    {{ SCOPE_LABEL[a.scope ?? ''] ?? a.scope }}
                  </span>
                  · {{ KIND_LABEL[a.kind ?? ''] ?? a.kind }} · 重要度 {{ (a.importance ?? 0).toFixed(1) }} · {{ a.ts }}
                </p>
              </div>
            </div>
            <p v-else class="rounded-xl border border-dashed border-slate-700/70 px-3 py-4 text-center text-[11px] text-slate-600">
              暂无审计记录 — 写入 / 更新 / 删除记忆后自动生成
            </p>
          </section>
          <button
            type="button"
            class="w-full rounded-lg border border-slate-700 py-1.5 text-xs text-slate-400 hover:bg-slate-800 hover:text-white"
            @click="loadAudit"
          >
            刷新审计
          </button>
        </template>

        <!-- 记忆梦游：对话提炼 / 整理现有记忆，分步展示中间结果 -->
        <template v-else>
          <p v-if="dreamError" class="rounded-lg bg-rose-500/10 px-3 py-2 text-xs text-rose-300">{{ dreamError }}</p>
          <section class="rounded-xl border border-slate-800 p-3">
            <h3 class="mb-2 text-xs font-semibold text-indigo-300">记忆梦游</h3>
            <div class="mb-2 flex rounded-lg bg-slate-900 p-0.5">
              <button
                type="button"
                class="flex-1 rounded-md py-1.5 text-[11px] transition"
                :class="dreamMode === 'extract' ? 'bg-indigo-500/30 text-indigo-200' : 'text-slate-500 hover:text-slate-300'"
                @click="dreamMode = 'extract'"
              >
                对话提炼
              </button>
              <button
                type="button"
                class="flex-1 rounded-md py-1.5 text-[11px] transition"
                :class="dreamMode === 'tidy' ? 'bg-indigo-500/30 text-indigo-200' : 'text-slate-500 hover:text-slate-300'"
                @click="dreamMode = 'tidy'"
              >
                整理记忆
              </button>
            </div>
            <p v-if="dreamMode === 'extract'" class="text-[11px] leading-relaxed text-slate-500">
              把本会话完整对话交给 LLM 提炼值得长期记住的事实，与已有记忆做语义匹配，
              落入模糊带的再交 LLM 裁决（合并 / 改口 / 另存），最后落库。全程展示每步中间结果。
            </p>
            <p v-else class="text-[11px] leading-relaxed text-slate-500">
              对齐 Claude Dreaming：读取现有记忆库（会话 + 常驻）全部条目，由 LLM 找出重复归并、
              过时替换与跨条目的新模式，直接落库，展示整理前后对比。
            </p>
            <button
              type="button"
              :disabled="dreaming"
              class="mt-3 w-full rounded-lg bg-indigo-500 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
              @click="dream"
            >
              {{ dreaming ? '梦游中…' : '开始梦游' }}
            </button>
          </section>

          <template v-if="tidyReport">
            <!-- 阶段摘要 -->
            <section v-if="tidyReport.steps.length" class="rounded-xl border border-slate-800 p-3">
              <h3 class="mb-2 text-xs font-semibold text-slate-300">整理过程</h3>
              <ol class="space-y-1.5">
                <li
                  v-for="(s, i) in tidyReport.steps"
                  :key="i"
                  class="flex items-start gap-2 text-[11px] text-slate-400"
                >
                  <span class="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-violet-500/20 text-[9px] text-violet-300">
                    {{ i + 1 }}
                  </span>
                  <span><b class="text-slate-300">{{ s.name }}</b>：{{ s.detail }}</span>
                </li>
              </ol>
            </section>

            <!-- 整理结果汇总 -->
            <section class="rounded-xl border border-slate-800 p-3">
              <h3 class="mb-2 text-xs font-semibold text-slate-300">整理结果</h3>
              <div class="grid grid-cols-3 gap-2 text-[11px]">
                <div class="rounded-lg bg-slate-800/50 px-2 py-1.5 text-center">
                  <p class="text-slate-500">合并重复</p>
                  <p class="text-base font-semibold text-sky-300">{{ tidyReport.summary.merge }}</p>
                </div>
                <div class="rounded-lg bg-slate-800/50 px-2 py-1.5 text-center">
                  <p class="text-slate-500">替换过时</p>
                  <p class="text-base font-semibold text-amber-300">{{ tidyReport.summary.replace }}</p>
                </div>
                <div class="rounded-lg bg-slate-800/50 px-2 py-1.5 text-center">
                  <p class="text-slate-500">挖新模式</p>
                  <p class="text-base font-semibold text-violet-300">{{ tidyReport.summary.pattern }}</p>
                </div>
              </div>
              <div class="mt-2 rounded-lg bg-slate-800/30 px-2 py-1.5 text-[11px] text-slate-400">
                会话记忆 {{ tidyReport.summary.before.session }} → {{ tidyReport.summary.after.session }} ·
                全局常驻 {{ tidyReport.summary.before.global }} → {{ tidyReport.summary.after.global }}
              </div>
            </section>

            <!-- 整理动作明细 -->
            <section v-if="tidyReport.actions.length">
              <h3 class="mb-2 text-xs font-semibold text-slate-300">整理动作（{{ tidyReport.actions.length }}）</h3>
              <div class="space-y-1.5">
                <div
                  v-for="(a, i) in tidyReport.actions"
                  :key="'t' + i"
                  class="rounded-lg border border-slate-800 bg-slate-800/40 px-3 py-2"
                >
                  <div class="flex items-start justify-between gap-2">
                    <p class="min-w-0 text-sm text-slate-200">{{ a.text }}</p>
                    <span class="shrink-0 rounded px-1.5 py-0.5 text-[10px]" :class="TIDY_ACTION_CLS[a.action] ?? 'bg-slate-600/30 text-slate-300'">
                      {{ TIDY_ACTION_LABEL[a.action] ?? a.action }}
                    </span>
                  </div>
                  <p class="mt-0.5 text-[10px] text-slate-500">
                    {{ SCOPE_LABEL[a.scope] ?? a.scope }} · {{ KIND_LABEL[a.kind ?? ''] ?? a.kind }} · 涉及 {{ a.ids.length }} 条
                    <span v-if="a.reason" class="text-slate-400"> · {{ a.reason }}</span>
                    <span v-if="!a.executed" class="text-rose-300"> · 未执行{{ a.error ? `：${a.error}` : '' }}</span>
                  </p>
                </div>
              </div>
            </section>
            <p v-else class="rounded-xl border border-dashed border-slate-700/70 px-3 py-4 text-center text-[11px] text-slate-600">
              未发现需要整理的记忆（重复 / 过时 / 可归纳模式）
            </p>
          </template>

          <template v-if="dreamReport">
            <!-- 阶段摘要 -->
            <section v-if="dreamReport.steps.length" class="rounded-xl border border-slate-800 p-3">
              <h3 class="mb-2 text-xs font-semibold text-slate-300">梦游过程</h3>
              <ol class="space-y-1.5">
                <li
                  v-for="(s, i) in dreamReport.steps"
                  :key="i"
                  class="flex items-start gap-2 text-[11px] text-slate-400"
                >
                  <span class="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-indigo-500/20 text-[9px] text-indigo-300">
                    {{ i + 1 }}
                  </span>
                  <span><b class="text-slate-300">{{ s.name }}</b>：{{ s.detail }}</span>
                </li>
              </ol>
            </section>

            <!-- 汇总对比 -->
            <section class="rounded-xl border border-slate-800 p-3">
              <h3 class="mb-2 text-xs font-semibold text-slate-300">梦游结果</h3>
              <div class="grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-4">
                <div class="rounded-lg bg-slate-800/50 px-2 py-1.5 text-center">
                  <p class="text-slate-500">提取事实</p>
                  <p class="text-base font-semibold text-indigo-300">{{ dreamReport.summary.extracted }}</p>
                </div>
                <div class="rounded-lg bg-slate-800/50 px-2 py-1.5 text-center">
                  <p class="text-slate-500">写入</p>
                  <p class="text-base font-semibold text-emerald-300">{{ dreamReport.summary.written }}</p>
                </div>
                <div class="rounded-lg bg-slate-800/50 px-2 py-1.5 text-center">
                  <p class="text-slate-500">过滤丢弃</p>
                  <p class="text-base font-semibold text-rose-300">{{ dreamReport.summary.dropped }}</p>
                </div>
                <div class="rounded-lg bg-slate-800/50 px-2 py-1.5 text-center">
                  <p class="text-slate-500">合并 / 改口</p>
                  <p class="text-base font-semibold text-amber-300">{{ dreamReport.summary.merged }} / {{ dreamReport.summary.conflicted }}</p>
                </div>
              </div>
              <div class="mt-2 rounded-lg bg-slate-800/30 px-2 py-1.5 text-[11px] text-slate-400">
                会话记忆 {{ dreamReport.summary.before.session }} → {{ dreamReport.summary.after.session }} ·
                全局常驻 {{ dreamReport.summary.before.global }} → {{ dreamReport.summary.after.global }}
              </div>
            </section>

            <!-- 提取的候选事实 -->
            <section v-if="dreamReport.extracted.length">
              <h3 class="mb-2 text-xs font-semibold text-slate-300">提取的候选事实（{{ dreamReport.extracted.length }}）</h3>
              <div class="space-y-1.5">
                <div
                  v-for="(it, i) in dreamReport.extracted"
                  :key="'e' + i"
                  class="rounded-lg border border-slate-800 bg-slate-800/40 px-3 py-2"
                >
                  <p class="text-sm text-slate-200">{{ it.text }}</p>
                  <p class="mt-0.5 text-[10px] text-slate-500">
                    <span class="text-amber-300/90">{{ KIND_LABEL[it.kind ?? ''] ?? it.kind }}</span>
                    · 重要度 {{ (it.importance ?? 0).toFixed(2) }} · scope {{ it.scope }}
                  </p>
                </div>
              </div>
            </section>

            <!-- 匹配区带 -->
            <section v-if="dreamReport.matches.length">
              <h3 class="mb-2 text-xs font-semibold text-slate-300">与已有记忆的匹配（{{ dreamReport.matches.length }}）</h3>
              <div class="space-y-1.5">
                <div
                  v-for="(m, i) in dreamReport.matches"
                  :key="'m' + i"
                  class="rounded-lg border border-slate-800 bg-slate-800/40 px-3 py-2"
                >
                  <div class="flex items-start justify-between gap-2">
                    <p class="min-w-0 text-sm text-slate-200">{{ m.text }}</p>
                    <span class="shrink-0 rounded px-1.5 py-0.5 text-[10px]" :class="m.zone === 'ambiguous' ? 'bg-amber-500/15 text-amber-300' : m.zone === 'high' ? 'bg-sky-500/15 text-sky-300' : 'bg-slate-600/30 text-slate-300'">
                      {{ ZONE_LABEL[m.zone ?? ''] ?? m.zone }}
                    </span>
                  </div>
                  <p class="mt-0.5 text-[10px] text-slate-500">
                    相似度 {{ m.sim != null ? m.sim.toFixed(4) : '—' }} · 目标 {{ m.scope }}
                    <span v-if="m.existing" class="text-slate-400"> · 已有：{{ m.existing }}</span>
                  </p>
                </div>
              </div>
            </section>

            <!-- 模糊带裁决 -->
            <section v-if="dreamReport.judgments.length">
              <h3 class="mb-2 text-xs font-semibold text-slate-300">模糊带裁决（{{ dreamReport.judgments.length }}）</h3>
              <div class="space-y-1.5">
                <div
                  v-for="(j, i) in dreamReport.judgments"
                  :key="'j' + i"
                  class="rounded-lg border border-slate-800 bg-slate-800/40 px-3 py-2"
                >
                  <div class="flex items-start justify-between gap-2">
                    <p class="min-w-0 text-sm text-slate-200">{{ j.text }}</p>
                    <span class="shrink-0 rounded px-1.5 py-0.5 text-[10px]" :class="DREAM_ACTION_CLS[j.action ?? ''] ?? 'bg-slate-600/30 text-slate-300'">
                      {{ DREAM_ACTION_LABEL[j.action ?? ''] ?? j.action }}
                    </span>
                  </div>
                  <p class="mt-0.5 text-[10px] text-slate-500">
                    已有：{{ j.existing }}
                    <span v-if="j.reason" class="text-slate-400"> · {{ j.reason }}</span>
                  </p>
                </div>
              </div>
            </section>

            <!-- 落库明细 -->
            <section v-if="dreamReport.written.length">
              <h3 class="mb-2 text-xs font-semibold text-slate-300">落库明细（{{ dreamReport.written.length }}）</h3>
              <div class="space-y-1.5">
                <div
                  v-for="(w, i) in dreamReport.written"
                  :key="'w' + i"
                  class="rounded-lg border border-slate-800 bg-slate-800/40 px-3 py-2"
                >
                  <div class="flex items-start justify-between gap-2">
                    <p class="min-w-0 text-sm text-slate-200">{{ w.text }}</p>
                    <span class="shrink-0 rounded px-1.5 py-0.5 text-[10px]" :class="DREAM_ACTION_CLS[w.action ?? ''] ?? 'bg-slate-600/30 text-slate-300'">
                      {{ DREAM_ACTION_LABEL[w.action ?? ''] ?? w.action }}
                    </span>
                  </div>
                  <p class="mt-0.5 text-[10px] text-slate-500">
                    {{ KIND_LABEL[w.kind ?? ''] ?? w.kind }} · 重要度 {{ (w.importance ?? 0).toFixed(2) }} · 写入{{ w.scope }}
                    <span v-if="w.reason" class="text-slate-400"> · {{ w.reason }}</span>
                  </p>
                </div>
              </div>
            </section>
          </template>

          <p v-if="dreaming" class="py-2 text-center text-[11px] text-indigo-300">正在整理对话、提炼记忆…</p>
          <p v-else-if="!dreamReport && !tidyReport && !dreamError" class="rounded-xl border border-dashed border-slate-700/70 px-3 py-4 text-center text-[11px] text-slate-600">
            {{ dreamMode === 'tidy' ? '点击「开始梦游」整理现有记忆（归并重复 / 替换过时 / 挖新模式）' : '先聊几轮、再点「开始梦游」体验一次显式的记忆巩固' }}
          </p>
        </template>
      </div>
    </div>
  </div>
</template>
