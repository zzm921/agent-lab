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

const tab = ref<'memory' | 'audit'>('memory')
const loading = ref(false)
const error = ref('')
const sessionItems = ref<MemItem[]>([])
const globalItems = ref<MemItem[]>([])
const auditItems = ref<AuditItem[]>([])

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

function switchTab(next: 'memory' | 'audit') {
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
        <template v-else>
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
      </div>
    </div>
  </div>
</template>
