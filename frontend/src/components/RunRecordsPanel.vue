<script setup lang="ts">
/** 运行记录面板：可观测性演示入口。
 * 每次对话自动落盘一条 run（完整 SSE 事件流 + LLM 调用明细 + 聚合统计），按会话分组回放。
 * 记录按设备指纹（X-Client-Id）隔离，只能看到本人本机的 run（与记忆隔离同源）。
 * 演示边界：仅单进程内、单次会话 run 的观测；不含分布式链路追踪 / 指标聚合看板 / 告警采样。
 */
import { computed, ref, watch } from 'vue'
import { getClientId } from '../services/sse'
import type { RunEvent, RunMeta, RunRecord } from '../types/agent'

const props = defineProps<{
  open: boolean
  sessionId: string
}>()

const emit = defineEmits<{ close: [] }>()

const loading = ref(false)
const detailLoading = ref(false)
const error = ref('')
const runs = ref<RunMeta[]>([])
const selected = ref<RunRecord | null>(null)
const expandedSeq = ref<number | null>(null)

const STATUS_LABEL: Record<string, string> = {
  done: '完成',
  pending: '审批中',
  interrupted: '中断',
  error: '错误',
}
const STATUS_CLS: Record<string, string> = {
  done: 'bg-emerald-500/15 text-emerald-300',
  pending: 'bg-amber-500/15 text-amber-300',
  interrupted: 'bg-sky-500/15 text-sky-300',
  error: 'bg-rose-500/15 text-rose-300',
}

const TYPE_LABEL: Record<string, string> = {
  meta: '元信息',
  thinking: '思考',
  message: '回复',
  revise: '修订',
  critique: '评审',
  tool_start: '工具开始',
  tool_end: '工具结束',
  tool_retry: '工具重试',
  plan: '计划',
  retrieve: '检索',
  rewrite: '改写',
  classify: '语义路由',
  decompose: '任务拆解',
  hyde: 'HyDE 假想文档',
  multi_hop_plan: '多跳规划',
  multi_hop: '多跳检索',
  multi_hop_verify: '多跳校验',
  compress: '检索压缩',
  context: '上下文管理',
  answerability: '可答性验证',
  agent_step: '检索 Agent 步骤',
  grade: '证据评审',
  correct: '纠错决策',
  verify: '答案校验',
  llm_call: 'LLM 调用',
  memory_write: '记忆写入',
  memory_read: '记忆召回',
  memory_constant: '常驻记忆',
  approval_request: '审批请求',
  reflect: '反思',
  agent_event: 'Agent 事件',
  done: '完成',
  error: '错误',
  guard_refused: '护栏拦截',
}

/** 按会话分组（最新会话在前，组内最新在前） */
const groups = computed(() => {
  const map = new Map<string, RunMeta[]>()
  for (const r of runs.value) {
    const sid = r.session_id || '未知会话'
    const list = map.get(sid) ?? []
    list.push(r)
    map.set(sid, list)
  }
  return [...map.entries()].map(([sid, items]) => ({ sessionId: sid, items }))
})

const llmCalls = computed(() =>
  (selected.value?.events ?? []).filter((e) => e.type === 'llm_call').map(llmView),
)

function llmView(ev: RunEvent) {
  const tk = (ev.tokens ?? {}) as { input?: number; output?: number; total?: number }
  return {
    model: String(ev.model ?? ''),
    scenario: String(ev.scenario ?? ''),
    method: String(ev.method ?? ''),
    latency: Number(ev.latency_ms ?? 0),
    success: ev.success === true,
    input: tk.input ?? 0,
    output: tk.output ?? 0,
    total: tk.total ?? 0,
    error: String(ev.error ?? ''),
  }
}

function clientHeaders(): Record<string, string> {
  const id = getClientId()
  return { ...(id ? { 'X-Client-Id': id } : {}) }
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const r = await fetch('/api/telemetry/runs?limit=100', { headers: clientHeaders() }).then((res) => res.json())
    runs.value = r.runs ?? []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

async function openRun(run: RunMeta) {
  expandedSeq.value = null
  detailLoading.value = true
  error.value = ''
  try {
    const resp = await fetch(`/api/telemetry/runs/${run.run_id}`, { headers: clientHeaders() })
    if (!resp.ok) {
      throw new Error(resp.status === 404 ? '记录不存在或已过期（TTL 7 天 / 上限 500 条）' : `加载失败：HTTP ${resp.status}`)
    }
    selected.value = (await resp.json()) as RunRecord
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    selected.value = null
  } finally {
    detailLoading.value = false
  }
}

function back() {
  selected.value = null
}

function fmtDuration(ms: number) {
  if (!ms) return '0ms'
  if (ms < 1000) return `${ms}ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

function fmtClock(ts: number) {
  const d = new Date(ts)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

/** start_ts = "YYYY-MM-DD HH:MM:SS"，裁剪为 MM-DD HH:MM:SS */
function fmtTs(ts?: string) {
  if (!ts) return '?'
  return ts.slice(5, 19)
}

function toggleExpand(seq: number) {
  expandedSeq.value = expandedSeq.value === seq ? null : seq
}

/** 类型化摘要：把事件压成一行可读文本（无则回退原始 JSON 片段） */
function evSummary(ev: RunEvent): string {
  switch (ev.type) {
    case 'llm_call': {
      const v = llmView(ev)
      return `${v.model} · ${v.scenario} · ${v.method} · ${v.latency}ms · ${v.success ? '成功' : '失败'} · ${v.total}t${v.error ? ` · ${v.error}` : ''}`
    }
    case 'tool_start': {
      const args = ev.args ? ` · ${JSON.stringify(ev.args).slice(0, 140)}` : ''
      return `${ev.tool}${args}`
    }
    case 'tool_end': {
      const r = typeof ev.result === 'string' ? ev.result.slice(0, 160) : JSON.stringify(ev.result ?? '').slice(0, 160)
      return `${ev.tool} · ${ev.success === true ? '成功' : '失败'}${r ? ` · ${r}` : ''}`
    }
    case 'tool_retry':
      return `${ev.tool} · 第 ${ev.attempt}/${ev.max} 次 · 延时 ${ev.delay}ms${ev.reason ? ` · ${ev.reason}` : ''}`
    case 'retrieve':
      return `${ev.query}${ev.scheme ? ` @${ev.scheme}` : ''} · 命中 ${Array.isArray(ev.hits) ? ev.hits.length : 0} 条`
    case 'classify':
      return ev.status === 'done'
        ? `${ev.retrieval_need ? '需检索' : '无需检索'} · ${ev.generation_mode ?? ''} · conf=${ev.confidence ?? ''}${ev.reason ? ` · ${ev.reason}` : ''}`
        : '路由中…'
    case 'context': {
      const metrics = ev.metrics ? ` · ${JSON.stringify(ev.metrics)}` : ''
      const file = ev.file ? ` · ${ev.file}` : ''
      return `${ev.kind}${metrics}${file}`
    }
    case 'approval_request':
      return `${ev.approval_id} · 待审工具 ${Array.isArray(ev.tool_calls) ? ev.tool_calls.length : 0} 个`
    case 'error':
      return `${ev.message ?? ''}${ev.detail ? ` · ${ev.detail}` : ''}`
    case 'done':
      return String(ev.summary ?? '').slice(0, 200)
    default: {
      const t = typeof ev.text === 'string' ? ev.text : ''
      return t ? t.slice(0, 200) : ''
    }
  }
}

function evRaw(ev: RunEvent): string {
  return JSON.stringify(ev, null, 2)
}

function typeCls(type: string): string {
  if (type === 'llm_call') return 'bg-indigo-500/15 text-indigo-300'
  if (type === 'error' || type === 'guard_refused') return 'bg-rose-500/15 text-rose-300'
  if (['retrieve', 'classify', 'decompose', 'hyde', 'multi_hop', 'multi_hop_plan', 'multi_hop_verify', 'answerability', 'grade', 'verify', 'correct', 'agent_step'].includes(type)) return 'bg-emerald-500/15 text-emerald-300'
  if (['tool_start', 'tool_end', 'tool_retry'].includes(type)) return 'bg-sky-500/15 text-sky-300'
  if (type === 'approval_request') return 'bg-amber-500/15 text-amber-300'
  if (type === 'context') return 'bg-cyan-500/15 text-cyan-300'
  if (['memory_read', 'memory_write', 'memory_constant'].includes(type)) return 'bg-fuchsia-500/15 text-fuchsia-300'
  return 'bg-slate-600/30 text-slate-300'
}

const s = computed(() => selected.value?.meta.stats)

const statCards = computed(() => {
  const st = s.value
  if (!st) return []
  return [
    { label: 'LLM 调用', value: st.llm_calls ?? 0, cls: 'text-indigo-300' },
    { label: 'Token 总量', value: `${st.tokens?.total ?? 0}`, cls: 'text-slate-200' },
    { label: '成本', value: `¥${st.cost_yuan ?? 0}`, cls: 'text-amber-300' },
    { label: '工具调用', value: Object.values(st.tool_calls ?? {}).reduce((a, b) => a + b, 0), cls: 'text-sky-300' },
    { label: '工具重试', value: st.retries ?? 0, cls: 'text-slate-200' },
    { label: '审批次数', value: st.approvals ?? 0, cls: 'text-amber-300' },
    { label: 'RAG 检索', value: `${st.rag_retrieves ?? 0} 次 / ${st.rag_hits ?? 0} 条`, cls: 'text-emerald-300' },
    { label: '记忆 读 / 写', value: `${st.memory_reads ?? 0} / ${st.memory_writes ?? 0}`, cls: 'text-fuchsia-300' },
    { label: '护栏拦截', value: st.guards ?? 0, cls: 'text-rose-300' },
    { label: '大输出落盘', value: st.offloads ?? 0, cls: 'text-cyan-300' },
  ]
})

const toolCards = computed(() => {
  const st = s.value
  if (!st) return []
  return Object.entries(st.tool_calls ?? {}).map(([name, count]) => ({
    name,
    count,
    failures: st.tool_failures?.[name] ?? 0,
  }))
})

watch(
  () => props.open,
  (v) => {
    if (v) void load()
    else selected.value = null
  },
  { immediate: true },
)
</script>

<template>
  <div v-if="open" class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" @click.self="emit('close')">
    <div class="flex max-h-[85vh] w-full max-w-3xl flex-col rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl">
      <div class="flex items-center justify-between border-b border-slate-800 px-5 py-3">
        <div>
          <h2 class="text-sm font-semibold text-white">运行记录</h2>
          <p class="text-[11px] text-slate-500">可观测性演示 · 每次对话一条 run · 事件流 + LLM 调用明细</p>
        </div>
        <button type="button" class="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white" @click="emit('close')">
          <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div class="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        <p v-if="error" class="rounded-lg bg-rose-500/10 px-3 py-2 text-xs text-rose-300">{{ error }}</p>

        <!-- 列表：按会话分组 -->
        <template v-if="!selected">
          <div class="flex items-center justify-between">
            <h3 class="text-xs font-semibold text-slate-300">本次 / 历史运行（按会话分组，最新在前）</h3>
            <button
              type="button"
              class="rounded-lg border border-slate-700 px-2.5 py-1 text-[11px] text-slate-400 hover:bg-slate-800 hover:text-white"
              @click="load"
            >
              刷新
            </button>
          </div>

          <div v-if="groups.length" class="space-y-4">
            <section v-for="g in groups" :key="g.sessionId">
              <h4 class="mb-1.5 flex items-center gap-2 text-[11px] font-medium text-slate-500">
                <span class="rounded bg-slate-800 px-1.5 py-0.5" :class="g.sessionId === props.sessionId ? 'text-amber-300' : ''">
                  {{ g.sessionId === props.sessionId ? '当前会话' : '历史会话' }}
                </span>
                <span class="font-mono">{{ g.sessionId.slice(0, 12) }}…</span>
                <span>{{ g.items.length }} 条</span>
              </h4>
              <div class="space-y-1.5">
                <button
                  v-for="r in g.items"
                  :key="r.run_id"
                  type="button"
                  class="block w-full rounded-xl border border-slate-800 bg-slate-800/40 px-3 py-2 text-left transition hover:border-indigo-500/50 hover:bg-slate-800"
                  @click="openRun(r)"
                >
                  <div class="flex items-center justify-between gap-2">
                    <span class="truncate text-sm text-slate-200">{{ r.message || '（无消息文本）' }}</span>
                    <span class="shrink-0 rounded px-1.5 py-0.5 text-[10px]" :class="STATUS_CLS[r.status] ?? 'bg-slate-600/30 text-slate-300'">
                      {{ STATUS_LABEL[r.status] ?? r.status }}
                    </span>
                  </div>
                  <p class="mt-1 truncate text-[10px] text-slate-500">
                    {{ fmtTs(r.start_ts) }} · 耗时 {{ fmtDuration(r.duration_ms) }}
                    · {{ r.mode }}{{ r.rag_scheme ? ` · RAG ${r.rag_scheme}` : '' }}
                    · LLM {{ r.stats?.llm_calls ?? 0 }} 次 · {{ r.stats?.tokens?.total ?? 0 }}t · ¥{{ r.stats?.cost_yuan ?? 0 }}
                  </p>
                </button>
              </div>
            </section>
          </div>
          <p v-else-if="!loading" class="rounded-xl border border-dashed border-slate-700/70 px-3 py-4 text-center text-[11px] text-slate-600">
            暂无运行记录 — 发送一轮对话后自动落盘一条 run
          </p>
          <p v-if="loading" class="py-2 text-center text-[11px] text-slate-500">加载中…</p>

          <!-- 演示边界：企业级可观测性缺口明示 -->
          <section class="rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-3">
            <h3 class="text-[11px] font-semibold text-amber-300">演示边界（企业级差距明示）</h3>
            <ul class="mt-1 space-y-0.5 text-[10px] leading-relaxed text-slate-400">
              <li>· 观测范围 = 单进程内一次会话 run：完整 SSE 事件流 + 每次 LLM 调用明细（模型 / 时延 / token / 成本 / 成败）</li>
              <li>· 模型分级：生产环境轻量语义判断（语义路由 / 查询改写 / 记忆选择 / CRAG 评审等）应交给低成本小模型，复杂生成交大模型（小模型 + 大模型混合调度）；演示为简化统一用同一模型（qwen3.5-flash）跑全链路，仅作形态演示</li>
              <li>· 不含：跨服务分布式链路追踪（traceID 贯通多进程）、指标聚合看板（Prometheus / Grafana）、告警与采样治理</li>
              <li>· 跨进程调用（如 MCP 工具）仅记录发起与结果，不深入远端内部执行细节</li>
              <li>· 记录按设备指纹隔离（X-Client-Id）；TTL 7 天 / 上限 500 条自动治理</li>
            </ul>
          </section>
        </template>

        <!-- 详情：回放 -->
        <template v-else>
          <button
            type="button"
            class="flex items-center gap-1 text-[11px] text-slate-400 hover:text-white"
            @click="back"
          >
            <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            返回列表
          </button>

          <div v-if="detailLoading" class="py-2 text-center text-[11px] text-slate-500">加载详情…</div>

          <template v-else-if="selected">
            <!-- meta 头 -->
            <section class="rounded-xl border border-slate-800 bg-slate-800/40 p-3">
              <div class="flex items-start justify-between gap-3">
                <p class="text-sm font-medium text-slate-100">{{ selected.meta.message || '（无消息文本）' }}</p>
                <span class="shrink-0 rounded px-1.5 py-0.5 text-[10px]" :class="STATUS_CLS[selected.meta.status] ?? 'bg-slate-600/30 text-slate-300'">
                  {{ STATUS_LABEL[selected.meta.status] ?? selected.meta.status }}
                </span>
              </div>
              <p class="mt-1.5 text-[10px] text-slate-500">
                run_id <span class="font-mono text-slate-400">{{ selected.meta.run_id }}</span> ·
                {{ fmtTs(selected.meta.start_ts) }} → {{ fmtTs(selected.meta.end_ts) }} ·
                耗时 {{ fmtDuration(selected.meta.duration_ms) }} ·
                {{ selected.meta.mode }}{{ selected.meta.prompt_strategy ? ` / ${selected.meta.prompt_strategy}` : '' }}{{ selected.meta.approval_policy ? ` / 审批 ${selected.meta.approval_policy}` : '' }}{{ selected.meta.rag_scheme ? ` / RAG ${selected.meta.rag_scheme}` : '' }}
              </p>
              <p v-if="selected.meta.error" class="mt-1 text-[10px] text-rose-300">错误：{{ selected.meta.error }}</p>
              <p v-if="selected.meta.summary" class="mt-1 text-[10px] text-slate-400">总结：{{ selected.meta.summary }}</p>
            </section>

            <!-- 聚合统计 -->
            <section>
              <h3 class="mb-1.5 text-xs font-semibold text-slate-300">聚合统计</h3>
              <div class="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
                <div v-for="c in statCards" :key="c.label" class="rounded-lg border border-slate-800 bg-slate-800/40 px-2.5 py-1.5">
                  <p class="text-[10px] text-slate-500">{{ c.label }}</p>
                  <p class="text-sm font-semibold" :class="c.cls ?? 'text-slate-200'">{{ c.value }}</p>
                </div>
              </div>
              <div v-if="toolCards.length" class="mt-1.5 flex flex-wrap gap-1.5">
                <span
                  v-for="t in toolCards"
                  :key="t.name"
                  class="rounded-md border border-slate-700 bg-slate-800/60 px-2 py-0.5 text-[10px] text-slate-300"
                >
                  {{ t.name }} ×{{ t.count }}<span v-if="t.failures" class="text-rose-300">（{{ t.failures }} 失败）</span>
                </span>
              </div>
            </section>

            <!-- LLM 调用明细 -->
            <section>
              <h3 class="mb-1.5 text-xs font-semibold text-slate-300">LLM 调用明细（{{ llmCalls.length }} 次）</h3>
              <div v-if="llmCalls.length" class="space-y-1.5">
                <div v-for="c in llmCalls" :key="c.model + c.latency" class="rounded-lg border border-slate-800 bg-slate-800/40 px-3 py-1.5">
                  <div class="flex items-center justify-between gap-2">
                    <span class="truncate text-[11px] text-slate-200">
                      <span class="font-mono text-indigo-300">{{ c.model }}</span>
                      · <span class="text-slate-400">{{ c.scenario }}</span> · {{ c.method }}
                    </span>
                    <span
                      class="shrink-0 rounded px-1.5 py-0.5 text-[10px]"
                      :class="c.success ? 'bg-emerald-500/15 text-emerald-300' : 'bg-rose-500/15 text-rose-300'"
                    >
                      {{ c.success ? '成功' : '失败' }}
                    </span>
                  </div>
                  <p class="mt-0.5 text-[10px] text-slate-500">
                    时延 {{ c.latency }}ms · 输入 {{ c.input }}t / 输出 {{ c.output }}t / 总 {{ c.total }}t
                    <span v-if="c.error" class="text-rose-300"> · {{ c.error }}</span>
                  </p>
                </div>
              </div>
              <p v-else class="rounded-xl border border-dashed border-slate-700/70 px-3 py-3 text-center text-[11px] text-slate-600">
                本次运行无 LLM 调用记录
              </p>
            </section>

            <!-- 事件流回放 -->
            <section>
              <h3 class="mb-1.5 text-xs font-semibold text-slate-300">事件流回放（{{ selected.events.length }} 条，点击展开原始数据）</h3>
              <div class="space-y-1">
                <div v-for="ev in selected.events" :key="ev.seq" class="overflow-hidden rounded-lg border border-slate-800 bg-slate-800/30">
                  <button type="button" class="flex w-full items-center gap-2 px-3 py-1.5 text-left" @click="toggleExpand(ev.seq)">
                    <span class="w-12 shrink-0 text-right font-mono text-[10px] text-slate-600">#{{ ev.seq }}</span>
                    <span class="w-14 shrink-0 font-mono text-[10px] text-slate-500">{{ fmtClock(ev.ts) }}</span>
                    <span class="shrink-0 rounded px-1.5 py-0.5 text-[10px]" :class="typeCls(ev.type)">
                      {{ TYPE_LABEL[ev.type] ?? ev.type }}
                    </span>
                    <span class="min-w-0 flex-1 truncate text-[11px] text-slate-400">{{ evSummary(ev) || '（无摘要）' }}</span>
                    <svg
                      class="h-3 w-3 shrink-0 text-slate-600 transition-transform"
                      :class="expandedSeq === ev.seq ? 'rotate-180' : ''"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="2"
                      viewBox="0 0 24 24"
                    >
                      <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  <pre
                    v-if="expandedSeq === ev.seq"
                    class="max-h-72 overflow-auto border-t border-slate-800 bg-black/30 px-3 py-2 text-[10px] leading-relaxed text-emerald-200/80"
                  >{{ evRaw(ev) }}</pre>
                </div>
              </div>
            </section>
          </template>
        </template>
      </div>
    </div>
  </div>
</template>
