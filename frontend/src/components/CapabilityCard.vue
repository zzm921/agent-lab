<script setup lang="ts">
import { computed } from 'vue'
import type { Capability } from '../types/agent'

const props = defineProps<{
  cap: Capability
  enabled: boolean
  compact?: boolean
  fault?: string
  faultTypes?: Record<string, string>
}>()
const emit = defineEmits<{ toggle: [id: string]; example: [cap: Capability]; fault: [id: string, mode: string] }>()

const available = props.cap.availability === 'available'

/** 故障注入类型说明（与后端 FAULT_TYPES 描述对齐） */
const FAULT_DESC: Record<string, string> = {
  timeout: '网络请求超时',
  conn_reset: '连接被重置',
  dns: 'DNS 解析失败',
  http_429: '触发限流 429',
  http_500: '服务端错误 500',
  http_502: '网关错误 502',
  http_503: '服务不可用 503',
  error: '通用业务报错',
  business: '业务错误：余额不足',
  http_400: '参数校验失败 400',
  http_401: '未授权 401',
  http_403: '权限不足 403',
  http_404: '资源不存在 404',
}

/** 瞬时错误（retryable）→ 工具层直接重试（透明重试） */
const retryableTypes = computed(() =>
  Object.entries(props.faultTypes ?? {}).filter(([, cls]) => cls === 'retryable'),
)
/** 参数/业务错误（permanent）→ 不直接重试，交给模型思考后重试 */
const permanentTypes = computed(() =>
  Object.entries(props.faultTypes ?? {}).filter(([, cls]) => cls === 'permanent'),
)

function faultTitle(fault: string): string {
  if (!fault || fault === 'off') return '故障注入：选择类型以验证两层重试机制'
  const cls = props.faultTypes?.[fault]
  const tag = cls === 'retryable' ? '瞬时错误 → 工具层直接重试' : '参数/业务错误 → 交给模型思考后重试'
  return `故障注入：${FAULT_DESC[fault] ?? fault}（${tag}）`
}

function onFaultChange(e: Event) {
  emit('fault', props.cap.id, (e.target as HTMLSelectElement).value)
}
</script>

<template>
  <div
    class="flex flex-col rounded-2xl border transition"
    :class="[
      available
        ? 'border-slate-800 bg-slate-900/60 hover:border-indigo-500/50'
        : 'border-slate-800/60 bg-slate-900/30 opacity-60',
      compact ? 'p-3' : 'p-4',
      available && compact ? 'cursor-pointer' : '',
    ]"
    @click="available && compact && emit('toggle', cap.id)"
  >
    <div class="flex items-start justify-between gap-2">
      <div class="flex items-center gap-2">
        <span
          class="rounded-md px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide"
          :class="cap.source === 'mcp' ? 'bg-fuchsia-500/15 text-fuchsia-300' : 'bg-indigo-500/15 text-indigo-300'"
        >
          {{ cap.source === 'mcp' ? 'MCP' : '内置' }}
        </span>
        <h3 class="text-sm font-semibold text-white">{{ cap.name }}</h3>
        <span v-if="cap.server" class="text-[10px] text-fuchsia-400/80">{{ cap.server }}</span>
      </div>

      <!-- 启用开关（热插拔） -->
      <button
        type="button"
        role="switch"
        :aria-checked="enabled"
        :disabled="!available"
        :title="available ? (enabled ? '关闭该能力' : '启用该能力') : '该能力不可用'"
        class="flex h-5 w-9 shrink-0 items-center rounded-full p-0.5 transition disabled:opacity-30"
        :class="enabled ? 'bg-indigo-500' : 'bg-slate-700'"
        @click.stop="emit('toggle', cap.id)"
      >
        <span class="h-4 w-4 rounded-full bg-white transition" :class="enabled ? 'translate-x-4' : ''"></span>
      </button>
    </div>

    <p class="flex-1 text-xs leading-relaxed text-slate-400" :class="compact ? 'mt-1.5 line-clamp-2' : 'mt-2'">
      {{ cap.desc }}
    </p>

    <div class="mt-3 flex items-center justify-between gap-2">
      <span v-if="available" class="text-xs text-emerald-400">● 可用</span>
      <span v-else class="text-xs text-rose-400" title="该能力不适配">● 不适配</span>
      <div class="flex items-center gap-2">
        <!-- 故障注入选择器：验证两层重试机制用（瞬时错误→直接重试 / 参数业务错误→交给模型），不影响工具类本身 -->
        <select
          v-if="available"
          :value="fault"
          :title="faultTitle(fault ?? 'off')"
          class="rounded-lg border border-slate-700 bg-slate-800 px-1.5 py-1 text-[11px] text-slate-300 outline-none transition hover:border-rose-400/60"
          @change="onFaultChange"
        >
          <option value="off">正常</option>
          <optgroup v-if="retryableTypes.length" label="瞬时错误 · 直接重试">
            <option v-for="[id] in retryableTypes" :key="id" :value="id">{{ FAULT_DESC[id] ?? id }}</option>
          </optgroup>
          <optgroup v-if="permanentTypes.length" label="参数/业务错误 · 交给模型">
            <option v-for="[id] in permanentTypes" :key="id" :value="id">{{ FAULT_DESC[id] ?? id }}</option>
          </optgroup>
        </select>
        <button
          v-if="available && cap.example && !compact"
          type="button"
          class="example-btn rounded-lg border border-slate-600 px-2.5 py-1 text-xs text-slate-300 transition hover:border-indigo-400 hover:text-white"
          @click="emit('example', cap)"
        >
          示例
        </button>
      </div>
    </div>

    <p v-if="!available && cap.unavailable_reason" class="mt-2 text-[11px] leading-relaxed text-slate-500">
      {{ cap.unavailable_reason }}
    </p>
  </div>
</template>
