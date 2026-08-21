<script setup lang="ts">
/** 沙箱文件面板：列出 / 下载沙箱工作目录中的文件（Agent 在沙箱里写出的产物）。 */
import { onMounted, ref, watch } from 'vue'

interface SandboxFile {
  path: string
  size: number
  mtime: number
}

const props = defineProps<{
  open?: boolean
  /** 外部触发刷新（如 run_command 执行结束后 +1） */
  refreshKey?: number
}>()

const emit = defineEmits<{ close: [] }>()

const files = ref<SandboxFile[]>([])
const loading = ref(false)
const error = ref('')

function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

function fmtTime(t: number): string {
  const d = new Date(t * 1000)
  const pad = (x: number) => String(x).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const resp = await fetch('/api/sandbox/files')
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    files.value = data.files ?? []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

function download(path: string) {
  window.open(`/api/sandbox/files/download?path=${encodeURIComponent(path)}`, '_blank')
}

watch(
  () => props.refreshKey,
  () => {
    if (props.open) void load()
  },
)
watch(
  () => props.open,
  (o) => {
    if (o) void load()
  },
)
onMounted(() => {
  if (props.open) void load()
})
</script>

<template>
  <aside
    class="fixed inset-y-0 right-0 z-30 flex w-80 flex-col border-l border-slate-800 bg-slate-950 transition-transform duration-200"
    :class="open ? 'translate-x-0' : 'translate-x-full'"
  >
    <div class="flex items-center justify-between border-b border-slate-800 px-4 py-3">
      <div>
        <h2 class="text-sm font-semibold text-white">沙箱文件</h2>
        <p class="text-[11px] text-slate-500">{{ files.length }} 个文件 · 点击下载</p>
      </div>
      <div class="flex items-center gap-1">
        <button
          type="button"
          class="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-800 hover:text-white"
          title="刷新列表"
          @click="load"
        >
          <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
          </svg>
        </button>
        <button
          type="button"
          class="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-800 hover:text-white"
          title="关闭"
          @click="emit('close')"
        >
          <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>

    <div class="flex-1 space-y-1 overflow-y-auto px-3 py-3">
      <div v-if="loading" class="py-8 text-center text-xs text-slate-500">加载中…</div>
      <div v-else-if="error" class="py-8 text-center text-xs text-rose-400">{{ error }}</div>
      <div v-else-if="files.length === 0" class="py-8 text-center text-xs leading-5 text-slate-500">
        暂无文件<br />
        让 Agent 在沙箱里写文件后，可在此下载
      </div>
      <div
        v-for="f in files"
        v-else
        :key="f.path"
        class="flex items-center gap-2 rounded-lg px-2 py-1.5 transition hover:bg-slate-800/60"
      >
        <svg class="h-4 w-4 shrink-0 text-slate-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
        </svg>
        <div class="min-w-0 flex-1">
          <p class="truncate text-xs text-slate-200" :title="f.path">{{ f.path }}</p>
          <p class="text-[10px] text-slate-500">{{ fmtSize(f.size) }} · {{ fmtTime(f.mtime) }}</p>
        </div>
        <button
          type="button"
          class="rounded-md border border-indigo-500/40 px-2 py-0.5 text-[11px] text-indigo-300 transition hover:bg-indigo-500/10"
          @click="download(f.path)"
        >
          下载
        </button>
      </div>
    </div>
  </aside>
</template>
