<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import hljs from 'highlight.js'
import { LINE_NOTES, type CodeNotes } from '../data/lineNotes'

const props = defineProps<{ codeKey: string }>()

const source = ref('')
const loading = ref(false)
const error = ref('')

async function load() {
  loading.value = true
  error.value = ''
  try {
    const res = await fetch(`/api/source/${props.codeKey}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    source.value = data.content ?? ''
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}
onMounted(load)
watch(() => props.codeKey, load)

const notes = computed<CodeNotes | undefined>(() => LINE_NOTES[props.codeKey])
const lines = computed(() => source.value.split('\n'))
const highlighted = computed(() => {
  if (!source.value) return ''
  try {
    return hljs.highlight(source.value, { language: 'python' }).value
  } catch {
    return source.value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  }
})
const highlightedLines = computed(() => {
  const set = new Set<number>()
  notes.value?.notes.forEach((n) => {
    for (let i = n.range[0]; i <= n.range[1]; i++) set.add(i)
  })
  return set
})
</script>

<template>
  <div class="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950">
    <div class="flex items-center justify-between gap-2 border-b border-slate-800 px-4 py-2">
      <span class="font-mono text-xs text-slate-300">{{ notes?.file ?? `${codeKey}.py` }}</span>
      <span class="text-[11px] text-slate-500">后端真实源码 · /api/source/{{ codeKey }}</span>
    </div>

    <div v-if="loading" class="py-8 text-center text-sm text-slate-500">源码加载中…</div>
    <div v-else-if="error" class="py-8 text-center text-sm text-rose-400">源码加载失败：{{ error }}</div>
    <div v-else-if="!source" class="py-8 text-center text-sm text-slate-500">该模块暂无源码</div>
    <div v-else class="flex text-xs">
      <pre class="select-none overflow-hidden border-r border-slate-800/60 px-2 py-3 text-right font-mono text-slate-600">
        <code>
          <span
            v-for="(_, i) in lines"
            :key="i"
            class="block leading-6"
            :class="highlightedLines.has(i + 1) ? 'bg-indigo-500/20 font-semibold text-indigo-300' : ''"
          >{{ i + 1 }}</span>
        </code>
      </pre>
      <div class="flex-1 overflow-x-auto py-3 pr-3">
        <pre class="font-mono text-slate-200"><code class="block leading-6 whitespace-pre" v-html="highlighted"></code></pre>
      </div>
    </div>

    <div v-if="notes?.notes.length" class="border-t border-slate-800 px-4 py-3">
      <p class="mb-1.5 text-[11px] font-medium text-slate-400">行注释</p>
      <ul class="space-y-1.5">
        <li v-for="(n, i) in notes.notes" :key="i" class="flex gap-2 text-[11px]">
          <span class="shrink-0 rounded bg-indigo-500/20 px-1.5 font-mono text-indigo-300">
            L{{ n.range[0] }}{{ n.range[1] > n.range[0] ? '-' + n.range[1] : '' }}
          </span>
          <span class="text-slate-300">{{ n.text }}</span>
        </li>
      </ul>
    </div>
  </div>
</template>
