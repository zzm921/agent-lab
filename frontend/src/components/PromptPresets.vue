<script setup lang="ts">
import { ref, watch } from 'vue'

const STORAGE_KEY = 'agent-lab-prompts'
const DEFAULT_PRESETS = [
  '帮我计算 (137×0.85−20)÷3 等于多少',
  '帮我搜索并总结最近 AI Agent 相关的新闻',
]

const emit = defineEmits<{ insert: [v: string] }>()

/** 从 localStorage 读取；无记录时回退内置示例（用户删光后持久化为空，不回退） */
function loadPresets(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed: unknown = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        return parsed.filter((p): p is string => typeof p === 'string' && p.trim().length > 0)
      }
    }
  } catch {
    // 存储损坏时回退内置示例
  }
  return [...DEFAULT_PRESETS]
}

const presets = ref<string[]>(loadPresets())
const manageOpen = ref(false)
const draft = ref('')

watch(
  presets,
  (v) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(v))
    } catch {
      // 存储失败（如隐私模式）静默忽略
    }
  },
  { deep: true },
)

function insert(p: string) {
  emit('insert', p)
}

function addPreset() {
  const v = draft.value.trim()
  if (!v) return
  presets.value.push(v)
  draft.value = ''
}

function removePreset(i: number) {
  presets.value.splice(i, 1)
}
</script>

<template>
  <div class="mt-3">
    <div class="flex items-center justify-between">
      <span class="text-[11px] text-slate-500">快捷 Prompt</span>
      <button
        type="button"
        class="manage-toggle flex h-5 w-5 items-center justify-center rounded text-slate-500 transition hover:bg-slate-800 hover:text-white"
        :class="manageOpen ? 'bg-slate-800 text-white' : ''"
        title="管理快捷 prompt（新增 / 删除）"
        @click="manageOpen = !manageOpen"
      >
        <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 010 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a7.723 7.723 0 010-.255c.007-.378-.138-.75-.43-.991l-1.004-.827a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28z"
          />
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </button>
    </div>

    <div v-if="manageOpen" class="mt-2 flex gap-2">
      <input
        v-model="draft"
        class="preset-input min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-900/70 px-2.5 py-1.5 text-xs text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-indigo-400"
        placeholder="输入新 prompt，回车或点击添加"
        @keydown.enter.prevent="addPreset"
      />
      <button
        type="button"
        class="preset-add shrink-0 rounded-lg border border-indigo-500/50 px-2.5 py-1.5 text-xs font-medium text-indigo-300 transition enabled:hover:bg-indigo-500/10 disabled:opacity-40"
        :disabled="!draft.trim()"
        @click="addPreset"
      >
        添加
      </button>
    </div>

    <div v-if="presets.length" class="mt-2 flex flex-wrap gap-1.5">
      <span
        v-for="(p, i) in presets"
        :key="i"
        class="flex max-w-full items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/50 px-2 py-1 text-xs text-slate-300 transition hover:border-indigo-400 hover:text-white"
        role="button"
        :title="p"
        @click="insert(p)"
      >
        <span class="truncate">{{ p }}</span>
        <button
          v-if="manageOpen"
          type="button"
          class="preset-remove shrink-0 text-slate-500 transition hover:text-rose-400"
          title="删除"
          @click.stop="removePreset(i)"
        >
          ✕
        </button>
      </span>
    </div>
    <div v-else class="mt-2 text-[11px] text-slate-600">
      暂无快捷 prompt，点击右上角 ⚙ 添加
    </div>
  </div>
</template>
