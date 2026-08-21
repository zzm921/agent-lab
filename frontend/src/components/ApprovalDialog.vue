<script setup lang="ts">
import { reactive, ref } from 'vue'
import type { ApprovalRequest, ToolCallInfo } from '../types/agent'

const props = defineProps<{ approval: ApprovalRequest }>()
const emit = defineEmits<{
  decision: [payload: { decision: 'approve' | 'reject' | 'modify'; modifiedArgs?: Record<string, unknown> }]
}>()

const keyOf = (c: ToolCallInfo) => c.id ?? c.name

// 每个工具调用一个可编辑 JSON 文本域，预填原参数
const edits = reactive<Record<string, string>>({})
for (const c of props.approval.tool_calls) {
  if (!(keyOf(c) in edits)) edits[keyOf(c)] = JSON.stringify(c.args ?? {}, null, 2)
}
const error = ref('')

function approve() {
  emit('decision', { decision: 'approve' })
}

function reject() {
  emit('decision', { decision: 'reject' })
}

function submitModify() {
  const modifiedArgs: Record<string, unknown> = {}
  for (const c of props.approval.tool_calls) {
    const raw = edits[keyOf(c)] ?? '{}'
    try {
      modifiedArgs[keyOf(c)] = JSON.parse(raw)
    } catch {
      error.value = `工具「${c.name}」的参数 JSON 解析失败，请检查格式`
      return
    }
  }
  emit('decision', { decision: 'modify', modifiedArgs })
}
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div class="w-full max-w-lg rounded-2xl border border-indigo-500/40 bg-slate-900 p-5 shadow-2xl">
        <div class="mb-3 flex items-center justify-between">
          <h3 class="text-base font-semibold text-white">工具调用审批</h3>
          <span class="rounded bg-indigo-500/20 px-2 py-0.5 text-xs text-indigo-300">HITL</span>
        </div>
        <p class="mb-3 text-xs text-slate-400">Agent 请求执行以下工具调用，请批准、拒绝或修改参数：</p>

        <div class="space-y-3">
          <div v-for="c in approval.tool_calls" :key="keyOf(c)" class="rounded-xl border border-slate-700 bg-slate-950 p-3">
            <p class="mb-1 font-mono text-sm font-semibold text-slate-100">{{ c.name }}</p>
            <textarea
              :value="edits[keyOf(c)]"
              rows="4"
              spellcheck="false"
              class="w-full resize-none rounded-lg border border-slate-700 bg-slate-900 p-2 font-mono text-xs text-slate-200 outline-none transition focus:border-indigo-400"
              @input="edits[keyOf(c)] = ($event.target as HTMLTextAreaElement).value"
            ></textarea>
          </div>
        </div>

        <p v-if="error" class="mt-2 text-xs text-rose-400">{{ error }}</p>

        <div class="mt-4 flex justify-end gap-2">
          <button
            type="button"
            class="rounded-lg border border-rose-500/50 px-3 py-1.5 text-sm text-rose-300 transition hover:bg-rose-500/10"
            @click="reject"
          >
            拒绝
          </button>
          <button
            type="button"
            class="rounded-lg border border-slate-600 px-3 py-1.5 text-sm text-slate-300 transition hover:bg-slate-700"
            @click="submitModify"
          >
            修改并提交
          </button>
          <button
            type="button"
            class="rounded-lg bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-3 py-1.5 text-sm font-semibold text-white transition hover:opacity-90"
            @click="approve"
          >
            批准
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
