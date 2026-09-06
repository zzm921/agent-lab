<script setup lang="ts">
/** 在线评估闭环 · 用户反馈栏：仅当本轮完成且存在知识库检索命中时展示 👍/👎。
 * 命中判断与后端「无检索命中不落样本」对齐（retrieve 步骤 hits 非空）。
 * 差评可补填原因（可选），提交走 POST /api/feedback 回填当日样本行。
 */
import { computed, ref, watch } from 'vue'
import { submitFeedback } from '../services/sse'
import type { ChatStream } from '../composables/useChatStream'

const props = defineProps<{ stream: ChatStream }>()

/** 本轮是否存在知识库检索命中（retrieve 步骤 hits 非空） */
const hasRagHit = computed(() =>
  props.stream.steps.some((s) => s.kind === 'retrieve' && (s.hits?.length ?? 0) > 0),
)
const showBar = computed(() => props.stream.status === 'done' && props.stream.ragEnabled && hasRagHit.value)

/** 本轮用户提问：时间线最后一条 user 步骤文本 */
const query = computed(() => {
  let q = ''
  for (const s of props.stream.steps) {
    if (s.kind === 'user') q = s.text ?? ''
  }
  return q
})

const voted = ref<'up' | 'down' | null>(null)
const reason = ref('')
const submitting = ref(false)
const sent = ref(false)

// 新的一轮开始/不再满足展示条件时重置
watch(showBar, (v) => {
  if (!v) {
    voted.value = null
    reason.value = ''
    submitting.value = false
    sent.value = false
  }
})

async function onVote(vote: 'up' | 'down') {
  if (voted.value || submitting.value) return
  voted.value = vote
  if (vote === 'down') return // 差评先留原因输入机会
  await submit()
}

async function submit() {
  if (submitting.value) return
  submitting.value = true
  try {
    await submitFeedback({
      session_id: props.stream.sessionId,
      query: query.value,
      vote: voted.value ?? 'up',
      reason: reason.value.trim(),
    })
    sent.value = true
  } catch {
    voted.value = null // 提交失败允许重试
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <section v-if="showBar" class="rounded-xl border border-slate-700/60 bg-slate-800/20 p-3 text-xs">
    <div class="flex flex-wrap items-center gap-2">
      <span class="text-slate-400">这个回答对您有帮助吗？</span>
      <button
        type="button"
        class="rounded-lg border px-2.5 py-1 transition disabled:cursor-default"
        :class="
          voted === 'up'
            ? 'border-emerald-500/60 bg-emerald-500/15 text-emerald-300'
            : 'border-slate-700 text-slate-300 hover:border-emerald-500/40 hover:text-emerald-300'
        "
        :disabled="voted !== null"
        @click="onVote('up')"
      >
        👍 有帮助
      </button>
      <button
        type="button"
        class="rounded-lg border px-2.5 py-1 transition disabled:cursor-default"
        :class="
          voted === 'down'
            ? 'border-rose-500/60 bg-rose-500/15 text-rose-300'
            : 'border-slate-700 text-slate-300 hover:border-rose-500/40 hover:text-rose-300'
        "
        :disabled="voted !== null"
        @click="onVote('down')"
      >
        👎 没帮助
      </button>
      <span v-if="sent" class="text-emerald-400">✓ 已反馈，感谢您的反馈</span>
    </div>
    <!-- 差评补填原因（可选），提交后一并回填样本行 -->
    <div v-if="voted === 'down' && !sent" class="mt-2 flex items-center gap-2">
      <input
        v-model="reason"
        type="text"
        maxlength="200"
        placeholder="可选：简要说明原因（如回答有误 / 未命中 / 不全）"
        class="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-slate-200 outline-none transition focus:border-indigo-500/60"
        @keyup.enter="submit"
      />
      <button
        type="button"
        class="shrink-0 rounded-lg bg-indigo-500/80 px-2.5 py-1.5 font-medium text-white transition enabled:hover:bg-indigo-500 disabled:opacity-50"
        :disabled="submitting"
        @click="submit"
      >
        {{ submitting ? '提交中…' : '提交' }}
      </button>
      <button
        type="button"
        class="shrink-0 rounded-lg border border-slate-700 px-2.5 py-1.5 text-slate-400 transition hover:text-slate-200"
        @click="voted = null"
      >
        取消
      </button>
    </div>
  </section>
</template>
