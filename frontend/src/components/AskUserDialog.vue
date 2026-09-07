<script setup lang="ts">
/** HITL 用户回复卡片：Agent 调用 ask_user 提问时弹出，支持多问题表单（逐题选择/输入/跳过，整体提交）。 */
import { computed, ref, watch } from 'vue'
import type { AskAnswer, AskQuestion, AskUserRequest } from '../types/agent'

const props = defineProps<{ askUser: AskUserRequest }>()
const emit = defineEmits<{
  reply: [payload: { answers: AskAnswer[] }]
}>()

/** 归一化问题列表：兼容 questions 数组与旧单问结构（question/options 字段） */
const questions = computed<AskQuestion[]>(() => {
  const qs = props.askUser?.questions
  if (Array.isArray(qs)) return qs.filter((q) => q && typeof q.question === 'string' && q.question.trim())
  if (props.askUser && typeof (props.askUser as unknown as Record<string, unknown>).question === 'string') {
    const legacy = props.askUser as unknown as { question: string; options?: string[] }
    return [{ question: legacy.question, options: legacy.options }]
  }
  return []
})

/** 防御性归一化：options 可能是字符串（模型偶发返回），统一转成数组，避免按字符拆分渲染 */
function optionList(q: AskQuestion): string[] {
  const opts = q?.options
  if (!opts) return []
  if (Array.isArray(opts)) return opts.filter((o) => typeof o === 'string' && o.trim())
  return [String(opts).trim()].filter(Boolean)
}

/** 无问题时兜底一个自由输入槽位，避免弹空卡片 */
const displayQuestions = computed<AskQuestion[]>(() =>
  questions.value.length ? questions.value : [{ question: '请输入您想补充的信息' }],
)

// 逐题草稿输入与显式跳过标记（按下标与 displayQuestions 对齐）
const drafts = ref<string[]>([])
const skipped = ref<boolean[]>([])

watch(
  () => props.askUser,
  () => {
    drafts.value = displayQuestions.value.map(() => '')
    skipped.value = displayQuestions.value.map(() => false)
  },
  { immediate: true },
)

function useOption(qi: number, opt: string) {
  drafts.value[qi] = opt
  skipped.value[qi] = false
}

function toggleSkip(qi: number) {
  skipped.value[qi] = !skipped.value[qi]
}

/** 至少回答一题或显式跳一题才可提交；其余未填题目按跳过处理 */
const canSubmit = computed(() =>
  displayQuestions.value.some((_, i) => skipped.value[i] || (drafts.value[i]?.trim() ?? '') !== ''),
)

function submit() {
  const answers: AskAnswer[] = displayQuestions.value.map((_, i) => {
    const text = drafts.value[i]?.trim() ?? ''
    return skipped.value[i] || !text ? { answer: '', skip: true } : { answer: text, skip: false }
  })
  emit('reply', { answers })
}

function skipAll() {
  emit('reply', { answers: displayQuestions.value.map(() => ({ answer: '', skip: true })) })
}
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div class="max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-2xl border border-fuchsia-500/40 bg-slate-900 p-5 shadow-2xl">
        <div class="mb-3 flex items-center justify-between">
          <h3 class="text-base font-semibold text-white">需要您补充信息</h3>
          <span class="rounded bg-fuchsia-500/20 px-2 py-0.5 text-xs text-fuchsia-300">HITL</span>
        </div>

        <div v-if="displayQuestions.length > 1" class="mb-3 text-[11px] text-slate-400">
          共 {{ displayQuestions.length }} 个问题，可逐题回答；未填写的题目将视为跳过。
        </div>

        <div
          v-for="(q, qi) in displayQuestions"
          :key="qi"
          class="mb-3 rounded-xl border border-slate-700/70 bg-slate-950 p-3"
          :class="skipped[qi] && 'opacity-50'"
        >
          <p class="mb-2 text-sm font-medium leading-relaxed text-white">
            <span class="mr-1.5 text-xs text-fuchsia-400">Q{{ qi + 1 }}</span>{{ q.question }}
          </p>

          <div v-if="optionList(q).length" class="mb-2 flex flex-wrap gap-2">
            <button
              v-for="opt in optionList(q)"
              :key="opt"
              type="button"
              class="rounded-lg border border-slate-600 px-3 py-1.5 text-xs text-slate-200 transition hover:border-fuchsia-400/60 hover:text-white"
              :class="drafts[qi] === opt && 'border-fuchsia-400/70 bg-fuchsia-500/10 text-fuchsia-200'"
              @click="useOption(qi, opt)"
            >
              {{ opt }}
            </button>
          </div>

          <textarea
            v-model="drafts[qi]"
            :disabled="skipped[qi]"
            rows="2"
            :placeholder="skipped[qi] ? '此题已跳过' : '输入您的回复，或点选上方选项…'"
            class="w-full resize-none rounded-lg border border-slate-700 bg-slate-900 p-2.5 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-fuchsia-400 disabled:opacity-50"
          ></textarea>

          <div class="mt-1.5 text-right">
            <button
              type="button"
              class="text-[11px] text-slate-500 transition hover:text-fuchsia-300"
              @click="toggleSkip(qi)"
            >
              {{ skipped[qi] ? '取消跳过' : '跳过此题 / 无法回答' }}
            </button>
          </div>
        </div>

        <div class="mt-4 flex items-center justify-between gap-2">
          <span class="text-[11px] text-slate-500">回复将作为澄清结果返回给 Agent</span>
          <div class="flex shrink-0 gap-2">
            <button
              type="button"
              class="rounded-lg border border-rose-500/50 px-3 py-1.5 text-sm text-rose-300 transition hover:bg-rose-500/10"
              @click="skipAll"
            >
              全部跳过
            </button>
            <button
              type="button"
              :disabled="!canSubmit"
              class="rounded-lg bg-gradient-to-r from-fuchsia-500 to-indigo-500 px-4 py-1.5 text-sm font-semibold text-white transition enabled:hover:opacity-90 disabled:opacity-40"
              @click="submit"
            >
              发送回复
            </button>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>
