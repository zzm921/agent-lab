<script setup lang="ts">
import type { ChatStream } from '../composables/useChatStream'
import StepTimeline from './StepTimeline.vue'
import ToolCallBadge from './ToolCallBadge.vue'
import StreamingText from './StreamingText.vue'
import ApprovalDialog from './ApprovalDialog.vue'
import ErrorBanner from './ErrorBanner.vue'

defineProps<{ stream: ChatStream }>()

const STATUS: Record<string, { label: string; cls: string }> = {
  idle: { label: '待命', cls: 'bg-slate-800 text-slate-400' },
  streaming: { label: '执行中', cls: 'bg-indigo-500/20 text-indigo-300' },
  waiting_approval: { label: '等待审批', cls: 'bg-amber-500/20 text-amber-300' },
  done: { label: '已完成', cls: 'bg-emerald-500/20 text-emerald-300' },
  error: { label: '出错', cls: 'bg-rose-500/20 text-rose-300' },
}

const RETRIEVE_KIND: Record<string, string> = {
  retrieve: '知识库检索',
  memory_read: '记忆召回',
  memory_write: '记忆写入',
}
</script>

<template>
  <div class="rounded-2xl border border-slate-800 bg-slate-900/30">
    <!-- 状态栏 -->
    <div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-4 py-2.5">
      <div class="flex items-center gap-2 text-xs">
        <span class="rounded bg-slate-800 px-2 py-0.5 font-mono text-slate-300">{{ stream.mode }}</span>
        <span class="rounded px-2 py-0.5" :class="STATUS[stream.status]?.cls ?? STATUS.idle.cls">
          {{ STATUS[stream.status]?.label ?? '待命' }}
        </span>
        <span v-if="stream.status !== 'idle'" class="text-slate-500">{{ stream.elapsed.toFixed(1) }}s</span>
      </div>
      <div class="flex gap-3 text-[11px] text-slate-500">
        <span>工具调用 {{ stream.steps.filter((s) => s.kind === 'tool').length }}</span>
        <span>计划 {{ stream.steps.filter((s) => s.kind === 'plan').length }} 步</span>
        <span>检索 {{ stream.steps.filter((s) => s.kind === 'retrieve' || s.kind === 'memory_read' || s.kind === 'memory_write').length }}</span>
        <span>Worker {{ stream.steps.filter((s) => s.kind === 'agent_event').length }}</span>
      </div>
    </div>

    <div class="space-y-4 p-4">
      <ErrorBanner :error="stream.error" @retry="stream.retry()" />

      <!-- 流水线：用户输入 → 思考 → 工具 → … → 输出，按发生顺序渲染、多轮交替 -->
      <template v-for="s in stream.steps" :key="s.id">
        <!-- 用户输入（右侧气泡） -->
        <div v-if="s.kind === 'user'" class="flex justify-end">
          <div class="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-indigo-600/90 px-4 py-2.5 text-sm leading-relaxed text-white">
            {{ s.text }}
          </div>
        </div>

        <!-- 思考过程（灰色斜体，弱化） -->
        <section v-else-if="s.kind === 'thinking'" class="border-l-2 border-slate-700 pl-3">
          <h4 class="mb-1 text-[11px] font-medium text-slate-500">思考过程</h4>
          <p class="text-xs italic text-slate-500">
            <StreamingText :text="s.text ?? ''" :streaming="s.streaming ?? false" />
          </p>
        </section>

        <!-- 工具调用（运行中带加载动画） -->
        <ToolCallBadge v-else-if="s.kind === 'tool'" :entry="s" />

        <!-- 执行计划 -->
        <section v-else-if="s.kind === 'plan'" class="rounded-xl border border-slate-800 p-3">
          <h4 class="mb-2 text-xs font-semibold text-slate-300">执行计划</h4>
          <StepTimeline
            :steps="s.steps ?? []"
            :current-step="s.currentStep ?? 0"
            :status="s.planStatus ?? 'pending'"
          />
        </section>

        <!-- 检索 / 记忆 -->
        <section
          v-else-if="s.kind === 'retrieve' || s.kind === 'memory_read' || s.kind === 'memory_write'"
          class="rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-3 text-xs"
        >
          <div class="flex items-center justify-between gap-2">
            <span class="font-medium text-cyan-300">{{ RETRIEVE_KIND[s.kind] }}</span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <ul v-if="s.hits?.length" class="mt-2 space-y-1.5">
            <li v-for="(h, i) in s.hits" :key="i" class="rounded-lg bg-black/20 p-2">
              <p class="text-slate-300">{{ h.text }}</p>
              <p class="mt-0.5 text-[11px] text-cyan-400/80">
                相关度 {{ typeof h.score === 'number' ? h.score.toFixed(3) : h.score }}
              </p>
            </li>
          </ul>
          <p v-else-if="s.content" class="mt-1.5 text-slate-300">{{ s.content }}</p>
        </section>

        <!-- 反思意见 -->
        <section v-else-if="s.kind === 'reflect'" class="rounded-xl border border-rose-500/30 bg-rose-500/5 p-3 text-xs">
          <p class="text-rose-300">{{ s.stage ? `阶段：${s.stage}` : '评审意见' }}</p>
          <p v-if="s.critique" class="mt-1 whitespace-pre-wrap text-slate-300">{{ s.critique }}</p>
        </section>

        <!-- 修订稿 -->
        <section v-else-if="s.kind === 'revise'" class="rounded-xl border border-orange-500/30 bg-orange-500/5 p-3 text-xs">
          <p class="text-orange-300">修订稿：</p>
          <p class="mt-1 text-slate-200">
            <StreamingText :text="s.text ?? ''" :streaming="s.streaming ?? false" />
          </p>
        </section>

        <!-- 多智能体 -->
        <section v-else-if="s.kind === 'agent_event'" class="rounded-xl border border-sky-500/30 bg-sky-500/5 p-3 text-xs">
          <div class="flex items-center gap-2">
            <span class="font-mono font-semibold text-sky-300">{{ s.worker }}</span>
            <span class="rounded bg-slate-800 px-1.5 text-slate-400">{{ s.agentStatus }}</span>
            <span v-if="s.task" class="text-slate-400">{{ s.task }}</span>
          </div>
          <p v-if="s.agentResult" class="mt-1 whitespace-pre-wrap text-slate-300">{{ s.agentResult }}</p>
        </section>

        <!-- 最终输出（左侧气泡） -->
        <div v-else-if="s.kind === 'message'" class="flex justify-start">
          <div class="max-w-[85%] rounded-2xl rounded-bl-sm border border-indigo-500/20 bg-indigo-500/5 px-4 py-3">
            <h4 class="mb-1.5 text-[11px] font-medium text-indigo-400">输出</h4>
            <p class="text-sm leading-relaxed text-slate-100">
              <StreamingText :text="s.text ?? ''" :streaming="s.streaming ?? false" placeholder="等待回答…" />
            </p>
          </div>
        </div>
      </template>

      <!-- 完成 -->
      <section
        v-if="stream.done"
        class="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs"
      >
        <p class="text-emerald-300">✓ {{ stream.done.summary }}</p>
        <p class="mt-1 text-slate-400">统计：{{ JSON.stringify(stream.done.stats) }}</p>
      </section>
    </div>

    <ApprovalDialog
      v-if="stream.approval"
      :key="stream.approval.approval_id"
      :approval="stream.approval"
      @decision="stream.decide($event.decision, $event.modifiedArgs)"
    />
  </div>
</template>
