<script setup lang="ts">
import { reactive } from 'vue'
import type { ChatStream } from '../composables/useChatStream'
import StepTimeline from './StepTimeline.vue'
import ToolCallBadge from './ToolCallBadge.vue'
import StreamingText from './StreamingText.vue'
import ApprovalDialog from './ApprovalDialog.vue'
import ErrorBanner from './ErrorBanner.vue'
import GuardBanner from './GuardBanner.vue'
defineProps<{ stream: ChatStream }>()

// 思考过程默认折叠，点击标题展开/收起（按步骤 id 独立记录）
const expandedThinking = reactive(new Set<number>())
function toggleThinking(id: number) {
  if (expandedThinking.has(id)) expandedThinking.delete(id)
  else expandedThinking.add(id)
}

const STATUS: Record<string, { label: string; cls: string }> = {
  idle: { label: '待命', cls: 'bg-slate-800 text-slate-400' },
  streaming: { label: '执行中', cls: 'bg-indigo-500/20 text-indigo-300' },
  waiting_approval: { label: '等待审批', cls: 'bg-amber-500/20 text-amber-300' },
  done: { label: '已完成', cls: 'bg-emerald-500/20 text-emerald-300' },
  error: { label: '出错', cls: 'bg-rose-500/20 text-rose-300' },
}

const RETRIEVE_KIND: Record<string, string> = {
  rewrite: 'Query 重写',
  classify: '语义路由',
  decompose: 'Query 分解',
  hyde: 'HyDE 假想文档检索',
  multi_hop_plan: '多跳规划',
  multi_hop: '多跳检索',
  multi_hop_verify: '多跳验证',
  compress: '上下文压缩',
  answerability: '答案充分性',
  retrieve: '知识库检索',
  agent_step: 'Agent 检索',
  grade: '证据评审',
  correct: '纠错决策',
  verify: '答案校验',
  memory_read: '记忆召回',
  memory_write: '记忆写入',
}

// 路由决策各维度的中文展示与配色（modular 五维）
const COMPLEXITY_LABEL: Record<string, string> = {
  simple: '简单（单次检索）',
  rewrite: '改写后检索',
  decompose: '分解后检索',
  multihop: '多跳（规划-执行-验证）',
}
const COMPLEXITY_CLS: Record<string, string> = {
  simple: 'bg-slate-500/20 text-slate-200',
  rewrite: 'bg-amber-500/20 text-amber-200',
  decompose: 'bg-violet-500/20 text-violet-200',
  multihop: 'bg-rose-500/20 text-rose-200',
}
const MODE_LABEL: Record<string, string> = {
  vector: '向量检索',
  hybrid: '混合检索',
  multi_recall: '多路召回',
}
const GEN_LABEL: Record<string, string> = {
  direct: '直接回答',
  citation: '引用回答',
  comparison: '对比回答',
}
// 答案充分性验证结论的中文展示与配色（answerability 卡片）
const RECOMMENDATION_LABEL: Record<string, string> = {
  answer: '可直接回答',
  escalate: '需升级检索',
  clarify: '需追问澄清',
}
const RECOMMENDATION_CLS: Record<string, string> = {
  answer: 'bg-emerald-500/20 text-emerald-200',
  escalate: 'bg-amber-500/20 text-amber-200',
  clarify: 'bg-rose-500/20 text-rose-200',
}
// agentic 检索工具动作的中文展示
const ACTION_LABEL: Record<string, string> = {
  search: '向量检索',
  hybrid: '混合检索',
  volume_search: '定向卷检索',
  multi_hop: '多跳检索',
}
// agentic 角色中文展示
const ROLE_LABEL: Record<string, string> = {
  retriever: '检索执行',
  route: '路由',
  plan: '规划',
  grade: '评审',
  correct: '纠错',
  verify: '校验',
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
      <GuardBanner :guard="stream.guard" />

      <!-- 流水线：用户输入 → 思考 → 工具 → … → 输出，按发生顺序渲染、多轮交替 -->
      <template v-for="s in stream.steps" :key="s.id">
        <!-- 用户输入（右侧气泡） -->
        <div v-if="s.kind === 'user'" class="flex justify-end">
          <div class="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-indigo-600/90 px-4 py-2.5 text-sm leading-relaxed text-white">
            {{ s.text }}
          </div>
        </div>

        <!-- 思考过程（默认折叠，流式中显示转圈动效，点击标题展开内容） -->
        <section v-else-if="s.kind === 'thinking'" class="border-l-2 border-slate-700 pl-3">
          <button
            type="button"
            class="flex items-center gap-1.5 text-[11px] font-medium text-slate-500 transition hover:text-slate-300"
            @click="toggleThinking(s.id)"
          >
            <svg
              v-if="s.streaming"
              class="h-3.5 w-3.5 shrink-0 animate-spin text-indigo-400"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
              <path class="opacity-90" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
            </svg>
            <svg
              v-else
              class="h-3 w-3 shrink-0 transition-transform"
              :class="expandedThinking.has(s.id) ? 'rotate-90' : ''"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="m9 18 6-6-6-6" />
            </svg>
            思考过程
          </button>
          <!-- 直接显示已累积的文本：不重放打字动画（思考过程可能已输出大量内容，展开时不应从头打字） -->
          <p v-if="expandedThinking.has(s.id)" class="mt-1 whitespace-pre-wrap break-words text-xs italic text-slate-500">
            {{ s.text }}
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

        <!-- 语义路由（modular：五维路由决策 → 执行计划编排） -->
        <section v-else-if="s.kind === 'classify'" class="rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-indigo-300">
              <svg
                v-if="s.running"
                class="h-3.5 w-3.5 shrink-0 animate-spin text-indigo-400"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-90" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
              </svg>
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-indigo-500/20 px-1.5 py-0.5 text-[10px] font-normal text-indigo-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <!-- 语义路由为纯 LLM 阻塞调用：running 占位显示转圈，done 后再填充五维决策 -->
          <p v-if="s.running" class="mt-2 text-indigo-300/70">语义路由分析中…</p>
          <template v-else>
            <div class="mt-2 flex flex-wrap items-center gap-1.5">
              <!-- retrieval_need=false 时 complexity/retrieval_mode/generation_mode 只是占位值（vector/simple/direct），不展示避免与"不检索"矛盾 -->
              <span
                v-if="s.complexity && s.retrieval_need !== false"
                class="rounded px-1.5 py-0.5"
                :class="COMPLEXITY_CLS[s.complexity] ?? 'bg-slate-800 text-slate-300'"
              >
                {{ COMPLEXITY_LABEL[s.complexity] ?? s.complexity }}
              </span>
              <span
                v-if="s.retrieval_mode && s.retrieval_need !== false"
                class="rounded bg-cyan-500/10 px-1.5 py-0.5 text-cyan-200"
              >
                检索：{{ MODE_LABEL[s.retrieval_mode] ?? s.retrieval_mode }}
              </span>
              <span
                v-if="s.generation_mode && s.retrieval_need !== false"
                class="rounded bg-emerald-500/10 px-1.5 py-0.5 text-emerald-200"
              >
                生成：{{ GEN_LABEL[s.generation_mode] ?? s.generation_mode }}
              </span>
              <span
                v-if="s.retrieval_need === false"
                class="rounded bg-slate-600/30 px-1.5 py-0.5 text-slate-300"
              >
                不检索（直接回答）
              </span>
              <span v-if="typeof s.confidence === 'number'" class="text-slate-500">
                置信度 {{ (s.confidence * 100).toFixed(0) }}%
              </span>
            </div>
            <p v-if="s.reason" class="mt-1.5 text-slate-500">{{ s.reason }}</p>
          </template>
        </section>

        <!-- Query 分解结果（modular：复杂对比/多实体问题的子查询） -->
        <section v-else-if="s.kind === 'decompose'" class="rounded-xl border border-violet-500/20 bg-violet-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-violet-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-normal text-violet-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <div class="mt-2 flex flex-wrap items-center gap-1.5">
            <span
              v-for="(q, qi) in s.sub_queries"
              :key="qi"
              class="rounded bg-violet-500/10 px-1.5 py-0.5 text-violet-200"
            >
              {{ q }}
            </span>
          </div>
        </section>

        <!-- 多跳规划（modular：规划-执行-验证的规划阶段，目标/依赖/可预判实体） -->
        <section v-else-if="s.kind === 'multi_hop_plan'" class="rounded-xl border border-rose-500/20 bg-rose-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-rose-300">
              <svg
                v-if="s.running"
                class="h-3.5 w-3.5 shrink-0 animate-spin text-rose-400"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-90" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
              </svg>
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-rose-500/20 px-1.5 py-0.5 text-[10px] font-normal text-rose-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <!-- 多跳规划为纯 LLM 阻塞调用：running 占位显示转圈，done 后再填充计划 -->
          <p v-if="s.running" class="mt-2 text-rose-300/70">多跳规划中…</p>
          <template v-else>
            <div v-if="s.plan?.steps?.length" class="mt-2 space-y-1.5">
              <div
                v-for="(st, si) in s.plan.steps"
                :key="si"
                class="flex flex-wrap items-center gap-1.5 rounded-lg bg-black/20 p-2"
              >
                <span class="rounded bg-rose-500/10 px-1.5 py-0.5 text-rose-200">{{ st.target }}</span>
                <span class="text-slate-300">{{ st.query }}</span>
                <span
                  v-if="st.depends_on?.length"
                  class="text-[11px] text-slate-500"
                  title="依赖的先前目标"
                >
                  ← {{ st.depends_on.join('、') }}
                </span>
                <span
                  v-if="st.entity"
                  class="rounded bg-amber-500/10 px-1.5 py-0.5 text-[11px] text-amber-200/80"
                  title="可预判实体：若已被证据覆盖则该步复用跳过"
                >
                  预判 {{ st.entity }}
                </span>
                <span
                  v-if="st.status === 'covered'"
                  class="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[11px] text-emerald-200"
                >
                  已覆盖（复用跳过）
                </span>
                <span
                  v-else-if="st.status === 'unexecuted'"
                  class="rounded bg-slate-600/30 px-1.5 py-0.5 text-[11px] text-slate-400"
                >
                  超预算未执行
                </span>
              </div>
              <p v-if="s.plan.reason" class="text-slate-500">{{ s.plan.reason }}</p>
            </div>
            <p v-else class="mt-1.5 text-slate-500">本轮未产生多跳计划</p>
          </template>
        </section>

        <!-- 多跳迭代检索（modular：逐跳子查询 + 命中，多轮召回拼出中间环节） -->
        <section v-else-if="s.kind === 'multi_hop'" class="rounded-xl border border-rose-500/20 bg-rose-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-rose-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-rose-500/20 px-1.5 py-0.5 text-[10px] font-normal text-rose-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <div v-if="s.hops?.length" class="mt-2 space-y-3">
            <div
              v-for="(hop, hi) in s.hops"
              :key="hi"
              class="border-l-2 border-rose-500/30 pl-3"
            >
              <p class="font-medium text-rose-300">
                第 {{ hi + 1 }} 跳：{{ hop.query }}
                <span v-if="hop.target" class="ml-1 font-normal text-rose-300/70">
                  （目标：{{ hop.target }}）
                </span>
                <span
                  v-if="hop.skipped"
                  class="ml-1 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[11px] font-normal text-emerald-300"
                  title="该步目标已被既有证据覆盖，复用跳过（不重复检索）"
                >
                  已覆盖，跳过
                </span>
                <span v-if="hop.next_query" class="ml-1 font-normal text-rose-300/70">
                  → 续查：{{ hop.next_query }}
                </span>
              </p>
              <ul v-if="hop.hits.length" class="mt-1.5 space-y-1.5">
                <li v-for="(h, i) in hop.hits" :key="i" class="rounded-lg bg-black/20 p-2">
                  <p class="text-slate-300">{{ h.text }}</p>
                  <p class="mt-0.5 text-[11px] text-cyan-400/80">
                    相关度 {{ typeof h.score === 'number' ? h.score.toFixed(3) : h.score }}
                  </p>
                </li>
              </ul>
              <p v-else class="mt-1.5 text-slate-500">该跳未召回相关内容</p>
            </div>
          </div>
          <p v-else class="mt-1.5 text-slate-500">本轮多跳检索未产生逐跳记录</p>
        </section>

        <!-- 多跳验证（modular：规划-执行-验证的验证阶段，质量闸门：覆盖对表 + 补缺子查询） -->
        <section v-else-if="s.kind === 'multi_hop_verify'" class="rounded-xl border border-rose-500/20 bg-rose-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-rose-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-rose-500/20 px-1.5 py-0.5 text-[10px] font-normal text-rose-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <div v-if="s.verification" class="mt-2 space-y-2">
            <p class="flex flex-wrap items-center gap-1.5">
              <span class="text-slate-500">已覆盖：</span>
              <template v-if="s.verification.covered?.length">
                <span
                  v-for="(c, ci) in s.verification.covered"
                  :key="ci"
                  class="rounded bg-emerald-500/10 px-1.5 py-0.5 text-emerald-200"
                >
                  {{ c }}
                </span>
              </template>
              <span v-else class="text-slate-600">—</span>
            </p>
            <p class="flex flex-wrap items-center gap-1.5">
              <span class="text-slate-500">缺口：</span>
              <template v-if="s.verification.missing?.length">
                <span
                  v-for="(m, mi) in s.verification.missing"
                  :key="mi"
                  class="rounded bg-rose-500/10 px-1.5 py-0.5 text-rose-200"
                >
                  {{ m }}
                </span>
              </template>
              <span v-else class="text-slate-600">—</span>
            </p>
            <div v-if="s.verification.patched?.length" class="space-y-1.5">
              <p class="text-slate-500">补缺子查询（局部修正）：</p>
              <div
                v-for="(p, pi) in s.verification.patched"
                :key="pi"
                class="rounded-lg bg-black/20 p-2"
              >
                <span v-if="p.target" class="mr-1.5 rounded bg-rose-500/10 px-1.5 py-0.5 text-rose-200">
                  {{ p.target }}
                </span>
                <span class="text-slate-300">{{ p.query }}</span>
              </div>
            </div>
          </div>
          <p v-else class="mt-1.5 text-slate-500">本轮未产生多跳验证结果</p>
        </section>

        <!-- 上下文压缩（modular：多路召回后控制进入 LLM 的噪声） -->
        <section v-else-if="s.kind === 'compress'" class="rounded-xl border border-orange-500/20 bg-orange-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-orange-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-orange-500/20 px-1.5 py-0.5 text-[10px] font-normal text-orange-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <p v-if="s.metrics" class="mt-1.5 text-slate-300">
            {{ s.metrics.original }} 条候选 → 保留 {{ s.metrics.kept }} 条
            <template v-if="s.metrics.truncated > 0">（截断超长 {{ s.metrics.truncated }} 条）</template>
          </p>
        </section>

        <!-- 答案充分性验证（modular：检索后质量闸门，跨复杂度路径统一检查可答/升级/澄清） -->
        <section v-else-if="s.kind === 'answerability'" class="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-emerald-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-normal text-emerald-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <div v-if="s.verdict" class="mt-2 space-y-1.5">
            <p class="flex flex-wrap items-center gap-1.5">
              <span
                class="rounded px-1.5 py-0.5 font-medium"
                :class="RECOMMENDATION_CLS[s.verdict.recommendation] ?? 'bg-slate-800 text-slate-300'"
              >
                {{ RECOMMENDATION_LABEL[s.verdict.recommendation] ?? s.verdict.recommendation }}
              </span>
              <span
                v-if="s.escalated"
                class="rounded bg-amber-500/10 px-1.5 py-0.5 text-amber-200"
                title="初始检索不足，已按验证建议升级检索路径后复核"
              >
                已升级检索
              </span>
              <span
                v-if="s.verdict.answerable"
                class="text-emerald-400/80"
              >
                检索内容足以回答
              </span>
              <span
                v-else-if="s.verdict.escalate_to"
                class="text-slate-500"
              >
                建议升级：{{ s.verdict.escalate_to }}
              </span>
            </p>
            <div v-if="s.verdict.missing_facts?.length" class="rounded-lg bg-black/20 p-2">
              <p class="text-slate-500">缺失的关键信息：</p>
              <ul class="mt-1 space-y-0.5">
                <li
                  v-for="(f, fi) in s.verdict.missing_facts"
                  :key="fi"
                  class="list-inside list-disc text-rose-300"
                >
                  {{ f }}
                </li>
              </ul>
            </div>
          </div>
          <p v-else class="mt-1.5 text-slate-500">本轮未产生验证结论</p>
        </section>

        <!-- Agent 检索（agentic：检索 Agent 单步工具执行，逐步流式追加） -->
        <section v-else-if="s.kind === 'agent_step'" class="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-amber-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-normal text-amber-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <div v-if="s.agentSteps?.length" class="mt-2 space-y-2">
            <div
              v-for="(st, si) in s.agentSteps"
              :key="si"
              class="border-l-2 border-amber-500/30 pl-3"
            >
              <p class="flex flex-wrap items-center gap-1.5 font-medium text-amber-300">
                <span class="text-slate-500">#{{ st.index }}</span>
                <span class="rounded bg-amber-500/10 px-1.5 py-0.5 font-normal">{{ ROLE_LABEL[st.role] ?? st.role }}</span>
                <span class="rounded bg-slate-800 px-1.5 py-0.5 font-normal text-slate-300">
                  {{ ACTION_LABEL[st.action] ?? st.action }}
                </span>
                <span v-if="st.params?.query" class="font-normal text-slate-300">「{{ st.params.query }}」</span>
                <span v-if="st.params?.volume" class="rounded bg-cyan-500/10 px-1.5 py-0.5 font-normal text-cyan-200">
                  卷 {{ st.params.volume }}
                </span>
              </p>
              <p class="mt-1 flex flex-wrap items-center gap-1.5 text-slate-400">
                <span v-if="typeof st.hits_count === 'number'">
                  {{ st.hits_count }} 条命中
                </span>
                <span
                  v-for="(v, vi) in st.volumes ?? []"
                  :key="vi"
                  class="rounded bg-slate-800 px-1.5 py-0.5 text-[11px] text-slate-400"
                >
                  {{ v.volume }}×{{ v.count }}
                </span>
                <span
                  v-if="st.note"
                  class="rounded bg-rose-500/10 px-1.5 py-0.5 text-[11px] text-rose-300"
                  title="护栏拦截/降级原因"
                >
                  {{ st.note }}
                </span>
              </p>
            </div>
          </div>
          <p v-else class="mt-1.5 text-slate-500">检索 Agent 尚未执行工具调用</p>
        </section>

        <!-- 证据评审（agentic：CRAG 检索评估器，逐条证据相关性 + 缺口归纳） -->
        <section v-else-if="s.kind === 'grade'" class="rounded-xl border border-violet-500/20 bg-violet-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-violet-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-normal text-violet-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <div class="mt-2 space-y-1.5">
            <p class="flex flex-wrap items-center gap-1.5">
              <span class="text-slate-500">相关证据：</span>
              <span class="rounded bg-violet-500/10 px-1.5 py-0.5 text-violet-200">
                {{ s.kept ?? 0 }} / {{ s.total ?? 0 }}
              </span>
            </p>
            <p v-if="s.missing_facts?.length" class="flex flex-wrap items-center gap-1.5">
              <span class="text-slate-500">可补充细节（供纠错参考）：</span>
              <span
                v-for="(m, mi) in s.missing_facts"
                :key="mi"
                class="rounded bg-amber-500/10 px-1.5 py-0.5 text-amber-200"
              >
                {{ m }}
              </span>
            </p>
            <p v-if="s.thought" class="text-slate-500">{{ s.thought }}</p>
          </div>
        </section>

        <!-- 纠错决策（agentic：CRAG 纠正分支，证据不足时给出下一波检索调用） -->
        <section v-else-if="s.kind === 'correct'" class="rounded-xl border border-orange-500/20 bg-orange-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-orange-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-orange-500/20 px-1.5 py-0.5 text-[10px] font-normal text-orange-200"
              >
                {{ s.scheme }}
              </span>
              <span
                v-if="s.round"
                class="rounded bg-orange-500/10 px-1.5 py-0.5 text-[10px] font-normal text-orange-200"
              >
                第 {{ s.round }} 轮
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <div v-if="s.calls?.length" class="mt-2 space-y-1.5">
            <div
              v-for="(c, ci) in s.calls"
              :key="ci"
              class="rounded-lg bg-black/20 p-2"
            >
              <p class="flex flex-wrap items-center gap-1.5">
                <span class="rounded bg-orange-500/10 px-1.5 py-0.5 text-orange-200">
                  {{ ACTION_LABEL[c.action] ?? c.action }}
                </span>
                <span v-if="c.volume" class="rounded bg-cyan-500/10 px-1.5 py-0.5 text-cyan-200">卷 {{ c.volume }}</span>
                <span class="text-slate-300">「{{ c.query }}」</span>
              </p>
              <p v-if="c.reason" class="mt-0.5 text-[11px] text-slate-500">{{ c.reason }}</p>
            </div>
          </div>
          <p v-if="s.thought" class="mt-1.5 text-slate-500">{{ s.thought }}</p>
          <p v-else class="mt-1.5 text-slate-500">本轮无可用纠错调用</p>
        </section>

        <!-- 答案校验（agentic：Self-RAG 支持度校验，事实-证据支持度矩阵） -->
        <section v-else-if="s.kind === 'verify'" class="rounded-xl border border-teal-500/20 bg-teal-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-teal-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-teal-500/20 px-1.5 py-0.5 text-[10px] font-normal text-teal-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <div class="mt-2 space-y-1.5">
            <p class="flex flex-wrap items-center gap-1.5">
              <span
                class="rounded px-1.5 py-0.5 font-medium"
                :class="s.answerable ? 'bg-emerald-500/20 text-emerald-200' : 'bg-rose-500/20 text-rose-200'"
              >
                {{ s.answerable ? '证据足以回答' : '证据不足，需追问澄清' }}
              </span>
            </p>
            <p v-if="s.missing_facts?.length" class="flex flex-wrap items-center gap-1.5">
              <span class="text-slate-500">阻断性缺口（需追问澄清）：</span>
              <span
                v-for="(m, mi) in s.missing_facts"
                :key="mi"
                class="rounded bg-rose-500/10 px-1.5 py-0.5 text-rose-200"
              >
                {{ m }}
              </span>
            </p>
            <p v-if="s.thought" class="text-slate-500">{{ s.thought }}</p>
          </div>
        </section>

        <!-- Query 重写结果（advanced：独立步骤，先于知识库检索展示） -->
        <section v-else-if="s.kind === 'rewrite'" class="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-cyan-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-cyan-500/20 px-1.5 py-0.5 text-[10px] font-normal text-cyan-200"
              >
                {{ s.scheme }}
              </span>
              <span
                v-if="s.reason"
                class="rounded bg-cyan-500/20 px-1.5 py-0.5 text-[10px] font-normal text-cyan-200"
              >
                {{ s.reason }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <div class="mt-2 flex flex-wrap items-center gap-1.5">
            <span
              v-for="(r, ri) in s.rewrites"
              :key="ri"
              class="rounded bg-cyan-500/10 px-1.5 py-0.5 text-cyan-200"
            >
              {{ r }}
            </span>
          </div>
        </section>

        <!-- HyDE 假想文档检索（modular 召回阶段隐式一路：LLM 生成假想答案文档 → doc-space 稠密召回） -->
        <section v-else-if="s.kind === 'hyde'" class="rounded-xl border border-sky-500/30 bg-sky-500/5 p-3 text-xs">
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-sky-300">
              <svg
                v-if="s.running"
                class="h-3.5 w-3.5 shrink-0 animate-spin text-sky-400"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-90" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
              </svg>
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.scheme"
                class="rounded bg-sky-500/20 px-1.5 py-0.5 text-[10px] font-normal text-sky-200"
              >
                {{ s.scheme }}
              </span>
            </span>
            <span v-if="s.query" class="break-all text-slate-500">query: {{ s.query }}</span>
          </div>
          <!-- 假想文档为纯 LLM 阻塞调用：running 占位转圈，done 后再填充 -->
          <p v-if="s.running" class="mt-2 text-sky-300/70">生成假想答案文档…</p>
          <template v-else-if="s.fired">
            <p class="mt-2 text-slate-400">假想文档 → doc-space 稠密召回 {{ s.recall ?? 0 }} 条（并入 RRF 融合）：</p>
            <p class="mt-1 whitespace-pre-wrap rounded-lg bg-black/20 p-2 text-slate-300">{{ s.doc }}</p>
          </template>
          <p v-else class="mt-2 text-slate-500">未生成假想文档（回退原查询，跳过 HyDE 一路）</p>
        </section>

        <!-- 知识库检索（RAG） -->
        <section
          v-else-if="s.kind === 'retrieve' || s.kind === 'memory_read' || s.kind === 'memory_write'"
          class="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3 text-xs"
        >
          <div class="flex items-center justify-between gap-2">
            <span class="flex items-center gap-1.5 font-medium text-cyan-300">
              {{ RETRIEVE_KIND[s.kind] }}
              <span
                v-if="s.kind === 'retrieve' && s.scheme"
                class="rounded bg-cyan-500/20 px-1.5 py-0.5 text-[10px] font-normal text-cyan-200"
                :title="`当前 RAG 方案：${s.scheme}`"
              >
                {{ s.scheme }}
              </span>
              <span
                v-if="s.kind === 'retrieve' && s.reranked"
                class="rounded bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-normal text-violet-200"
                title="多路召回后经交叉编码器重排精排"
              >
                重排
              </span>
            </span>
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
          <p
            v-else-if="s.kind === 'retrieve'"
            class="mt-1.5 text-slate-500"
          >
            未检索到相关内容（本轮不注入知识库上下文）
          </p>
        </section>

        <!-- 反思意见（评审过程流式展示，流式中显示转圈动效） -->
        <section v-else-if="s.kind === 'reflect'" class="rounded-xl border border-rose-500/30 bg-rose-500/5 p-3 text-xs">
          <p class="flex items-center gap-1.5 text-rose-300">
            <svg
              v-if="s.streaming && !s.stage"
              class="h-3 w-3 shrink-0 animate-spin text-rose-400"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
              <path class="opacity-90" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
            </svg>
            {{ s.stage ? `阶段：${s.stage}` : '评审意见' }}
          </p>
          <p v-if="s.critique" class="mt-1 whitespace-pre-wrap text-slate-300">{{ s.critique }}</p>
        </section>

        <!-- 修订稿 -->
        <section v-else-if="s.kind === 'revise'" class="rounded-xl border border-orange-500/30 bg-orange-500/5 p-3 text-xs">
          <p class="text-orange-300">修订稿：</p>
          <div class="mt-1 text-slate-200">
            <StreamingText :text="s.text ?? ''" :streaming="s.streaming ?? false" markdown />
          </div>
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
            <div class="text-sm leading-relaxed text-slate-100">
              <StreamingText :text="s.text ?? ''" :streaming="s.streaming ?? false" placeholder="等待回答…" markdown />
            </div>
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
