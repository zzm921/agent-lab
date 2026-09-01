/** 对话流状态机：连接 / 接收 / rAF 节流累加 / 流水线步骤 / 错误 / 审批 / 完成。 */
import { reactive } from 'vue'
import { streamEvents } from '../services/sse'
import type {
  AgentEvent,
  ApprovalPolicy,
  ApprovalRequest,
  HitItem,
  ModeId,
  PromptStrategy,
  RagSchemeId,
} from '../types/agent'

export type StreamStatus = 'idle' | 'streaming' | 'waiting_approval' | 'done' | 'error'

/** 工具调用条目（ToolCallBadge 使用） */
export interface ToolCallEntry {
  id: number
  tool: string
  args: Record<string, unknown>
  status: 'running' | 'success' | 'failed' | 'rejected'
  result?: string
  /** 工具层透明重试进度（tool_retry 事件更新） */
  retryCount?: number
  retryMax?: number
  /** 本次重试实际等待秒数（含抖动） */
  retryDelay?: number
  /** 本次重试纯指数退避秒数（不含抖动，用于展示退避曲线） */
  retryBaseDelay?: number
  retryReason?: string
}

/** 流水线步骤类型：按发生顺序记录，用户输入与思考/工具/输出交替呈现 */
export type StepKind =
  | 'user' // 用户输入（右侧气泡）
  | 'thinking' // 思考过程（reason，灰色斜体，流式增量）
  | 'message' // 最终输出（output，流式增量）
  | 'revise' // 反思修订稿（流式增量）
  | 'tool' // 工具调用（running → success/failed/rejected）
  | 'plan' // 执行计划
  | 'retrieve' // RAG 检索
  | 'rewrite' // Query 重写结果
  | 'classify' // 查询语义路由（五维决策，modular）
  | 'decompose' // Query 分解子问题（modular）
  | 'multi_hop_plan' // 多跳检索计划（规划-执行-验证，modular）
  | 'multi_hop' // 多跳迭代检索（modular）
  | 'multi_hop_verify' // 多跳质量闸门验证（规划-执行-验证，modular）
  | 'compress' // 上下文压缩统计（modular）
  | 'answerability' // 检索后答案充分性验证（跨复杂度路径质量闸门，modular）
  | 'agent_step' // 检索 Agent 单步工具执行（agentic）
  | 'grade' // CRAG 证据评审（agentic）
  | 'correct' // CRAG 纠错决策（agentic）
  | 'verify' // Self-RAG 答案校验（agentic）
  | 'memory_read' // 记忆召回
  | 'memory_write' // 记忆写入
  | 'reflect' // 反思意见
  | 'agent_event' // 多智能体 worker 事件

export interface StepEntry {
  id: number
  kind: StepKind
  /** thinking / message / revise 的流式文本 */
  text?: string
  streaming?: boolean
  /** tool */
  tool?: string
  args?: Record<string, unknown>
  status?: 'running' | 'success' | 'failed' | 'rejected'
  result?: string
  /** 工具层透明重试进度（tool_retry 事件更新） */
  retryCount?: number
  retryMax?: number
  /** 本次重试实际等待秒数（含抖动） */
  retryDelay?: number
  /** 本次重试纯指数退避秒数（不含抖动，用于展示退避曲线） */
  retryBaseDelay?: number
  retryReason?: string
  /** plan */
  steps?: string[]
  currentStep?: number
  planStatus?: string
  /** classify / multi_hop_plan：running 占位（阶段进行中，内容未出，卡片显示转圈） */
  running?: boolean
  /** retrieve / memory_read */
  query?: string
  hits?: HitItem[]
  /** retrieve：本轮使用的 RAG 方案 id */
  scheme?: string
  /** retrieve：Query 重写变体（advanced 有值） */
  rewrites?: string[]
  /** retrieve：是否经过重排 */
  reranked?: boolean
  /** classify：五维路由决策（D1 是否检索 / D3 检索策略 / D4 复杂度 / D5 生成模式 + 置信度） */
  retrieval_need?: boolean
  retrieval_mode?: string
  complexity?: string
  generation_mode?: string
  confidence?: number
  /** classify：判定理由；rewrite：改写原因（modular 指代消解时为「指代消解」） */
  reason?: string
  /** decompose：查询分解子问题 */
  sub_queries?: string[]
  /** multi_hop_plan：多跳检索计划（规划-执行-验证） */
  plan?: {
    steps: { target: string; query: string; entity?: string | null; depends_on?: string[]; status?: string }[]
    reason?: string
  }
  /** multi_hop：多跳迭代检索逐跳记录（每跳子查询 + 命中 + 目标 + 覆盖复用标记），前端逐跳流式追加 */
  hops?: { query: string; hits: HitItem[]; next_query?: string | null; target?: string | null; skipped?: boolean }[]
  /** multi_hop_verify：多跳质量闸门结果（covered / missing / patched） */
  verification?: { covered: string[]; missing: string[]; patched: { target: string; query: string }[] }
  /** compress：上下文压缩统计（original / kept / truncated） */
  metrics?: { original: number; kept: number; truncated: number }
  /** answerability：检索后答案充分性验证（answerable / missing_facts / recommendation / escalated） */
  verdict?: { answerable: boolean; missing_facts: string[]; recommendation: string; escalate_to?: string | null }
  escalated?: boolean
  /** agent_step：检索 Agent 单步工具执行记录（同一卡片逐步追加） */
  agentSteps?: {
    index: number
    role: string
    action: string
    params: Record<string, unknown>
    note?: string | null
    hits_count?: number
    volumes?: { volume: string; count: number }[]
  }[]
  /** grade：CRAG 证据评审（保留相关证据数 / 候选总数 / 缺口 / 理由） */
  kept?: number
  total?: number
  missing_facts?: string[]
  thought?: string
  /** correct：CRAG 纠错决策（纠错轮次 + 下一波工具调用） */
  round?: number
  calls?: { action: string; query: string; volume?: string; reason?: string }[]
  /** verify：Self-RAG 答案校验（可答 / 缺口） */
  answerable?: boolean
  /** memory_write */
  content?: string
  /** reflect */
  stage?: string
  critique?: string
  /** agent_event */
  worker?: string
  agentStatus?: string
  task?: string
  agentResult?: string
}

export interface ErrorInfo {
  message: string
  detail?: string
}

export interface SendParams {
  message: string
  mode: ModeId
  enabled: string[]
  strategy: PromptStrategy
  policy: ApprovalPolicy
  ragScheme: RagSchemeId
  /** 本轮是否启用知识库检索（RAG 前置检索），后端能力默认开启，由前端开关控制 */
  ragEnabled: boolean
  /** 覆盖会话 id（对比视图每个 runner 独立会话） */
  sessionId?: string
}

export interface ChatStream {
  status: StreamStatus
  sessionId: string
  mode: ModeId
  ragScheme: RagSchemeId
  ragEnabled: boolean
  /** 流水线时间线：按事件发生顺序排列的步骤 */
  steps: StepEntry[]
  done: { summary: string; stats: Record<string, unknown> } | null
  error: ErrorInfo | null
  approval: ApprovalRequest | null
  elapsed: number
  enabled: string[]
  strategy: PromptStrategy
  policy: ApprovalPolicy
  send: (params: SendParams) => Promise<void>
  decide: (decision: 'approve' | 'reject' | 'modify', modifiedArgs?: Record<string, unknown>) => Promise<void>
  stop: () => void
  retry: () => void
}

export function genId(): string {
  const c = globalThis.crypto as Crypto | undefined
  if (c && typeof c.randomUUID === 'function') return c.randomUUID()
  return 'id-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10)
}

/** rAF 节流的文本累加器：缓冲原始串，下一帧一次性写入当前步骤 */
function makeStepAccumulator(append: (text: string) => void) {
  let raf = 0
  let raw = ''
  return {
    add(chunk: string) {
      raw += chunk
      if (!raf) {
        raf = requestAnimationFrame(() => {
          raf = 0
          const s = raw
          raw = ''
          append(s)
        })
      }
    },
    flush() {
      if (raf) {
        cancelAnimationFrame(raf)
        raf = 0
      }
      if (raw) {
        const s = raw
        raw = ''
        append(s)
      }
    },
  }
}

export function useChatStream(): ChatStream {
  let stream: ChatStream
  let controller: AbortController | null = null
  let lastParams: SendParams | null = null
  let seq = 0
  let timer: ReturnType<typeof setInterval> | null = null

  /** 关闭当前正在流式输出的文本步骤（新步骤出现或流程结束时调用） */
  function closeStreamingStep() {
    const prev = stream.steps[stream.steps.length - 1]
    if (prev && prev.streaming) prev.streaming = false
  }

  /** 追加流式文本：与当前正在流的同类步骤合并，否则新开一步（流水线） */
  function appendText(kind: 'thinking' | 'message' | 'revise', text: string) {
    const last = stream.steps[stream.steps.length - 1]
    if (last && last.kind === kind && last.streaming) {
      last.text = (last.text ?? '') + text
    } else {
      closeStreamingStep()
      stream.steps.push({ id: ++seq, kind, text, streaming: true })
    }
  }

  /** 追加流式评审文本：写入当前反思（reflect）步骤的 critique 字段（评审过程流式展示） */
  function appendCritique(text: string) {
    const last = stream.steps[stream.steps.length - 1]
    if (last && last.kind === 'reflect' && last.streaming) {
      last.critique = (last.critique ?? '') + text
    } else {
      closeStreamingStep()
      stream.steps.push({ id: ++seq, kind: 'reflect', critique: text, streaming: true })
    }
  }

  /** 新增非流式步骤（工具/计划/检索/反思/代理等），并关闭上一个流式步骤 */
  function pushStep(step: Omit<StepEntry, 'id' | 'streaming'>) {
    // 先把积压的流式文本按事件顺序落定：rAF 节流延迟时若直接压入新步骤，
    // 会使 message/revise 等文本步骤排到后续步骤之后（如两条评审中间缺修订稿）。
    accThinking.flush()
    accMessage.flush()
    accCritique.flush()
    accRevise.flush()
    closeStreamingStep()
    stream.steps.push({ ...step, id: ++seq, streaming: false })
  }

  const accThinking = makeStepAccumulator((text) => appendText('thinking', text))
  const accMessage = makeStepAccumulator((text) => appendText('message', text))
  const accRevise = makeStepAccumulator((text) => appendText('revise', text))
  const accCritique = makeStepAccumulator((text) => appendCritique(text))

  function flushAll() {
    accThinking.flush()
    accMessage.flush()
    accCritique.flush()
    accRevise.flush()
  }

  function finishStreaming() {
    flushAll()
    closeStreamingStep()
  }

  function startTimer() {
    stream.elapsed = 0
    stopTimer()
    timer = setInterval(() => {
      if (stream.status === 'streaming' || stream.status === 'waiting_approval') {
        stream.elapsed += 0.5
      }
    }, 500)
  }

  function stopTimer() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  function handleEvent(ev: AgentEvent) {
    switch (ev.type) {
      case 'meta':
        stream.sessionId = ev.session_id
        stream.mode = ev.mode as ModeId
        stream.enabled = ev.capabilities
        if (ev.rag_scheme) stream.ragScheme = ev.rag_scheme as RagSchemeId
        if (ev.rag_enabled !== undefined) stream.ragEnabled = ev.rag_enabled
        break
      case 'thinking':
        accThinking.add(ev.delta)
        break
      case 'message':
        accMessage.add(ev.delta)
        break
      case 'revise':
        accRevise.add(ev.delta)
        break
      case 'critique':
        accCritique.add(ev.delta)
        break
      case 'plan': {
        const last = stream.steps[stream.steps.length - 1]
        if (last && last.kind === 'plan') {
          // 同一计划就地更新进度，保持在流水线中的原始位置
          last.steps = ev.steps
          last.currentStep = ev.current_step
          last.planStatus = ev.status
        } else {
          pushStep({
            kind: 'plan',
            steps: ev.steps,
            currentStep: ev.current_step,
            planStatus: ev.status,
          })
        }
        break
      }
      case 'tool_start':
        pushStep({ kind: 'tool', tool: ev.tool, args: ev.args, status: 'running' })
        break
      case 'tool_end': {
        for (let i = stream.steps.length - 1; i >= 0; i--) {
          const s = stream.steps[i]
          if (s.kind === 'tool' && s.tool === ev.tool && s.status === 'running') {
            s.status = ev.success ? 'success' : 'failed'
            s.result = ev.result
            break
          }
        }
        break
      }
      case 'tool_retry': {
        // 工具层透明重试进度：更新正在执行中的工具卡片（同一步骤就地更新）
        for (let i = stream.steps.length - 1; i >= 0; i--) {
          const s = stream.steps[i]
          if (s.kind === 'tool' && s.tool === ev.tool && s.status === 'running') {
            s.retryCount = ev.attempt
            s.retryMax = ev.max
            s.retryDelay = ev.delay
            s.retryBaseDelay = ev.base_delay ?? ev.delay
            s.retryReason = ev.reason
            break
          }
        }
        break
      }
      case 'retrieve':
        pushStep({ kind: 'retrieve', query: ev.query, hits: ev.hits, scheme: ev.scheme, reranked: ev.reranked })
        break
      case 'rewrite':
        pushStep({ kind: 'rewrite', query: ev.query, scheme: ev.scheme, rewrites: ev.rewrites, reason: ev.reason })
        break
      case 'classify': {
        // 语义路由：先收到 running 占位（卡片转圈），完成后再收到 done 就地填充同一张卡片
        const last = stream.steps[stream.steps.length - 1]
        if (last && last.kind === 'classify' && last.running) {
          last.running = false
          last.retrieval_need = ev.retrieval_need
          last.retrieval_mode = ev.retrieval_mode
          last.complexity = ev.complexity
          last.generation_mode = ev.generation_mode
          last.confidence = ev.confidence
          last.reason = ev.reason
        } else {
          pushStep({
            kind: 'classify',
            query: ev.query,
            scheme: ev.scheme,
            running: ev.status === 'running',
            retrieval_need: ev.retrieval_need,
            retrieval_mode: ev.retrieval_mode,
            complexity: ev.complexity,
            generation_mode: ev.generation_mode,
            confidence: ev.confidence,
            reason: ev.reason,
          })
        }
        break
      }
      case 'decompose':
        pushStep({ kind: 'decompose', query: ev.query, scheme: ev.scheme, sub_queries: ev.sub_queries })
        break
      case 'multi_hop_plan': {
        // 规划-执行-验证：先收到 running 占位（卡片转圈），完成后再收到 done 就地填充计划
        const last = stream.steps[stream.steps.length - 1]
        if (last && last.kind === 'multi_hop_plan' && last.running) {
          last.running = false
          last.plan = ev.plan
        } else {
          pushStep({
            kind: 'multi_hop_plan',
            query: ev.query,
            scheme: ev.scheme,
            running: ev.status === 'running',
            plan: ev.plan,
          })
        }
        break
      }
      case 'multi_hop': {
        // 逐跳流式：每跳一个事件，追加到当前多跳卡片（就地填充，逐跳呈现而非一次性返回全部跳）
        const last = stream.steps[stream.steps.length - 1]
        if (last && last.kind === 'multi_hop') {
          last.hops = [...(last.hops ?? []), ev.hop]
        } else {
          pushStep({ kind: 'multi_hop', query: ev.query, scheme: ev.scheme, hops: [ev.hop] })
        }
        break
      }
      case 'multi_hop_verify':
        // 规划-执行-验证：最后展示质量闸门结果（覆盖对表 + 补缺子查询）
        pushStep({
          kind: 'multi_hop_verify',
          query: ev.query,
          scheme: ev.scheme,
          verification: ev.verification,
        })
        break
      case 'compress':
        pushStep({ kind: 'compress', query: ev.query, scheme: ev.scheme, metrics: ev.metrics })
        break
      case 'answerability':
        // 检索后答案充分性验证：展示最终结论（可答 / 升级检索 / 追问澄清）与缺失事实
        pushStep({
          kind: 'answerability',
          query: ev.query,
          scheme: ev.scheme,
          verdict: ev.verdict,
          escalated: ev.escalated,
        })
        break
      case 'agent_step': {
        // agentic：检索 Agent 单步工具执行，逐步流式追加到同一张卡片（保持流水线原始位置）
        const lastStep = stream.steps[stream.steps.length - 1]
        if (lastStep && lastStep.kind === 'agent_step') {
          lastStep.agentSteps = [...(lastStep.agentSteps ?? []), ev.step]
        } else {
          pushStep({ kind: 'agent_step', query: ev.query, scheme: ev.scheme, agentSteps: [ev.step] })
        }
        break
      }
      case 'grade':
        // agentic：CRAG 证据评审（保留相关证据数 / 候选总数 / 缺口）
        pushStep({
          kind: 'grade',
          query: ev.query,
          scheme: ev.scheme,
          kept: ev.kept,
          total: ev.total,
          missing_facts: ev.missing_facts,
          thought: ev.thought,
        })
        break
      case 'correct':
        // agentic：CRAG 纠错决策（纠错轮次 + 下一波工具调用）
        pushStep({
          kind: 'correct',
          query: ev.query,
          scheme: ev.scheme,
          round: ev.round,
          thought: ev.thought,
          calls: ev.calls,
        })
        break
      case 'verify':
        // agentic：Self-RAG 答案校验（可答 / 缺口）
        pushStep({
          kind: 'verify',
          query: ev.query,
          scheme: ev.scheme,
          answerable: ev.answerable,
          missing_facts: ev.missing_facts,
          thought: ev.thought,
        })
        break
      case 'memory_write':
        pushStep({ kind: 'memory_write', content: ev.content })
        break
      case 'memory_read':
        pushStep({ kind: 'memory_read', query: ev.query, hits: ev.hits })
        break
      case 'reflect':
        pushStep({ kind: 'reflect', stage: ev.stage, critique: ev.critique })
        break
      case 'agent_event':
        pushStep({
          kind: 'agent_event',
          worker: ev.worker,
          agentStatus: ev.status,
          task: ev.task,
          agentResult: ev.result,
        })
        break
      case 'approval_request':
        stream.approval = { approval_id: ev.approval_id, tool_calls: ev.tool_calls }
        break
      case 'done':
        stream.done = { summary: ev.summary, stats: ev.stats }
        break
      case 'error':
        stream.error = { message: ev.message, detail: ev.detail }
        break
    }
  }

  async function consume(url: string, body: unknown, signal: AbortSignal) {
    try {
      for await (const ev of streamEvents(url, body, signal)) {
        if (signal.aborted) break
        handleEvent(ev)
        if (stream.approval) {
          stream.status = 'waiting_approval'
          finishStreaming()
          return
        }
      }
      finishStreaming()
      stopTimer()
      if (signal.aborted) return
      if (stream.error) stream.status = 'error'
      else if (stream.done) stream.status = 'done'
      else stream.status = 'idle'
    } catch (err) {
      finishStreaming()
      stopTimer()
      if (signal.aborted) return
      stream.status = 'error'
      stream.error = {
        message: '连接失败或后端返回错误',
        detail: err instanceof Error ? err.message : String(err),
      }
    }
  }

  /** 启动/重试一轮：清空瞬态状态（保留对话历史），按需追加用户消息步骤 */
  async function run(params: SendParams, withUserStep: boolean) {
    stop()
    lastParams = params
    stream.done = null
    stream.error = null
    stream.approval = null
    stream.elapsed = 0
    if (withUserStep) pushStep({ kind: 'user', text: params.message })
    if (params.sessionId) stream.sessionId = params.sessionId
    else if (!stream.sessionId) stream.sessionId = genId()
    stream.mode = params.mode
    stream.strategy = params.strategy
    stream.policy = params.policy
    stream.status = 'streaming'
    startTimer()
    controller = new AbortController()
    await consume(
      '/api/stream',
      {
        session_id: stream.sessionId,
        message: params.message,
        mode: params.mode,
        enabled_capabilities: params.enabled,
        prompt_strategy: params.strategy,
        approval_policy: params.policy,
        rag_scheme: params.ragScheme,
        rag_enabled: params.ragEnabled,
      },
      controller.signal,
    )
  }

  const send = (params: SendParams) => run(params, true)

  const decide = async (
    decision: 'approve' | 'reject' | 'modify',
    modifiedArgs?: Record<string, unknown>,
  ) => {
    const a = stream.approval
    if (!a) return
    stream.approval = null
    stream.status = 'streaming'
    controller = new AbortController()
    await consume(
      '/api/approve',
      { approval_id: a.approval_id, decision, modified_args: modifiedArgs ?? null },
      controller.signal,
    )
  }

  const stop = () => {
    const sid = stream.sessionId
    if (controller) {
      controller.abort()
      controller = null
    }
    // 通知后端立即取消该会话的后台执行，及时停止以节省 token
    if (sid) {
      void fetch('/api/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid }),
      }).catch(() => {})
    }
    finishStreaming()
    stopTimer()
    if (stream.status === 'streaming' || stream.status === 'waiting_approval') {
      stream.status = 'idle'
    }
  }

  const retry = () => {
    if (lastParams) void run({ ...lastParams }, false)
  }

  stream = reactive<ChatStream>({
    status: 'idle',
    sessionId: '',
    mode: 'react',
    ragScheme: 'naive',
    ragEnabled: true,
    steps: [],
    done: null,
    error: null,
    approval: null,
    elapsed: 0,
    enabled: [],
    strategy: 'standard',
    policy: 'always',
    send,
    decide,
    stop,
    retry,
  })
  return stream
}
