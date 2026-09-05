/** Agent 平台前端类型定义：能力 / 模式 / SSE 事件协议，与后端严格对齐。 */

export type ModeId = 'react' | 'plan_execute' | 'reflection' | 'multi_agent'
export type PromptStrategy = 'standard' | 'few_shot' | 'cot'
export type ApprovalPolicy = 'always' | 'never'
export type ApprovalDecision = 'approve' | 'reject' | 'modify'
export type RagSchemeId = 'naive' | 'advanced' | 'modular' | 'agentic'

export interface Capability {
  id: string
  name: string
  source: 'builtin' | 'mcp'
  desc: string
  example: string
  code_key: string
  availability: 'available' | 'unavailable'
  unavailable_reason: string | null
  server?: string
}

export interface HitItem {
  text: string
  score: number
  metadata?: Record<string, unknown>
}

/** 单路检索的查询变换明细（pipeline.strategy[].query_pipeline）：检索类型 / 向量体系 / 是否 HyDE / 展开后查询 */
export interface PipelineQueryTransform {
  type?: string
  embedding?: string
  hyde?: boolean
  expanded?: string | null
  hyde_doc?: string | null
}

/** 每路检索策略明细（pipeline.strategy）：动作 / 查询 / 卷 / 护栏备注 / 召回数与命中分数分布 */
export interface PipelineStrategyEntry {
  tool: string
  query: string
  volume?: string | null
  reason?: string
  guarded?: string | null
  recall_k?: number
  hits: number
  scores?: number[]
  query_pipeline?: PipelineQueryTransform
}

/** 实际应用的筛选规则（pipeline.filters）：guard 护栏拦截 / grade 评审剔除 / compress 压缩去重截断 */
export interface PipelineFilter {
  name: string
  dropped?: number
  kept?: number
  total?: number
  original?: number
  truncated?: number
}

/** 最终结果排序依据（pipeline.ranking）：RRF 融合 + 重排模型与前后保留数 */
export interface PipelineRanking {
  fusion?: { method: string; fused: number; keep: number }
  rerank?: { model: string; before: number; after: number }
}

/** 检索链路完整明细（pipeline）：触发条件 → 查询向量生成 → 每路策略 → 筛选 → 排序（agentic） */
export interface PipelineDetail {
  trigger: { retrieval_need: boolean; mode: string; reason: string }
  query_pipeline: { hyde: boolean; embedding: string }
  strategy: PipelineStrategyEntry[]
  filters: PipelineFilter[]
  ranking: PipelineRanking
}

/** 检索任务图节点（agentic knowledge_task：拆解器产出的子查询节点 DAG） */
export interface TaskGraphNode {
  id: string
  query: string
  /** 依赖节点 id（须先执行） */
  deps: string[]
  reason?: string
}

/** 检索任务图完成度（task_done 事件载荷汇总） */
export interface TaskGraphSummary {
  completion: string
  resolved: number
  gaps: number
  confidence: number
}

/** RAG 方案目录项（GET /api/rag/schemes） */
export interface RagScheme {
  id: string
  name: string
  description: string
  collection: string
  count: number
}

export interface ToolCallInfo {
  name: string
  args: Record<string, unknown>
  id?: string
}

export interface ApprovalRequest {
  approval_id: string
  tool_calls: ToolCallInfo[]
}

/** SSE 事件联合类型（data 行 JSON 的结构化描述） */
export type AgentEvent =
  | { type: 'meta'; session_id: string; mode: string; capabilities: string[]; rag_scheme?: string; rag_enabled?: boolean }
  | { type: 'thinking'; delta: string }
  | { type: 'message'; delta: string }
  | { type: 'tool_start'; tool: string; args: Record<string, unknown> }
  | { type: 'tool_end'; tool: string; args?: Record<string, unknown>; result: string; success: boolean }
  | { type: 'tool_retry'; tool: string; attempt: number; max: number; delay: number; base_delay?: number; reason: string }
  | { type: 'plan'; steps: string[]; current_step: number; status: string }
  | { type: 'retrieve'; query: string; scheme?: string; hits: HitItem[]; reranked?: boolean; pipeline?: PipelineDetail; task_id?: string; node_id?: string }
  | { type: 'rewrite'; query: string; scheme?: string; rewrites: string[]; reason?: string }
  | { type: 'seed_reuse'; query: string; scheme?: string; count: number }
  | {
      type: 'classify'
      query: string
      scheme?: string
      /** running：语义路由进行中（占位，决策未出）；done：携带完整五维路由决策 */
      status?: 'running' | 'done'
      retrieval_need?: boolean
      retrieval_mode?: string
      complexity?: string
      generation_mode?: string
      confidence?: number
      reason?: string
    }
  | { type: 'decompose'; query: string; scheme?: string; sub_queries: string[] }
  | {
      type: 'hyde'
      query: string
      scheme?: string
      /** running：LLM 生成假想答案文档中（占位，转圈）；done：携带结果 */
      status?: 'running' | 'done'
      /** 是否真的生成了假想文档并追加一路 doc-space 稠密召回（false = 规则回退原查询，跳过） */
      fired?: boolean
      /** 生成的假想答案文档全文（仅演示展示，不注入答案） */
      doc?: string
      /** 假想文档这一路召回到的条数（RRF 融合前） */
      recall?: number
    }
  | {
      type: 'multi_hop_plan'
      query: string
      scheme?: string
      /** running：多跳规划进行中（占位，计划未出）；done：携带完整计划 */
      status?: 'running' | 'done'
      /** 规划-执行-验证：多跳检索计划（目标/依赖/可预判实体） */
      plan?: {
        steps: {
          target: string
          query: string
          entity?: string | null
          depends_on?: string[]
          status?: string
        }[]
        reason?: string
      }
    }
  | {
      type: 'multi_hop'
      query: string
      scheme?: string
      /** 逐跳流式：每完成一跳下发一个事件，index 从 1 递增 */
      index: number
      hop: { query: string; hits: HitItem[]; next_query?: string | null; target?: string | null; skipped?: boolean }
    }
  | {
      type: 'multi_hop_verify'
      query: string
      scheme?: string
      /** 规划-执行-验证：质量闸门结果（覆盖对表 + 补缺子查询） */
      verification: {
        covered: string[]
        missing: string[]
        patched: { target: string; query: string }[]
      }
    }
  | {
      type: 'compress'
      query: string
      scheme?: string
      metrics: { original: number; kept: number; truncated: number }
    }
  | {
      type: 'context'
      /** 上下文管理与压缩四层：对话修剪 / 旧工具结果占位 / LLM 摘要 / 大输出落盘 */
      kind: 'snip_compact' | 'micro_compact' | 'auto_compact' | 'offload'
      metrics?: { original: number; kept: number; dropped?: number; truncated?: number; threshold?: number; summary?: string }
      /** 每轮压缩演示模式的保留轮数（snip 事件带出） */
      keep_rounds?: number
      tool?: string
      chars?: number
      file?: string
    }
  | {
      type: 'answerability'
      query: string
      scheme?: string
      task_id?: string
      node_id?: string
      /** 检索后答案充分性验证（跨复杂度路径的质量闸门）：可答 / 升级检索 / 追问澄清 */
      verdict: {
        answerable: boolean
        missing_facts: string[]
        recommendation: string // answer | escalate | clarify
        escalate_to?: string | null
      }
      /** 是否为检索不足后升级检索再验证的最终结论 */
      escalated?: boolean
      /** 该轮检索答案置信度 [0,1] */
      confidence?: number
    }
  | {
      type: 'agent_step'
      query: string
      scheme?: string
      task_id?: string
      node_id?: string
      /** agentic：检索 Agent 单步工具执行（逐步流式，index 递增） */
      step: {
        index: number
        role: string
        action: string
        params: Record<string, unknown>
        note?: string | null
        hits_count?: number
        volumes?: { volume: string; count: number }[]
      }
    }
  | {
      type: 'grade'
      query: string
      scheme?: string
      task_id?: string
      node_id?: string
      /** agentic：CRAG 证据评审结果（保留相关证据数 / 候选总数 / 缺口） */
      kept: number
      total: number
      missing_facts: string[]
      thought?: string
    }
  | {
      type: 'correct'
      query: string
      scheme?: string
      task_id?: string
      node_id?: string
      /** agentic：CRAG 纠错决策（纠错轮次 + 下一波工具调用） */
      round: number
      thought?: string
      calls: { action: string; query: string; volume?: string; reason?: string }[]
    }
  | {
      type: 'verify'
      query: string
      scheme?: string
      task_id?: string
      node_id?: string
      /** agentic：Self-RAG 答案校验结论（可答 / 缺口） */
      answerable: boolean
      missing_facts: string[]
      thought?: string
    }
  | { type: 'memory_write'; content: string; kind?: string; importance?: number; scope?: string; source?: string }
  | {
      type: 'memory_read'
      query: string
      hits: HitItem[]
      /** 召回来源：tool（memory_recall 工具，模型被动触发）| proactive（L2 主动语义召回，系统每轮触发） */
      source?: 'tool' | 'proactive'
      /** proactive：selector 触发判断结果（need=false 表示本轮判断无需记忆，跳过召回） */
      need?: boolean
      reason?: string
    }
  | { type: 'memory_constant'; count: number }
  | { type: 'approval_request'; approval_id: string; tool_calls: ToolCallInfo[] }
  | { type: 'reflect'; stage?: string; critique?: string }
  | { type: 'revise'; delta: string }
  | { type: 'critique'; delta: string }
  | { type: 'agent_event'; worker: string; status: string; task?: string; result?: string }
  | {
      type: 'task_plan'
      task_id: string
      query: string
      /** 拆解器产出的子查询节点 DAG */
      nodes: TaskGraphNode[]
      thought?: string
      note?: string
    }
  | {
      type: 'task_retry'
      task_id: string
      query: string
      node_id: string
      query_prev: string
      rewrite_query: string
      retries: number
      gap_type?: string
      reason?: string
    }
  | {
      type: 'task_node'
      task_id: string
      query: string
      node_id: string
      node_query: string
      state: 'resolved' | 'gap'
      verdict?: { answerable?: boolean }
      missing_facts: string[]
      confidence: number
      cost?: Record<string, unknown>
      hits_count?: number
      retries?: number
      note?: string
    }
  | {
      type: 'task_done'
      task_id: string
      query: string
      result: {
        task_id: string
        query: string
        nodes: TaskGraphNode[]
        completion: string
        resolved: number
        gaps: number
        confidence: number
        cost?: Record<string, unknown>
        gap_list?: { node_id: string; query: string; gap_type: string; action: string; note: string }[]
      }
    }
  | { type: 'done'; summary: string; stats: Record<string, unknown> }
  | { type: 'error'; message: string; detail?: string }
  | { type: 'guard_refused'; reason: string; matched?: string }
