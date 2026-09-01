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
  | { type: 'retrieve'; query: string; scheme?: string; hits: HitItem[]; reranked?: boolean }
  | { type: 'rewrite'; query: string; scheme?: string; rewrites: string[]; reason?: string }
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
      type: 'answerability'
      query: string
      scheme?: string
      /** 检索后答案充分性验证（跨复杂度路径的质量闸门）：可答 / 升级检索 / 追问澄清 */
      verdict: {
        answerable: boolean
        missing_facts: string[]
        recommendation: string // answer | escalate | clarify
        escalate_to?: string | null
      }
      /** 是否为检索不足后升级检索再验证的最终结论 */
      escalated?: boolean
    }
  | {
      type: 'agent_step'
      query: string
      scheme?: string
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
      /** agentic：CRAG 纠错决策（纠错轮次 + 下一波工具调用） */
      round: number
      thought?: string
      calls: { action: string; query: string; volume?: string; reason?: string }[]
    }
  | {
      type: 'verify'
      query: string
      scheme?: string
      /** agentic：Self-RAG 答案校验结论（可答 / 缺口） */
      answerable: boolean
      missing_facts: string[]
      thought?: string
    }
  | { type: 'memory_write'; content: string }
  | { type: 'memory_read'; query: string; hits: HitItem[] }
  | { type: 'approval_request'; approval_id: string; tool_calls: ToolCallInfo[] }
  | { type: 'reflect'; stage?: string; critique?: string }
  | { type: 'revise'; delta: string }
  | { type: 'critique'; delta: string }
  | { type: 'agent_event'; worker: string; status: string; task?: string; result?: string }
  | { type: 'done'; summary: string; stats: Record<string, unknown> }
  | { type: 'error'; message: string; detail?: string }
