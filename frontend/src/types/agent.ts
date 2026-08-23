/** Agent 平台前端类型定义：能力 / 模式 / SSE 事件协议，与后端严格对齐。 */

export type ModeId = 'react' | 'plan_execute' | 'reflection' | 'multi_agent'
export type PromptStrategy = 'standard' | 'few_shot' | 'cot'
export type ApprovalPolicy = 'always' | 'never'
export type ApprovalDecision = 'approve' | 'reject' | 'modify'
export type RagSchemeId = 'naive' | 'advanced'

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
  | { type: 'meta'; session_id: string; mode: string; capabilities: string[]; rag_scheme?: string }
  | { type: 'thinking'; delta: string }
  | { type: 'message'; delta: string }
  | { type: 'tool_start'; tool: string; args: Record<string, unknown> }
  | { type: 'tool_end'; tool: string; args?: Record<string, unknown>; result: string; success: boolean }
  | { type: 'tool_retry'; tool: string; attempt: number; max: number; delay: number; base_delay?: number; reason: string }
  | { type: 'plan'; steps: string[]; current_step: number; status: string }
  | { type: 'retrieve'; query: string; scheme?: string; hits: HitItem[]; rewrites?: string[]; reranked?: boolean }
  | { type: 'memory_write'; content: string }
  | { type: 'memory_read'; query: string; hits: HitItem[] }
  | { type: 'approval_request'; approval_id: string; tool_calls: ToolCallInfo[] }
  | { type: 'reflect'; stage?: string; critique?: string }
  | { type: 'revise'; delta: string }
  | { type: 'critique'; delta: string }
  | { type: 'agent_event'; worker: string; status: string; task?: string; result?: string }
  | { type: 'done'; summary: string; stats: Record<string, unknown> }
  | { type: 'error'; message: string; detail?: string }
