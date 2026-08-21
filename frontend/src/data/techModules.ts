/** 四种推理模式的元信息：名称 / 简介 / 标签 / 执行步骤 / 流程图节点与边 / 默认示例。 */
import type { ModeId } from '../types/agent'

export interface FlowNode {
  id: string
  label: string
  kind: 'start' | 'llm' | 'tool' | 'plan' | 'decision' | 'worker' | 'reflect' | 'revise' | 'end'
  note?: string
}

export interface FlowEdge {
  from: string
  to: string
  label?: string
}

export interface ModeMeta {
  id: ModeId
  name: string
  tagline: string
  description: string
  tags: string[]
  steps: string[]
  nodes: FlowNode[]
  edges: FlowEdge[]
  defaultPrompt: string
}

export const MODES: ModeMeta[] = [
  {
    id: 'react',
    name: 'ReAct',
    tagline: '思考-行动-观察循环',
    description:
      'Agent 交替执行「思考（决定下一步）」与「行动（调用工具）」，把工具的观察结果带回上下文，直到无需调用工具时给出最终回答。',
    tags: ['ReAct', '循环', '工具调用', '经典范式'],
    steps: [
      '接收用户任务',
      'Agent 思考：决定下一步行动或直接回答',
      '若有工具调用则执行工具并观察结果',
      '重复「思考-行动-观察」直至无需调用工具',
      '输出最终回答',
    ],
    nodes: [
      { id: 'start', label: '接收任务', kind: 'start' },
      { id: 'think', label: 'Agent 思考', kind: 'llm', note: '生成回复或决定调用工具' },
      { id: 'action', label: '执行工具', kind: 'tool', note: '计算 / 搜索 / 检索…' },
      { id: 'final', label: '无工具调用 → 生成回答', kind: 'llm' },
      { id: 'end', label: '结束', kind: 'end' },
    ],
    edges: [
      { from: 'start', to: 'think' },
      { from: 'think', to: 'action', label: '有工具调用' },
      { from: 'action', to: 'think', label: '观察结果后继续' },
      { from: 'think', to: 'final', label: '无工具调用' },
      { from: 'final', to: 'end' },
    ],
    defaultPrompt: '帮我计算 (137×0.85−20)÷3 等于多少，并解释每一步。',
  },
  {
    id: 'plan_execute',
    name: 'Plan-and-Execute',
    tagline: '计划分解，逐步执行',
    description:
      '先把任务拆解为 2-5 个有序子步骤形成计划，再逐条执行（可调用工具），执行完检查是否还有剩余步骤，最后收尾输出。',
    tags: ['规划', '分解', '逐步执行', 'Replan'],
    steps: [
      '任务解析并分解为 2-5 个有序子步骤',
      '按计划执行当前步骤（可调用工具）',
      '工具结果回填，继续下一步',
      '检查剩余步骤：继续执行或收尾',
      '输出最终回答',
    ],
    nodes: [
      { id: 'start', label: '接收任务', kind: 'start' },
      { id: 'plan', label: '规划器：任务分解', kind: 'plan', note: '拆解为有序子步骤' },
      { id: 'exec', label: '执行器：执行当前步骤', kind: 'llm' },
      { id: 'tools', label: '执行工具', kind: 'tool' },
      { id: 'replan', label: '检查剩余步骤', kind: 'decision' },
      { id: 'final', label: '计划完成 → 输出', kind: 'llm' },
      { id: 'end', label: '结束', kind: 'end' },
    ],
    edges: [
      { from: 'start', to: 'plan' },
      { from: 'plan', to: 'exec' },
      { from: 'exec', to: 'tools', label: '有工具调用' },
      { from: 'tools', to: 'exec', label: '结果回填' },
      { from: 'exec', to: 'replan', label: '无工具调用' },
      { from: 'replan', to: 'exec', label: '仍有步骤' },
      { from: 'replan', to: 'final', label: '全部完成' },
      { from: 'final', to: 'end' },
    ],
    defaultPrompt: '规划并分步执行：计算 (137×0.85−20)÷3 等于多少，并给出每步结果。',
  },
  {
    id: 'reflection',
    name: 'Reflection',
    tagline: '生成-反思-修订',
    description:
      '先生成首版草稿，再由评审员对草稿输出批评意见；若批评非空则根据意见修订，重复「反思-修订」直到批评为空或达到最大轮次。',
    tags: ['自我反思', '迭代修订', '质量提升'],
    steps: [
      '生成首版草稿',
      '自我评审：输出批评意见',
      '若有批评则根据意见修订答案',
      '重复「评审-修订」直到通过',
      '输出最终回答',
    ],
    nodes: [
      { id: 'start', label: '接收任务', kind: 'start' },
      { id: 'gen', label: '生成草稿', kind: 'llm' },
      { id: 'reflect', label: '自我评审', kind: 'reflect', note: '输出批评意见' },
      { id: 'revise', label: '按意见修订', kind: 'revise' },
      { id: 'end', label: '批评为空 → 结束', kind: 'end' },
    ],
    edges: [
      { from: 'start', to: 'gen' },
      { from: 'gen', to: 'reflect' },
      { from: 'reflect', to: 'revise', label: '有批评' },
      { from: 'revise', to: 'reflect', label: '再评审' },
      { from: 'reflect', to: 'end', label: '无批评' },
    ],
    defaultPrompt: '写一段 3 句话介绍「RAG 如何工作」，并反思修订直到满意。',
  },
  {
    id: 'multi_agent',
    name: 'Multi-Agent',
    tagline: '编排者分派多个 Worker',
    description:
      'Orchestrator 解析任务后同时分派给计算 Worker 与分析 Worker 并行处理，各自产出子结论，再由汇总者整合成最终答案。',
    tags: ['多智能体', '并行', '分派', '汇总'],
    steps: [
      'Orchestrator 解析任务并分派',
      '计算 Worker：处理数值 / 计算类子任务',
      '分析 Worker：处理逻辑分析类子任务',
      '各 Worker 并行执行后汇总',
      '输出最终回答',
    ],
    nodes: [
      { id: 'start', label: '接收任务', kind: 'start' },
      { id: 'orch', label: 'Orchestrator 分派', kind: 'llm', note: '解析任务并分派' },
      { id: 'compute', label: '计算 Worker', kind: 'worker', note: '数值 / 计算子任务' },
      { id: 'analyze', label: '分析 Worker', kind: 'worker', note: '逻辑 / 分析子任务' },
      { id: 'agg', label: '汇总者整合', kind: 'llm' },
      { id: 'end', label: '结束', kind: 'end' },
    ],
    edges: [
      { from: 'start', to: 'orch' },
      { from: 'orch', to: 'compute' },
      { from: 'orch', to: 'analyze' },
      { from: 'compute', to: 'agg' },
      { from: 'analyze', to: 'agg' },
      { from: 'agg', to: 'end' },
    ],
    defaultPrompt: '结合计算与分析：计算 12×8 与 2026 的差值，并简要说明思路。',
  },
]

export function modeMeta(id: string): ModeMeta | undefined {
  return MODES.find((m) => m.id === id)
}
