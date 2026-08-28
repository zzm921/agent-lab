/** 落地页静态常量与类型：卡片数据由后端 /api/content 接口运行时拉取（见 useContentData） */
import type { ApprovalPolicy, ModeId, PromptStrategy, RagSchemeId } from '../types/agent'

export type Difficulty = 'beg' | 'int' | 'adv'
export type TechId = 'all' | 'LangGraph' | 'MCP' | 'FastAPI' | 'Qdrant' | 'Vue3'

/** 能力卡片：全部字段（含详情正文）由 content/<id>.md 的 frontmatter + 正文解析而来 */
export interface LandingCapability {
  id: string
  name: string
  shortDesc: string
  icon: string
  difficulty: Difficulty
  difficultyLabel: string
  /** 真实能力卡=实现完成度；纯知识卡为 null（模板隐藏徽标） */
  completeLevel: number | null
  tags: string[]
  techFilters: TechId[]
  accent: string
  /** 点击「立即体验」时跳转的实验室模式；null 表示该能力在实验室内是配置项而非独立模式 */
  mode: ModeId | null
  /** 进入实验室后建议启用的能力 ID 列表（能力开关） */
  enabledTools: string[]
  /** 工具故障注入配置：tool_id → 故障类型（off 表示正常；其余见后端 /api/faults/types） */
  faults: Record<string, string>
  /** 进入实验室的 prompt 策略；null 用实验室默认 */
  strategy: PromptStrategy | null
  /** 进入实验室的审批策略；null 用实验室默认 */
  policy: ApprovalPolicy | null
  /** 进入实验室的 RAG 方案；null 用实验室默认（naive） */
  ragScheme: RagSchemeId | null
  /** 进入实验室的预设任务列表：跳转后自动填入第一条，其余显示在输入框下方快捷区 */
  prompts: string[]
  /** 纯知识卡置 false：隐藏「立即体验」按钮与完成度徽标 */
  experience: boolean
  /** 详情抽屉正文 Markdown */
  content: string
}

export interface KnowledgeGroup {
  title: string
  cards: LandingCapability[]
}

export interface KnowledgeTag {
  id: string
  title: string
  description: string
  /** 标签内全部卡片（含 groups 内的卡片）的扁平列表，用于筛选 */
  cards: LandingCapability[]
  /** 可选二级分组（如工程演进标签内的 Prompt 层 / Context 层 / Harness 层），仅影响展示分段 */
  groups?: KnowledgeGroup[]
}

export interface TechStackItem {
  id: TechId
  label: string
  color: string
}

export interface ArchLayer {
  name: string
  sub: string
  techs: string[]
  color: string
  capability: string
}

/** 落地页 → 实验室传递 prompt 列表的 sessionStorage 键（避免长文本进 URL） */
export const LAB_PRESET_STORAGE_KEY = 'labPresetPrompts'

/** Agent 模式 → 智能体名称（卡片上"对应智能体"徽标用） */
export const MODE_AGENT_LABELS: Record<ModeId, string> = {
  react: 'ReAct 智能体',
  plan_execute: '计划执行智能体',
  reflection: '反思修订智能体',
  multi_agent: '多智能体',
}

export const TECH_STACK: TechStackItem[] = [
  { id: 'all', label: '全部能力', color: '#7c5cff' },
  { id: 'LangGraph', label: 'LangGraph', color: '#38bdf8' },
  { id: 'MCP', label: 'MCP', color: '#22d3a8' },
  { id: 'FastAPI', label: 'FastAPI', color: '#f59e0b' },
  { id: 'Qdrant', label: 'Qdrant', color: '#ef4444' },
  { id: 'Vue3', label: 'Vue 3', color: '#22d3a8' },
]

export const ARCH_LAYERS: ArchLayer[] = [
  {
    name: '前端层',
    sub: '交互与可视化',
    techs: ['Vue 3', 'TypeScript', 'Pinia', 'Monaco Editor'],
    color: '#38bdf8',
    capability: 'react',
  },
  {
    name: '编排层',
    sub: 'Agent 状态机与调度',
    techs: ['LangGraph', 'FastAPI', 'Pydantic'],
    color: '#7c5cff',
    capability: 'multi-agent',
  },
  {
    name: '工具层',
    sub: '能力扩展与协议',
    techs: ['MCP', 'Tool Registry', 'Fault Injector'],
    color: '#22d3a8',
    capability: 'mcp',
  },
  {
    name: '存储层',
    sub: '持久化与检索',
    techs: ['Qdrant', 'PostgreSQL', 'Redis'],
    color: '#f59e0b',
    capability: 'rag',
  },
]
