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
  /** 重点知识卡：置顶展示（frontmatter featured: true） */
  featured: boolean
  /** 新技术卡：紧随重点卡之后（frontmatter new: true） */
  isNew: boolean
  /** 所属大标签（分类）id 列表，用于卡片上渲染大标签徽标（tags.md 归属推导） */
  categoryIds: string[]
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

/** 大标签（分类）→ 徽标颜色，与 tags.md 的标签 id 对齐 */
export const CATEGORY_COLORS: Record<string, string> = {
  'agent-engineering': '#7c5cff',
  agent: '#38bdf8',
  rag: '#f59e0b',
  protocol: '#22d3a8',
  eval: '#f43f5e',
  ops: '#10b981',
}

export interface TechStackItem {
  id: TechId
  label: string
  color: string
}

/** 运行时旅程：一次对话从提问到观测的完整链路（替代原 ARCH_LAYERS 分层展示） */
export interface JourneyChannel {
  id: string
  name: string
  desc: string
  techs: string[]
  color: string
  icon: string
}

export interface JourneyStop {
  id: string
  name: string
  icon: string
  color: string
  /** 单通道阶段说明 */
  desc?: string
  techs?: string[]
  /** 双通道阶段：并行执行的两个通道（如 工具执行 / 检索增强） */
  channels?: JourneyChannel[]
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

/** 一次对话的完整旅程（7 站）：每条消息背后平台按序做了什么 */
export const JOURNEY_STAGES: JourneyStop[] = [
  {
    id: 'guard-in',
    name: '安全闸门',
    icon: 'shield',
    color: '#10b981',
    desc: '输入 Guardrail 过滤越狱与提示注入，敏感信息先行脱敏',
    techs: ['Input Guardrail', '注入防护', '脱敏'],
  },
  {
    id: 'assembly',
    name: '装配与路由',
    icon: 'compass',
    color: '#38bdf8',
    desc: '按前端开关装配：推理模式 · RAG 方案 · 能力池热插拔 · 记忆注入',
    techs: ['能力注册表', 'tools_builder', '记忆注入'],
  },
  {
    id: 'orchestrate',
    name: '编排引擎',
    icon: 'cpu',
    color: '#7c5cff',
    desc: 'LangGraph 状态机驱动四种模式：react / plan_execute / reflection / multi_agent',
    techs: ['LangGraph', 'LangChain', '中间件', 'Harness 护栏'],
  },
  {
    id: 'execute',
    name: '并行执行',
    icon: 'zap',
    color: '#22d3a8',
    channels: [
      {
        id: 'tools',
        name: '工具执行',
        desc: '内置 + MCP 工具统一执行，两层重试（透明退避 / 思考后重试），HITL 审批',
        techs: ['MCP', '内置工具', '两层重试', 'HITL'],
        color: '#22d3a8',
        icon: 'terminal',
      },
      {
        id: 'rag',
        name: '检索增强',
        desc: 'Agentic RAG 五角色循环：路由 → 规划 → 评审 → 纠错 → 校验（CRAG + Self-RAG）',
        techs: ['五角色', 'CRAG', 'Self-RAG', '规则回退'],
        color: '#f59e0b',
        icon: 'database',
      },
    ],
  },
  {
    id: 'generate',
    name: '模型生成',
    icon: 'brain',
    color: '#d946ef',
    desc: 'Qwen 流式输出：思考过程与回答分离，token 记账与成本估算',
    techs: ['DashScope Qwen', '流式 SSE', 'token 记账'],
  },
  {
    id: 'guard-out',
    name: '输出治理',
    icon: 'check',
    color: '#f43f5e',
    desc: '输出 Guardrail、敏感数据脱敏、超大输出落盘、引用核验',
    techs: ['Output Guardrail', '脱敏', '大输出落盘'],
  },
  {
    id: 'observe',
    name: '全程观测',
    icon: 'chart-bar',
    color: '#818cf8',
    desc: '运行记录回放（SSE 事件流 + LLM 明细 + 成本），在线评测采样与反馈回流',
    techs: ['Telemetry', '运行记录', '在线评测'],
  },
]
