/** 能力卡片内容：通过后端 /api/content 接口获取（后端实时解析 backend/content/ 下的 md） */
import { ref } from 'vue'
import type { Difficulty, KnowledgeTag, LandingCapability, TechId } from '../data/capabilityData'
import type { ApprovalPolicy, ModeId, PromptStrategy, RagSchemeId } from '../types/agent'

/** 难度 → 难度展示文案 */
const DIFFICULTY_LABELS: Record<Difficulty, string> = {
  beg: '入门',
  int: '进阶',
  adv: '高阶',
}

const DEFAULT_ICON = 'sparkles'
const DEFAULT_ACCENT = '#7c5cff'

/** 合法取值白名单（非法值回退默认，不中断页面） */
const VALID_STRATEGIES: PromptStrategy[] = ['standard', 'few_shot', 'cot']
const VALID_POLICIES: ApprovalPolicy[] = ['always', 'never']
const VALID_RAG_SCHEMES: RagSchemeId[] = ['naive', 'advanced', 'modular', 'agentic']

/** tags.md 中标签注册表的原始结构 */
interface TagManifest {
  id: string
  title: string
  description: string
  cards?: string[]
  groups?: { title: string; cards?: string[] }[]
}

/** 后端解析出的卡片元数据 + 正文 */
interface CardPayload {
  id: string
  name: string
  shortDesc?: string
  icon?: string
  difficulty?: string
  completeLevel?: number
  tags?: string[]
  techFilters?: string[]
  accent?: string
  mode?: string
  enabledTools?: string[]
  faults?: Record<string, string>
  strategy?: string
  policy?: string
  ragScheme?: string
  prompts?: string[]
  prompt?: string
  experience?: boolean
  body: string
}

interface ContentPayload {
  tags: TagManifest[]
  cards: CardPayload[]
}

/** 把后端返回的卡片数据归一化为 LandingCapability；缺 id/name 视为非法卡片 */
function normalizeCard(card: CardPayload): LandingCapability | null {
  if (!card.id || !card.name) {
    console.warn('[content] 卡片缺少 id/name，已跳过：', card)
    return null
  }
  const difficulty = (['beg', 'int', 'adv'].includes(card.difficulty ?? '') ? card.difficulty : 'int') as Difficulty
  return {
    id: card.id,
    name: card.name,
    shortDesc: typeof card.shortDesc === 'string' ? card.shortDesc : '',
    icon: typeof card.icon === 'string' ? card.icon : DEFAULT_ICON,
    difficulty,
    difficultyLabel: DIFFICULTY_LABELS[difficulty],
    completeLevel: typeof card.completeLevel === 'number' ? card.completeLevel : null,
    tags: Array.isArray(card.tags) ? card.tags : [],
    techFilters: Array.isArray(card.techFilters) ? (card.techFilters as TechId[]) : [],
    accent: typeof card.accent === 'string' ? card.accent : DEFAULT_ACCENT,
    mode: typeof card.mode === 'string' ? (card.mode as ModeId) : null,
    enabledTools: Array.isArray(card.enabledTools) ? card.enabledTools : [],
    faults:
      card.faults && typeof card.faults === 'object'
        ? Object.fromEntries(Object.entries(card.faults).filter(([, v]) => typeof v === 'string'))
        : {},
    strategy: VALID_STRATEGIES.includes(card.strategy as PromptStrategy) ? (card.strategy as PromptStrategy) : null,
    policy: VALID_POLICIES.includes(card.policy as ApprovalPolicy) ? (card.policy as ApprovalPolicy) : null,
    ragScheme: VALID_RAG_SCHEMES.includes(card.ragScheme as RagSchemeId)
      ? (card.ragScheme as RagSchemeId)
      : null,
    prompts: Array.isArray(card.prompts)
      ? card.prompts.filter((p): p is string => typeof p === 'string').map((p) => p.trim()).filter(Boolean)
      : typeof card.prompt === 'string'
        ? [card.prompt]
        : [],
    experience: card.experience !== false,
    content: typeof card.body === 'string' ? card.body : '',
  }
}

/**
 * 调用后端 /api/content 拉取标签清单与全部卡片（后端实时读 md 解析），
 * 返回响应式数据：标签分组 / 全部卡片 / id→正文 / loading / error。
 */
export function useContentData() {
  const tags = ref<KnowledgeTag[]>([])
  const caps = ref<LandingCapability[]>([])
  const cardContent = ref<Record<string, string>>({})
  const loading = ref(true)
  const error = ref<string | null>(null)

  async function load() {
    loading.value = true
    error.value = null
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}api/content`)
      if (!res.ok) {
        throw new Error(`内容接口拉取失败：HTTP ${res.status}`)
      }
      const payload = (await res.json()) as ContentPayload
      const tagsList = Array.isArray(payload.tags) ? payload.tags : []

      const cardMap = new Map<string, LandingCapability>()
      for (const raw of Array.isArray(payload.cards) ? payload.cards : []) {
        const card = normalizeCard(raw)
        if (card) cardMap.set(card.id, card)
      }

      const knowledgeTags: KnowledgeTag[] = []
      const allCards: LandingCapability[] = []
      const contents: Record<string, string> = {}
      for (const tag of tagsList) {
        const flatIds: string[] = [...(tag.cards ?? [])]
        for (const group of tag.groups ?? []) {
          flatIds.push(...(group.cards ?? []))
        }
        const cards = flatIds
          .map((id) => cardMap.get(id))
          .filter((card): card is LandingCapability => Boolean(card))
        const groups = (tag.groups ?? []).map((g) => ({
          title: g.title,
          cards: (g.cards ?? [])
            .map((id) => cardMap.get(id))
            .filter((card): card is LandingCapability => Boolean(card)),
        }))
        knowledgeTags.push({ id: tag.id, title: tag.title, description: tag.description ?? '', cards, groups })
        allCards.push(...cards)
      }
      for (const card of allCards) {
        contents[card.id] = card.content
      }

      tags.value = knowledgeTags
      caps.value = allCards
      cardContent.value = contents
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  void load()

  return { tags, caps, cardContent, loading, error }
}
