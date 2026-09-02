<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import hljs from 'highlight.js'
import { marked } from 'marked'
import CapabilityHeroVisual from '../components/CapabilityHeroVisual.vue'
import ArchitectureOverview from '../components/ArchitectureOverview.vue'
import { useContentData } from '../composables/useContentData'
import {
  ARCH_LAYERS,
  LAB_PRESET_STORAGE_KEY,
  MODE_AGENT_LABELS,
  type LandingCapability,
} from '../data/capabilityData'

const router = useRouter()
const { tags, caps, cardContent, loading, error } = useContentData()
const activeTag = ref<string>('all')
const detailCap = ref<LandingCapability | null>(null)
const detailOpen = ref(false)
const detailFull = ref(false)

/** 当前标签筛选下平铺可见的卡片（all = 全部） */
const visibleCards = computed(() => {
  if (activeTag.value === 'all') return caps.value
  const tag = tags.value.find((t) => t.id === activeTag.value)
  return tag ? tag.cards : []
})

/** 卡片区段：有二级分组的标签按组渲染，无分组整体平铺；全部能力时平铺全部卡片 */
const cardSections = computed(() => {
  if (activeTag.value === 'all') return [{ title: null, cards: caps.value }]
  const tag = tags.value.find((t) => t.id === activeTag.value)
  if (!tag) return []
  if (tag.groups && tag.groups.length) return tag.groups.map((g) => ({ title: g.title, cards: g.cards }))
  return [{ title: null, cards: tag.cards }]
})

/** 卡片对应的智能体名称（仅真实 Agent 卡有 mode；知识/工具卡返回 null 不显示） */
function agentOf(cap: LandingCapability) {
  return cap.mode ? MODE_AGENT_LABELS[cap.mode] : null
}

// Markdown 渲染配置：代码块交给 highlight.js 高亮（未知语言回退 plaintext）
marked.use({
  renderer: {
    code({ text, lang }) {
      const language = lang && hljs.getLanguage(lang) ? lang : 'plaintext'
      const html = hljs.highlight(text, { language }).value
      return `<pre class="markdown-code"><code class="hljs language-${language}">${html}</code></pre>`
    },
  },
})

/** 详情抽屉正文：由 content/<id>.md 运行时拉取并渲染 */
const detailHtml = computed(() => {
  const cap = detailCap.value
  return cap ? (marked.parse(cardContent.value[cap.id] ?? '') as string) : ''
})

const difficultyClass = (diff: string) => {
  const map: Record<string, string> = {
    adv: 'bg-rose-500/15 text-rose-300',
    int: 'bg-sky-500/15 text-sky-300',
    beg: 'bg-emerald-500/15 text-emerald-300',
  }
  return map[diff] ?? 'bg-slate-500/15 text-slate-300'
}

function openDetail(cap: LandingCapability) {
  detailCap.value = cap
  detailFull.value = false
  detailOpen.value = true
}

function closeDetail() {
  detailOpen.value = false
}

/** 从详情抽屉返回首页（关闭抽屉并回到落地页） */
function backHome() {
  detailOpen.value = false
  router.push('/')
}

watch(detailOpen, (open) => {
  if (typeof document !== 'undefined') {
    document.body.style.overflow = open ? 'hidden' : ''
  }
})

function experience(cap: LandingCapability) {
  const query: Record<string, string> = {}
  if (cap.mode) {
    query.mode = cap.mode
  }
  if (cap.enabledTools?.length) {
    query.tools = cap.enabledTools.join(',')
  }
  if (cap.strategy) {
    query.strategy = cap.strategy
  }
  if (cap.policy) {
    query.policy = cap.policy
  }
  if (cap.ragScheme) {
    query.rag_scheme = cap.ragScheme
  }
  if (cap.prompts.length) {
    // 长文本不放进 URL：写入 sessionStorage，URL 只带短 nonce
    const nonce = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
    try {
      sessionStorage.setItem(LAB_PRESET_STORAGE_KEY, JSON.stringify({ nonce, prompts: cap.prompts }))
      query.jump = nonce
    } catch {
      // sessionStorage 不可用（如隐私模式）：退回 URL 传递
      query.prompts = cap.prompts.join('\n')
    }
  }
  const faults = Object.entries(cap.faults ?? {})
    .filter(([, type]) => type && type !== 'off')
    .map(([tool, type]) => `${tool}:${type}`)
    .join(',')
  if (faults) {
    query.faults = faults
  }
  void router.push({ path: '/lab', query })
}

const ICONS: Record<string, string> = {
  brain: 'M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z',
  refresh: 'M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99',
  list: 'M8.25 6.75h12M8.25 12h12m-12 5.25h12M3.75 6.75h.007v.008H3.75V6.75Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0ZM3.75 12h.007v.008H3.75V12Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm-.375 5.25h.007v.008H3.75v-.008Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z',
  network: 'M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0 0 20.25 18V6A2.25 2.25 0 0 0 18 3.75H6A2.25 2.25 0 0 0 3.75 6v12A2.25 2.25 0 0 0 6 20.25Z',
  plug: 'M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z',
  shield: 'M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z',
  check: 'M9 12.75 11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 0 1-1.043 3.296 3.745 3.745 0 0 1-3.296 1.043A3.745 3.745 0 0 1 12 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 0 1-3.296-1.043 3.745 3.745 0 0 1-1.043-3.296A3.745 3.745 0 0 1 3 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 0 1 1.043-3.296 3.746 3.746 0 0 1 3.296-1.043A3.746 3.746 0 0 1 12 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 0 1 3.296 1.043 3.746 3.746 0 0 1 1.043 3.296A3.745 3.745 0 0 1 21 12Z',
  sparkles: 'M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456Z',
  database: 'M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 3.75v3.75m-16.5-3.75v3.75',
  terminal: 'm6.75 7.5 3 2.25-3 2.25m4.5 0h3m-9 8.25h13.5A2.25 2.25 0 0 0 21 18V6a2.25 2.25 0 0 0-2.25-2.25H5.25A2.25 2.25 0 0 0 3 6v12a2.25 2.25 0 0 0 2.25 2.25Z',
  'code-bracket': 'M17.25 6.75 22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3-4.5 16.5',
  cube: 'M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9',
  zap: 'M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z',
  globe: 'M12 21a9.004 9.004 0 0 0 8.716-6.747M12 21a9.004 9.004 0 0 1-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 0 1 7.843 4.582M12 3a8.997 8.997 0 0 0-7.843 4.582m15.686 0A11.953 11.953 0 0 1 12 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0 1 21 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0 1 12 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 0 1 3 12c0-1.605.42-3.113 1.157-4.418',
  compass: 'M16.712 4.33a9.027 9.027 0 0 1 1.652 1.306c.51.51.944 1.064 1.306 1.652M16.712 4.33l-3.448 4.138m3.448-4.138a9.014 9.014 0 0 0-9.424 0M19.67 7.288l-4.138 3.448m4.138-3.448a9.014 9.014 0 0 1 0 9.424m-4.138-5.976a3.736 3.736 0 0 0-.88-1.388 3.737 3.737 0 0 0-1.388-.88m2.268 2.268a3.765 3.765 0 0 1-2.268 2.268m0-4.536a3.765 3.765 0 0 1-2.268 2.268m0 4.536a3.737 3.737 0 0 1-1.388-.88 3.737 3.737 0 0 1-.88-1.388m0 2.268a9.015 9.015 0 0 1-4.418-1.157M4.33 16.712a9.027 9.027 0 0 1-1.306-1.652m1.306 1.652 4.138-3.448m-4.138 3.448a9.014 9.014 0 0 0 9.424 0M4.67 7.288l4.138 3.448m-4.138-3.448a9.014 9.014 0 0 0 0 9.424m0-9.424a9.027 9.027 0 0 1 1.306-1.652m0 0a9.014 9.014 0 0 1 9.424 0',
  monitor: 'M9 17.25v1.007a3 3 0 0 1-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0 1 15 18.257V17.25m6-12V15a2.25 2.25 0 0 1-2.25 2.25H5.25A2.25 2.25 0 0 1 3 15V5.25m18 0A2.25 2.25 0 0 0 18.75 3H5.25A2.25 2.25 0 0 0 3 5.25m18 0V12a2.25 2.25 0 0 1-2.25 2.25H5.25A2.25 2.25 0 0 1 3 12V5.25',
  history: 'M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
  'chart-bar': 'M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z',
  lock: 'M16.5 10.5V6.75a4.5 4.5 0 1 0-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H6.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z',
  cpu: 'M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 0 0 2.25-2.25V6.75a2.25 2.25 0 0 0-2.25-2.25H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Zm.75-12h9v9h-9v-9Z',
  'arrows-right-left': 'M7.5 21 3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5',
  info: 'M11.25 11.25l.041-.02a.75.75 0 0 1 1.063.852c-.04.154-.111.312-.198.5l-.025.06a2.25 2.25 0 0 0-.268.75c-.01.123-.015.246-.015.37v.75c0 .138.112.25.25.25H12M9 12.75h.008v.008H9V12.75Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z',
  play: 'M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z',
  close: 'M6 18L18 6M6 6l12 12',
  maximize:
    'M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9.75M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9.75M3.75 20.25h4.5m-4.5 0v-4.5m0 4.5L9 14.25M20.25 20.25h-4.5m4.5 0v-4.5m0 4.5L15 14.25',
  minimize:
    'M9 9V4.5M9 9H4.5M9 9 3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5 5.25 5.25',
  arrowRight: 'M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3',
  arrowLeft: 'M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18',
}

function iconPath(name: string) {
  return ICONS[name] || ICONS.sparkles
}
</script>

<template>
  <div class="relative min-h-screen overflow-x-hidden bg-slate-950 pb-20 text-slate-200">
    <!-- 背景网格 -->
    <div
      class="pointer-events-none absolute inset-0 opacity-20"
      style="background-image: radial-gradient(rgba(255,255,255,0.08) 1px, transparent 1px); background-size: 24px 24px;"
    ></div>

    <div class="relative px-4 sm:px-6 lg:px-8">
      <!-- Hero -->
      <section class="grid items-center gap-10 py-16 md:grid-cols-2 md:py-24">
        <div>
          <h1 class="text-3xl font-bold leading-tight text-white sm:text-4xl md:text-5xl">
            Agent 技术实验室
            <span class="block bg-gradient-to-r from-indigo-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
              以场景为尺，向前沿而行
            </span>
          </h1>
          <p class="mt-5 text-sm leading-relaxed text-slate-400 sm:text-base">
            这里是 AI Agent 技术实验室。从提示词到协议，覆盖六大技术域——推理模式、工具调用、多智能体编排、故障容错……每一项能力都有原理讲解、核心代码与可运行的在线演示。技术没有绝对的高低，适合场景才是最好；新技术代表前沿方向，项目未必用得上，但不能不了解。点开任意卡片，直接进入实验室体验。
          </p>

          <div class="mt-8 flex flex-wrap items-center gap-6 text-sm">
            <div class="flex items-center gap-2">
              <span class="text-xl font-bold text-white">{{ caps.length }}</span>
              <span class="text-slate-400">技术能力点</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="text-xl font-bold text-white">{{ ARCH_LAYERS.length }}</span>
              <span class="text-slate-400">架构分层</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="text-xl font-bold text-white">100%</span>
              <span class="text-slate-400">可交互体验</span>
            </div>
          </div>
        </div>

        <div class="hidden md:block">
          <CapabilityHeroVisual />
        </div>
      </section>

      <!-- 能力地图标题（置于标签筛选之上） -->
      <section class="pt-16">
        <div class="mb-8">
          <h2 class="text-2xl font-semibold text-white">能力地图</h2>
          <p class="mt-1 text-sm text-slate-400">
            共 {{ visibleCards.length }} 项能力 · {{ tags.length }} 个标签 · 点击查看详情，或直接进入实验室体验
          </p>
        </div>
      </section>

      <!-- 标签筛选 -->
      <div class="flex flex-wrap items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.03] p-3">
        <span class="px-2 text-xs font-medium text-slate-500">标签</span>
        <button
          type="button"
          class="rounded-full px-3 py-1.5 text-xs transition"
          :class="
            activeTag === 'all'
              ? 'bg-white/10 text-white'
              : 'text-slate-400 hover:bg-white/5 hover:text-white'
          "
          @click="activeTag = 'all'"
        >
          全部能力
        </button>
        <button
          v-for="tag in tags"
          :key="tag.id"
          type="button"
          class="rounded-full px-3 py-1.5 text-xs transition"
          :class="
            activeTag === tag.id
              ? 'bg-white/10 text-white'
              : 'text-slate-400 hover:bg-white/5 hover:text-white'
          "
          @click="activeTag = tag.id"
        >
          {{ tag.title }}
        </button>
      </div>

      <!-- 能力卡片（标签筛选后平铺） -->
      <section class="pt-10">
        <div
          v-if="loading"
          class="rounded-2xl border border-white/10 bg-white/[0.03] p-10 text-center text-sm text-slate-400"
        >
          正在加载能力内容…
        </div>

        <div
          v-else-if="error"
          class="rounded-2xl border border-rose-500/20 bg-rose-500/5 p-10 text-center text-sm text-rose-300"
        >
          内容加载失败：{{ error }}。请确认 public/content/ 下的 md 文件存在且可访问。
        </div>

        <template v-else>
          <template v-for="section in cardSections" :key="section.title ?? '__all__'">
            <h4 v-if="section.title" class="mb-3 mt-8 text-sm font-medium text-slate-400">
              {{ section.title }}
            </h4>
            <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div
            v-for="cap in section.cards"
            :key="cap.id"
            class="group flex cursor-pointer flex-col rounded-2xl border border-white/10 bg-white/[0.03] p-5 transition hover:border-white/20 hover:bg-white/[0.05]"
            @click="openDetail(cap)"
          >
            <div class="flex items-start justify-between gap-3">
              <div
                class="grid h-10 w-10 place-items-center rounded-xl"
                :style="{ background: `${cap.accent}20`, color: cap.accent }"
              >
                <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" :d="iconPath(cap.icon)" />
                </svg>
              </div>
              <div class="flex flex-wrap justify-end gap-1.5">
                <span
                  v-if="agentOf(cap)"
                  class="rounded-md bg-indigo-500/15 px-2 py-0.5 text-[10px] font-medium text-indigo-300"
                >
                  {{ agentOf(cap) }}
                </span>
                <span class="rounded-md px-2 py-0.5 text-[10px] font-medium" :class="difficultyClass(cap.difficulty)">
                  {{ cap.difficultyLabel }}
                </span>
                <span
                  v-if="cap.completeLevel != null"
                  class="rounded-md bg-slate-800 px-2 py-0.5 text-[10px] font-medium text-slate-300"
                >
                  {{ cap.completeLevel }}%
                </span>
              </div>
            </div>

            <h4 class="mt-4 text-base font-semibold text-white">{{ cap.name }}</h4>
            <p class="mt-2 text-xs leading-relaxed text-slate-400">{{ cap.shortDesc }}</p>
            <p class="mt-2 text-[11px] text-slate-500">{{ cap.tags.join(' ') }}</p>

            <div class="mt-5 flex items-center justify-between gap-2 pt-4 border-t border-white/10">
              <button
                type="button"
                class="flex items-center gap-1 text-xs text-slate-400 transition hover:text-white"
                @click.stop="openDetail(cap)"
              >
                <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" :d="iconPath('info')" />
                </svg>
                查看原理
              </button>
              <button
                v-if="cap.experience !== false"
                type="button"
                class="btn-accent-white flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs font-medium text-white transition"
                :style="{ background: cap.accent }"
                @click.stop="experience(cap)"
              >
                <svg class="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
                  <path :d="iconPath('play')" />
                </svg>
                立即体验
              </button>
            </div>
          </div>
            </div>
          </template>
        </template>
      </section>

      <!-- 架构总览 -->
      <!-- <ArchitectureOverview @experience="(id) => experience(caps.find((c) => c.id === id) ?? caps[0])" /> -->
    </div>

    <!-- 详情抽屉 -->
    <Transition name="fade">
      <div
        v-if="detailOpen"
        class="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
        @click="closeDetail"
      ></div>
    </Transition>

    <Transition name="slide">
      <aside
        v-if="detailOpen && detailCap"
        class="fixed z-50 overflow-y-auto bg-slate-950 shadow-2xl"
        :class="
          detailFull
            ? 'inset-0 w-full p-6 sm:p-10'
            : 'inset-y-0 right-0 w-full border-l border-white/10 p-6 sm:w-[28rem]'
        "
      >
        <div class="absolute right-4 top-4 flex items-center gap-1">
          <button
            type="button"
            class="rounded-lg p-2 text-slate-400 hover:bg-slate-800 hover:text-white"
            :title="detailFull ? '收起为侧栏' : '全屏查看'"
            @click="detailFull = !detailFull"
          >
            <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" :d="iconPath(detailFull ? 'minimize' : 'maximize')" />
            </svg>
          </button>
          <button
            type="button"
            class="rounded-lg p-2 text-slate-400 hover:bg-slate-800 hover:text-white"
            @click="closeDetail"
          >
            <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" :d="iconPath('close')" />
            </svg>
          </button>
        </div>

        <div :class="detailFull ? 'mx-auto max-w-4xl' : ''">
        <button
          type="button"
          class="mb-5 flex items-center gap-1.5 text-sm text-slate-400 transition hover:text-white"
          @click="backHome"
        >
          <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" :d="iconPath('arrowLeft')" />
          </svg>
          返回首页
        </button>
        <div class="mb-6">
          <div
            class="grid h-14 w-14 place-items-center rounded-2xl"
            :style="{ background: `${detailCap.accent}20`, color: detailCap.accent }"
          >
            <svg class="h-7 w-7" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" :d="iconPath(detailCap.icon)" />
            </svg>
          </div>
          <h2 class="mt-4 text-2xl font-semibold text-white">{{ detailCap.name }}</h2>
          <div class="mt-3 flex flex-wrap gap-2">
            <span
              v-for="tag in detailCap.tags"
              :key="tag"
              class="rounded border border-white/10 bg-white/5 px-2 py-0.5 text-[11px] text-slate-400"
            >
              {{ tag }}
            </span>
          </div>
        </div>

        <div v-if="detailHtml" class="markdown-body" v-html="detailHtml"></div>

        <div v-if="detailCap.experience !== false" class="mt-8">
          <button
            type="button"
            class="btn-accent-white flex w-full items-center justify-center gap-2 rounded-xl py-2.5 text-sm font-medium text-white transition"
            :style="{ background: detailCap.accent }"
            @click="experience(detailCap); closeDetail()"
          >
            <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
              <path :d="iconPath('play')" />
            </svg>
            去实验室体验这个能力
          </button>
        </div>
        </div>
      </aside>
    </Transition>
  </div>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.25s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.slide-enter-active,
.slide-leave-active {
  transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.slide-enter-from,
.slide-leave-to {
  transform: translateX(100%);
}
</style>
