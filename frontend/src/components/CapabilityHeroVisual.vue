<script setup lang="ts">
/** 轨道星云图：Agent 核心 + 七大能力节点环绕协同，纯视觉无框架 */

/** 星云图几何参数：中心 (160,160)，卫星轨道半径 112 */
const CX = 160
const CY = 160
const R = 112

/** 七大技术域（AI Agent 技术概况通用划分） */
const nodes = [
  { name: '推理规划', icon: 'brain', color: '#7c5cff' },
  { name: '工具调用', icon: 'terminal', color: '#22d3a8' },
  { name: '多智能体', icon: 'network', color: '#38bdf8' },
  { name: '检索增强', icon: 'database', color: '#f59e0b' },
  { name: '记忆系统', icon: 'history', color: '#a855f7' },
  { name: '安全护栏', icon: 'shield', color: '#10b981' },
  { name: '评测观测', icon: 'chart-bar', color: '#f43f5e' },
]

/** 节点按角度均布在轨道上 */
const satellites = nodes.map((node, i) => {
  const angle = (i / nodes.length) * 2 * Math.PI - Math.PI / 2
  return {
    ...node,
    x: +(CX + R * Math.cos(angle)).toFixed(1),
    y: +(CY + R * Math.sin(angle)).toFixed(1),
  }
})

/** 中心到每个节点的连接线（能量流动动画） */
const links = satellites.map((s) => ({
  x1: CX,
  y1: CY,
  x2: s.x,
  y2: s.y,
}))

/** 轨道圆路径（SMIL animateMotion 用）：半径为 R 的整圆 */
const orbitPath = `M ${CX} ${CY - R} a ${R} ${R} 0 1 1 0 ${2 * R} a ${R} ${R} 0 1 1 0 ${-2 * R}`
/** 内环圆路径：半径为 R/2 的整圆，反向运行 */
const innerPath = `M ${CX} ${CY - R / 2} a ${R / 2} ${R / 2} 0 1 1 0 ${R} a ${R / 2} ${R / 2} 0 1 1 0 ${-R}`

const ICONS: Record<string, string> = {
  brain:
    'M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z',
  terminal:
    'm6.75 7.5 3 2.25-3 2.25m4.5 0h3m-9 8.25h13.5A2.25 2.25 0 0 0 21 18V6a2.25 2.25 0 0 0-2.25-2.25H5.25A2.25 2.25 0 0 0 3 6v12a2.25 2.25 0 0 0 2.25 2.25Z',
  network:
    'M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0 0 20.25 18V6A2.25 2.25 0 0 0 18 3.75H6A2.25 2.25 0 0 0 3.75 6v12A2.25 2.25 0 0 0 6 20.25Z',
  database:
    'M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 3.75v3.75m-16.5-3.75v3.75',
  history: 'M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
  shield:
    'M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z',
  'chart-bar':
    'M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z',
  sparkles:
    'M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z',
}

function iconPath(name: string) {
  return ICONS[name] || ICONS.sparkles
}
</script>

<template>
  <!-- 轨道星云图：Agent 核心 + 七大能力节点，无背景框，放大展示 -->
  <div class="hero-orbit animate-slide-in relative">
    <svg viewBox="0 0 320 320" class="mx-auto block w-full max-w-[460px]">
      <defs>
        <!-- 星云背景辉光 -->
        <radialGradient id="nebula" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#7c5cff" stop-opacity="0.14" />
          <stop offset="55%" stop-color="#38bdf8" stop-opacity="0.05" />
          <stop offset="100%" stop-color="transparent" stop-opacity="0" />
        </radialGradient>
        <!-- 核心辉光 -->
        <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#a78bfa" stop-opacity="0.55" />
          <stop offset="100%" stop-color="#a78bfa" stop-opacity="0" />
        </radialGradient>
        <!-- 核心实心 -->
        <linearGradient id="coreSolid" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#8b5cf6" />
          <stop offset="100%" stop-color="#6366f1" />
        </linearGradient>
      </defs>

      <!-- 星云背景 -->
      <rect x="0" y="0" width="320" height="320" rx="16" fill="url(#nebula)" />

      <!-- 装饰环（反向旋转） -->
      <circle cx="160" cy="160" r="138" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="1" class="ring-spin" />
      <circle cx="160" cy="160" r="84" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1" stroke-dasharray="2 6" class="ring-spin-rev" />

      <!-- 卫星轨道环 -->
      <circle cx="160" cy="160" r="112" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1" />

      <!-- 轨道能量粒子（正向 / 反向） -->
      <circle r="2.5" fill="#c4b5fd">
        <animateMotion dur="9s" repeatCount="indefinite" :path="orbitPath" />
      </circle>
      <circle r="2" fill="#67e8f9">
        <animateMotion dur="6s" repeatCount="indefinite" :path="innerPath" begin="1.5s" />
      </circle>

      <!-- 中心到节点的连接线（能量流动） -->
      <line
        v-for="(link, i) in links"
        :key="`link-${i}`"
        :x1="link.x1"
        :y1="link.y1"
        :x2="link.x2"
        :y2="link.y2"
        :stroke="satellites[i].color"
        stroke-opacity="0.25"
        stroke-width="1"
        class="link-flow"
        :style="{ animationDelay: `${i * 0.35}s` }"
      />

      <!-- 核心 -->
      <circle cx="160" cy="160" r="46" fill="url(#coreGlow)" class="core-pulse" />
      <circle cx="160" cy="160" r="21" fill="url(#coreSolid)" />
      <g transform="translate(160 160)">
        <g transform="translate(-6.5 -6.5) scale(0.54)">
          <path d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" fill="#fff" />
        </g>
      </g>
      <text x="160" y="198" text-anchor="middle" class="core-label">Agent 核心</text>

      <!-- 7 个能力节点 -->
      <g
        v-for="(sat, i) in satellites"
        :key="sat.name"
        class="twinkle"
        :style="{ animationDelay: `${i * 0.35}s` }"
        :transform="`translate(${sat.x} ${sat.y})`"
      >
        <circle r="15" :fill="`${sat.color}1f`" :stroke="sat.color" stroke-width="1.2" />
        <g transform="translate(-6.5 -6.5) scale(0.54)">
          <path :d="iconPath(sat.icon)" :fill="sat.color" />
        </g>
        <text y="34" text-anchor="middle" class="sat-label">{{ sat.name }}</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
@keyframes slide-in {
  from {
    opacity: 0;
    transform: translateX(20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}
.hero-orbit {
  opacity: 0;
  animation: slide-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* 装饰环旋转（SVG 以自身中心为原点） */
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
.ring-spin {
  transform-box: fill-box;
  transform-origin: center;
  animation: spin 42s linear infinite;
}
.ring-spin-rev {
  transform-box: fill-box;
  transform-origin: center;
  animation: spin 28s linear infinite reverse;
}

/* 核心辉光呼吸 */
@keyframes pulse {
  0%,
  100% {
    opacity: 0.5;
  }
  50% {
    opacity: 1;
  }
}
.core-pulse {
  animation: pulse 2.6s ease-in-out infinite;
}

/* 连接线能量流动 */
@keyframes flow {
  to {
    stroke-dashoffset: -36;
  }
}
.link-flow {
  stroke-dasharray: 3 6;
  animation: flow 2.2s linear infinite;
}

/* 节点呼吸闪烁 */
@keyframes twinkle {
  0%,
  100% {
    opacity: 0.75;
  }
  50% {
    opacity: 1;
  }
}
.twinkle {
  animation: twinkle 3s ease-in-out infinite;
}

.core-label {
  fill: #cbd5e1;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.05em;
}
.sat-label {
  fill: #94a3b8;
  font-size: 10.5px;
}
</style>
