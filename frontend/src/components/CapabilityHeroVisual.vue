<script setup lang="ts">
// 真实技术栈分组（按功能域，不按层级）：替代原虚构分层（前端层/Pinia、存储层/Redis 等）
const techGroups = [
  { name: '界面与交互', color: '#38bdf8', items: ['Vue 3', 'TypeScript', 'Vite', 'Tailwind'] },
  { name: 'Agent 引擎', color: '#7c5cff', items: ['LangGraph', 'LangChain', 'FastAPI'] },
  { name: '模型服务', color: '#d946ef', items: ['Qwen', '多模态', 'Rerank'] },
  { name: '检索与记忆', color: '#22d3a8', items: ['Qdrant', 'Elasticsearch', 'MCP'] },
  { name: '治理与观测', color: '#f59e0b', items: ['Guardrail', '熔断', 'Telemetry'] },
]
</script>

<template>
  <div class="flex h-full w-full flex-col justify-center gap-2">
    <div
      v-for="(group, idx) in techGroups"
      :key="group.name"
      class="hero-layer animate-slide-in rounded-xl border px-4 py-3"
      :style="{
        background: `linear-gradient(135deg, ${group.color}15 0%, ${group.color}08 100%)`,
        borderColor: `${group.color}40`,
        animationDelay: `${idx * 100}ms`,
      }"
    >
      <div class="flex items-center justify-between gap-3">
        <span
          class="text-sm font-semibold"
          :style="{ color: group.color }"
        >
          {{ group.name }}
        </span>
        <div class="flex flex-wrap gap-1.5">
          <span
            v-for="item in group.items"
            :key="item"
            class="rounded bg-white/5 px-2 py-0.5 text-[11px] text-slate-300 border border-white/10"
          >
            {{ item }}
          </span>
        </div>
      </div>
    </div>
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
.hero-layer {
  opacity: 0;
  animation: slide-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
</style>
