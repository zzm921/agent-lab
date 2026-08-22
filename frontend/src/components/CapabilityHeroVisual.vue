<script setup lang="ts">
const layers = [
  { name: '前端层', color: '#38bdf8', items: ['Vue 3', 'Pinia'] },
  { name: '编排层', color: '#7c5cff', items: ['LangGraph', 'FastAPI'] },
  { name: '工具层', color: '#22d3a8', items: ['MCP', 'Tools'] },
  { name: '存储层', color: '#f59e0b', items: ['Qdrant', 'Redis'] },
]
</script>

<template>
  <div class="flex h-full w-full flex-col justify-center gap-2">
    <div
      v-for="(layer, idx) in layers"
      :key="layer.name"
      class="hero-layer animate-slide-in rounded-xl border px-4 py-3"
      :style="{
        background: `linear-gradient(135deg, ${layer.color}15 0%, ${layer.color}08 100%)`,
        borderColor: `${layer.color}40`,
        animationDelay: `${idx * 100}ms`,
      }"
    >
      <div class="flex items-center justify-between gap-3">
        <span
          class="text-sm font-semibold"
          :style="{ color: layer.color }"
        >
          {{ layer.name }}
        </span>
        <div class="flex flex-wrap gap-1.5">
          <span
            v-for="item in layer.items"
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
