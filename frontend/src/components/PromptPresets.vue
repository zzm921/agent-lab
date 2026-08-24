<script setup lang="ts">
defineProps<{ presets: string[] }>()
const emit = defineEmits<{ insert: [v: string] }>()
</script>

<template>
  <div class="mt-3">
    <span class="text-[11px] text-slate-500">快捷 Prompt · 来自已启用能力</span>
    <div v-if="presets.length" class="mt-2 flex flex-wrap gap-1.5">
      <span
        v-for="(p, i) in presets"
        :key="i"
        class="flex max-w-full items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/50 px-2 py-1 text-xs text-slate-300 transition hover:border-indigo-400 hover:text-white"
        role="button"
        :title="p"
        @click="emit('insert', p)"
      >
        <span class="truncate">{{ p }}</span>
      </span>
    </div>
    <div v-else class="mt-2 text-[11px] text-slate-600">
      暂无快捷 prompt，启用能力卡片后自动出现
    </div>
  </div>
</template>
