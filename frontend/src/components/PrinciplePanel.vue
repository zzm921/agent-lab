<script setup lang="ts">
import { PRINCIPLES } from '../data/principles'

defineProps<{ topic: string }>()
</script>

<template>
  <div v-if="PRINCIPLES[topic]" class="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
    <h3 class="mb-1 text-base font-semibold text-white">{{ PRINCIPLES[topic].title }}</h3>
    <p class="mb-3 text-sm text-slate-400">{{ PRINCIPLES[topic].summary }}</p>
    <ul class="space-y-2">
      <li v-for="(p, i) in PRINCIPLES[topic].points" :key="i" class="flex gap-2 text-sm text-slate-300">
        <span class="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-400"></span>
        <span>{{ p }}</span>
      </li>
    </ul>
  </div>
</template>
