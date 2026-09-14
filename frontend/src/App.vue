<script setup lang="ts">
import AppHeader from './components/AppHeader.vue'
</script>

<template>
  <div class="flex h-dvh flex-col overflow-hidden">
    <AppHeader />
    <main class="relative flex-1 min-h-0 overflow-y-auto">
      <RouterView />
    </main>
  </div>
</template>
