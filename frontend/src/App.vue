<script setup lang="ts">
import AppHeader from './components/AppHeader.vue'
import FooterBar from './components/FooterBar.vue'
</script>

<template>
  <div class="flex h-dvh flex-col overflow-hidden">
    <AppHeader />
    <main class="relative flex-1 min-h-0 overflow-y-auto">
      <RouterView />
    </main>
    <FooterBar />
  </div>
</template>
