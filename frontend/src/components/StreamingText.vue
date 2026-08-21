<script setup lang="ts">
import { ref, watch } from 'vue'

const props = defineProps<{ text: string; streaming?: boolean; placeholder?: string }>()

const displayed = ref('')
const timer = ref<ReturnType<typeof setInterval> | null>(null)

function startTyping(target: string) {
  if (timer.value) {
    clearInterval(timer.value)
    timer.value = null
  }
  if (!props.streaming) {
    displayed.value = target
    return
  }
  if (displayed.value.length > target.length) {
    displayed.value = ''
  }
  timer.value = setInterval(() => {
    if (displayed.value.length < target.length) {
      displayed.value = target.slice(0, displayed.value.length + 1)
    } else {
      if (timer.value) {
        clearInterval(timer.value)
        timer.value = null
      }
    }
  }, 18)
}

watch(
  () => props.text,
  (target) => {
    startTyping(target)
  },
  { immediate: true },
)

watch(
  () => props.streaming,
  (streaming) => {
    if (!streaming) {
      if (timer.value) {
        clearInterval(timer.value)
        timer.value = null
      }
      displayed.value = props.text
    }
  },
)
</script>

<template>
  <span class="whitespace-pre-wrap break-words">{{ displayed || placeholder }}</span>
  <span
    v-if="streaming"
    class="ml-0.5 inline-block h-4 w-2 animate-pulse rounded-sm bg-indigo-400 align-middle"
  ></span>
</template>
