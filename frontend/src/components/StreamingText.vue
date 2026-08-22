<script setup lang="ts">
import { ref, watch } from 'vue'

const props = defineProps<{ text: string; streaming?: boolean; placeholder?: string }>()

const displayed = ref('')
let raf = 0

function startTyping(target: string) {
  if (raf) {
    cancelAnimationFrame(raf)
    raf = 0
  }
  if (!props.streaming) {
    displayed.value = target
    return
  }
  if (displayed.value.length > target.length) {
    displayed.value = ''
  }
  // 自适应打字：按剩余缺口比例逐帧追赶，保证打印速度跟得上输出速度，
  // 避免长文本输出时动画长期落后、后续步骤已出现而当前内容还在“打字”。
  const tick = () => {
    if (displayed.value.length < target.length) {
      const gap = target.length - displayed.value.length
      const step = Math.max(1, Math.ceil(gap / 12))
      displayed.value = target.slice(0, displayed.value.length + step)
      raf = requestAnimationFrame(tick)
    } else {
      raf = 0
    }
  }
  raf = requestAnimationFrame(tick)
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
      if (raf) {
        cancelAnimationFrame(raf)
        raf = 0
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
