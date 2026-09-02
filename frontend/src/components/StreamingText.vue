<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import hljs from 'highlight.js'
import { marked } from 'marked'

const props = defineProps<{
  text: string
  streaming?: boolean
  placeholder?: string
  /** 开启后按 markdown 渲染（代码块走 highlight.js 高亮），流式期间渲染增量内容 */
  markdown?: boolean
}>()

const displayed = ref('')
let raf = 0

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

/** markdown 模式：把已显示的增量文本实时解析为 HTML */
const html = computed(() => (props.markdown ? (marked.parse(displayed.value) as string) : ''))

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
  <div v-if="markdown && displayed" class="markdown-body break-words" v-html="html"></div>
  <span v-else-if="markdown" class="whitespace-pre-wrap break-words">{{ placeholder }}</span>
  <span v-else class="whitespace-pre-wrap break-words">{{ displayed || placeholder }}</span>
  <span
    v-if="streaming"
    class="ml-0.5 inline-block h-4 w-2 animate-pulse rounded-sm bg-indigo-400 align-middle"
  ></span>
</template>
