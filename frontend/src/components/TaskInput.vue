<script setup lang="ts">
defineProps<{ modelValue: string; placeholder?: string }>()
const emit = defineEmits<{ 'update:modelValue': [v: string]; submit: [] }>()

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    emit('submit')
  }
}
</script>

<template>
  <textarea
    :value="modelValue"
    :placeholder="placeholder ?? '输入任务，例如：帮我计算 (137×0.85−20)÷3 等于多少'"
    rows="3"
    class="w-full resize-none rounded-xl border border-slate-700 bg-slate-900/70 p-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-indigo-400"
    @input="emit('update:modelValue', ($event.target as HTMLTextAreaElement).value)"
    @keydown="onKeydown"
  ></textarea>
</template>
