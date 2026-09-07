<script setup lang="ts">
/** 会话侧边栏：多会话管理（新建 / 切换 / 重命名 / 删除）。
 *
 * 数据源为后端会话元数据（按客户端隔离、文件持久化、7 天 TTL），
 * 切换会话仅更新「当前会话 id」，历史由后端按 session_id 恢复；
 * 删除会话联动后端清理其 checkpoint 线程与会话记忆库。
 */
import { ref } from 'vue'
import type { SessionMeta } from '../composables/useSessions'

const props = defineProps<{
  sessions: SessionMeta[]
  currentId: string
  loading?: boolean
  error?: string
  open?: boolean
}>()

const emit = defineEmits<{
  create: []
  select: [id: string]
  rename: [id: string, title: string]
  remove: [id: string]
  close: []
}>()

/** 会话最后活跃时间的相对展示：今天 → HH:mm，昨天 → 昨天，更早 → M月d日 */
function relTime(ts: number): string {
  const d = new Date(ts)
  const now = new Date()
  if (d.toDateString() === now.toDateString()) {
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  }
  const yest = new Date(now)
  yest.setDate(now.getDate() - 1)
  if (d.toDateString() === yest.toDateString()) return '昨天'
  return `${d.getMonth() + 1}月${d.getDate()}日`
}

/** 内联重命名状态 */
const renamingId = ref('')
const renameText = ref('')

function startRename(s: SessionMeta) {
  renamingId.value = s.session_id
  renameText.value = s.title
}

function commitRename() {
  const title = renameText.value.trim()
  const id = renamingId.value
  renamingId.value = ''
  if (title) emit('rename', id, title)
}

function onRemove(s: SessionMeta) {
  const name = s.title || '新会话'
  if (!window.confirm(`删除会话「${name}」？其对话历史与会话记忆将一并清除。`)) return
  emit('remove', s.session_id)
}
</script>

<template>
  <aside
    class="fixed inset-y-0 left-0 z-40 flex w-52 flex-col border-r border-slate-800 bg-slate-950/95 backdrop-blur md:static md:inset-auto md:z-20 md:bg-slate-950"
    :class="open ? 'translate-x-0' : '-translate-x-full md:translate-x-0'"
  >
    <div class="flex items-center justify-between border-b border-slate-800 px-3 py-3">
      <div class="flex items-center gap-2">
        <svg class="h-4 w-4 text-indigo-300" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M8 10h8m-8 4h5m-8-8h8m-8 0a2 2 0 100 4 2 2 0 000-4zm0 0a2 2 0 112 2m8-2a2 2 0 112 2M4 20a5 5 0 0116 0" />
        </svg>
        <div>
          <h2 class="text-sm font-semibold text-white">会话</h2>
          <p class="text-[10px] text-slate-500">保留 7 天</p>
        </div>
      </div>
      <div class="flex items-center gap-1">
        <button
          type="button"
          title="新建会话"
          class="rounded-lg p-1.5 text-slate-300 transition hover:bg-slate-800 hover:text-white"
          @click="emit('create')"
        >
          <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
          </svg>
        </button>
        <button
          type="button"
          class="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white md:hidden"
          @click="emit('close')"
        >
          <svg class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>

    <div class="flex-1 space-y-1 overflow-y-auto px-2 py-2">
      <div v-if="error" class="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-300">
        会话加载失败：{{ error }}
      </div>

      <div
        v-else-if="!loading && !sessions.length"
        class="rounded-lg border border-dashed border-slate-700/70 px-3 py-4 text-center"
      >
        <p class="text-[11px] text-slate-500">暂无会话</p>
        <button
          type="button"
          class="mt-2 rounded-lg border border-indigo-500/40 bg-indigo-500/10 px-3 py-1 text-[11px] text-indigo-300 transition hover:bg-indigo-500/20"
          @click="emit('create')"
        >
          新建会话
        </button>
      </div>

      <div
        v-for="s in sessions"
        :key="s.session_id"
        class="group relative rounded-lg transition"
        :class="s.session_id === currentId ? 'bg-indigo-500/15 ring-1 ring-indigo-500/40' : 'hover:bg-slate-800/60'"
      >
        <button
          type="button"
          class="flex w-full items-center gap-2 px-2.5 py-2 text-left"
          :title="s.title || '新会话'"
          @click="emit('select', s.session_id)"
        >
          <span class="min-w-0 flex-1">
            <!-- 内联重命名：输入框替代标题 -->
            <input
              v-if="renamingId === s.session_id"
              v-model="renameText"
              type="text"
              class="w-full rounded border border-indigo-500/50 bg-slate-900 px-1.5 py-0.5 text-xs text-white outline-none"
              maxlength="30"
              placeholder="会话标题"
              @keydown.enter.prevent="commitRename"
              @keydown.esc="renamingId = ''"
              @blur="commitRename"
              @click.stop
            />
            <template v-else>
              <span
                class="block truncate text-xs"
                :class="s.session_id === currentId ? 'font-medium text-white' : 'text-slate-300'"
              >
                {{ s.title || '新会话' }}
              </span>
              <span class="block text-[10px] text-slate-500">
                {{ relTime(s.last_active) }} · {{ s.message_count }} 轮
              </span>
            </template>
          </span>
        </button>

        <!-- 悬浮操作：重命名 / 删除 -->
        <div
          v-if="renamingId !== s.session_id"
          class="absolute right-1.5 top-1/2 hidden -translate-y-1/2 gap-0.5 group-hover:flex"
        >
          <button
            type="button"
            title="重命名"
            class="rounded p-1 text-slate-400 transition hover:bg-slate-700 hover:text-white"
            @click.stop="startRename(s)"
          >
            <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.862 4.487z" />
            </svg>
          </button>
          <button
            type="button"
            title="删除（联动清理历史与记忆）"
            class="rounded p-1 text-slate-400 transition hover:bg-rose-500/20 hover:text-rose-300"
            @click.stop="onRemove(s)"
          >
            <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  </aside>
</template>
