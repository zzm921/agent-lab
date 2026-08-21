/** 对比视图运行器：同一任务在多个模式下并行运行，各自独立会话与事件流。 */
import { computed, ref } from 'vue'
import { genId, useChatStream, type ChatStream } from './useChatStream'
import type { ApprovalPolicy, ModeId, PromptStrategy } from '../types/agent'

export interface Runner {
  mode: ModeId
  stream: ChatStream
}

export interface StartOptions {
  task: string
  modes: ModeId[]
  enabled: string[]
  strategy: PromptStrategy
  policy: ApprovalPolicy
}

export function useDemoRunner() {
  const runners = ref<Runner[]>([])

  const running = computed(() =>
    runners.value.some((r) => r.stream.status === 'streaming' || r.stream.status === 'waiting_approval'),
  )

  function stopAll() {
    runners.value.forEach((r) => r.stream.stop())
  }

  async function start(opts: StartOptions) {
    stopAll()
    runners.value = opts.modes.map((mode) => {
      const stream = useChatStream()
      void stream.send({
        message: opts.task,
        mode,
        enabled: opts.enabled,
        strategy: opts.strategy,
        policy: opts.policy,
        sessionId: genId(),
      })
      return { mode, stream }
    })
  }

  return { runners, running, start, stopAll }
}
