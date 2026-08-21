import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import ApprovalDialog from '../src/components/ApprovalDialog.vue'
import type { ApprovalRequest } from '../src/types/agent'

const approval: ApprovalRequest = {
  approval_id: 'a1',
  tool_calls: [
    { name: 'calculator', args: { expression: '1+1' }, id: 'call1' },
    { name: 'time_now', args: {}, id: 'call2' },
  ],
}

const mountDialog = () =>
  mount(ApprovalDialog, { props: { approval }, global: { stubs: { Teleport: true } } })

function buttonByText(wrapper: ReturnType<typeof mountDialog>, text: string) {
  const btn = wrapper.findAll('button').find((b) => b.text().includes(text))
  if (!btn) throw new Error(`button not found: ${text}`)
  return btn
}

describe('ApprovalDialog', () => {
  it('渲染待审批工具名称与参数', () => {
    const wrapper = mountDialog()
    expect(wrapper.text()).toContain('calculator')
    expect(wrapper.text()).toContain('time_now')
    expect(wrapper.text()).toContain('审批')
    // 参数 JSON 位于可编辑 textarea 的 value 中
    const value = (wrapper.find('textarea').element as HTMLTextAreaElement).value
    expect(value).toContain('expression')
  })

  it('批准发出 approve 决策', async () => {
    const wrapper = mountDialog()
    await buttonByText(wrapper, '批准').trigger('click')
    expect(wrapper.emitted('decision')?.[0]).toEqual([{ decision: 'approve' }])
  })

  it('拒绝发出 reject 决策', async () => {
    const wrapper = mountDialog()
    await buttonByText(wrapper, '拒绝').trigger('click')
    expect(wrapper.emitted('decision')?.[0]).toEqual([{ decision: 'reject' }])
  })

  it('修改参数后提交发出 modify 决策与 modified_args', async () => {
    const wrapper = mountDialog()
    const textareas = wrapper.findAll('textarea')
    await textareas[0].setValue('{"expression":"2+2"}')
    await buttonByText(wrapper, '修改并提交').trigger('click')

    const payload = wrapper.emitted('decision')?.[0]?.[0] as {
      decision: string
      modifiedArgs: Record<string, unknown>
    }
    expect(payload.decision).toBe('modify')
    expect(payload.modifiedArgs.call1).toEqual({ expression: '2+2' })
    expect(payload.modifiedArgs.call2).toEqual({})
  })

  it('非法 JSON 显示错误且不提交', async () => {
    const wrapper = mountDialog()
    await wrapper.find('textarea').setValue('{bad json')
    await buttonByText(wrapper, '修改并提交').trigger('click')
    expect(wrapper.text()).toContain('解析失败')
    expect(wrapper.emitted('decision')).toBeUndefined()
  })
})
