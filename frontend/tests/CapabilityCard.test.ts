import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import CapabilityCard from '../src/components/CapabilityCard.vue'
import type { Capability } from '../src/types/agent'

const available: Capability = {
  id: 'calculator',
  name: '计算器',
  source: 'builtin',
  desc: '安全计算数学表达式',
  example: '帮我计算 1+1',
  code_key: 'calculator',
  availability: 'available',
  unavailable_reason: null,
}

const unavailable: Capability = {
  id: 'rag',
  name: '知识库检索',
  source: 'builtin',
  desc: '向量检索',
  example: '检索一下',
  code_key: 'rag',
  availability: 'unavailable',
  unavailable_reason: '未配置 Embedding API Key',
}

describe('CapabilityCard', () => {
  it('可用能力渲染名称/来源/描述，开关可点击并发出 toggle', async () => {
    const wrapper = mount(CapabilityCard, { props: { cap: available, enabled: false } })
    expect(wrapper.text()).toContain('计算器')
    expect(wrapper.text()).toContain('内置')
    expect(wrapper.text()).toContain('安全计算数学表达式')
    expect(wrapper.text()).toContain('可用')

    const sw = wrapper.find('[role="switch"]')
    expect(sw.attributes('aria-checked')).toBe('false')
    expect(sw.attributes('disabled')).toBeUndefined()
    await sw.trigger('click')
    expect(wrapper.emitted('toggle')?.[0]).toEqual(['calculator'])
  })

  it('启用状态正确反映 aria-checked', () => {
    const wrapper = mount(CapabilityCard, { props: { cap: available, enabled: true } })
    expect(wrapper.find('[role="switch"]').attributes('aria-checked')).toBe('true')
  })

  it('不可用能力置灰显示不适配与原因，开关禁用且不发出事件', async () => {
    const wrapper = mount(CapabilityCard, { props: { cap: unavailable, enabled: false } })
    expect(wrapper.text()).toContain('不适配')
    expect(wrapper.text()).toContain('未配置 Embedding API Key')

    const sw = wrapper.find('[role="switch"]')
    expect(sw.attributes('disabled')).toBeDefined()
    await sw.trigger('click')
    expect(wrapper.emitted('toggle')).toBeUndefined()
  })

  it('可用能力点击「示例」发出 example 事件', async () => {
    const wrapper = mount(CapabilityCard, { props: { cap: available, enabled: false } })
    await wrapper.find('.example-btn').trigger('click')
    expect(wrapper.emitted('example')?.[0]).toEqual([available])
  })

  it('不可用能力不渲染示例按钮', () => {
    const wrapper = mount(CapabilityCard, { props: { cap: unavailable, enabled: false } })
    expect(wrapper.find('.example-btn').exists()).toBe(false)
  })
})
