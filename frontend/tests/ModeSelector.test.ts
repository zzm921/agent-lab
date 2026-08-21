import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import ModeSelector from '../src/components/ModeSelector.vue'

describe('ModeSelector', () => {
  it('渲染 4 种模式', () => {
    const wrapper = mount(ModeSelector, { props: { modelValue: 'react' } })
    expect(wrapper.findAll('button')).toHaveLength(4)
    expect(wrapper.text()).toContain('ReAct')
    expect(wrapper.text()).toContain('计划执行')
    expect(wrapper.text()).toContain('反思修订')
    expect(wrapper.text()).toContain('多智能体')
  })

  it('点击未选中模式发出 update:modelValue', async () => {
    const wrapper = mount(ModeSelector, { props: { modelValue: 'react' } })
    const buttons = wrapper.findAll('button')
    await buttons[2].trigger('click') // reflection
    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual(['reflection'])
  })

  it('当前选中模式高亮', () => {
    const wrapper = mount(ModeSelector, { props: { modelValue: 'multi_agent' } })
    const active = wrapper.findAll('button').find((b) => b.text().includes('多智能体'))
    expect(active?.classes()).toContain('border-indigo-400')
  })
})
