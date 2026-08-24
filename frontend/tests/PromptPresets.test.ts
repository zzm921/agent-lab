import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import PromptPresets from '../src/components/PromptPresets.vue'

const presets = ['帮我计算 1+1', '现在几点？', '搜索一下']

describe('PromptPresets', () => {
  it('渲染传入的快捷 prompt，点击标签发出 insert', async () => {
    const wrapper = mount(PromptPresets, { props: { presets } })
    expect(wrapper.text()).toContain('快捷 Prompt')
    const chips = wrapper.findAll('[role="button"]')
    expect(chips).toHaveLength(3)
    await chips[1].trigger('click')
    expect(wrapper.emitted('insert')?.[0]?.[0]).toBe('现在几点？')
  })

  it('空列表时展示空态提示且不渲染标签', () => {
    const wrapper = mount(PromptPresets, { props: { presets: [] } })
    expect(wrapper.findAll('[role="button"]')).toHaveLength(0)
    expect(wrapper.text()).toContain('暂无快捷 prompt')
  })
})
