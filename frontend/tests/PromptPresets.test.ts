import { beforeEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import PromptPresets from '../src/components/PromptPresets.vue'

const STORAGE_KEY = 'agent-lab-prompts'

describe('PromptPresets', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('无存储时展示内置示例，点击标签发出 insert（内容为完整 prompt）', async () => {
    const wrapper = mount(PromptPresets)
    expect(wrapper.text()).toContain('快捷 Prompt')
    const chips = wrapper.findAll('[role="button"]')
    expect(chips.length).toBeGreaterThan(0)
    await chips[0].trigger('click')
    expect(wrapper.emitted('insert')?.[0]?.[0]).toBe(chips[0].text())
  })

  it('管理面板可新增 prompt 并持久化到 localStorage', async () => {
    const wrapper = mount(PromptPresets)
    await wrapper.find('.manage-toggle').trigger('click')
    await wrapper.find('.preset-input').setValue('自定义 prompt')
    await wrapper.find('.preset-add').trigger('click')
    expect(wrapper.text()).toContain('自定义 prompt')
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]') as string[]
    expect(stored).toContain('自定义 prompt')
  })

  it('管理面板可删除 prompt 且不回退内置示例', async () => {
    const wrapper = mount(PromptPresets)
    const before = wrapper.findAll('[role="button"]').length
    await wrapper.find('.manage-toggle').trigger('click')
    await wrapper.findAll('.preset-remove')[0].trigger('click')
    expect(wrapper.findAll('[role="button"]').length).toBe(before - 1)
    // 清空后标签消失，展示空态提示
    for (let i = before - 1; i > 0; i--) {
      await wrapper.findAll('.preset-remove')[0].trigger('click')
    }
    expect(wrapper.findAll('[role="button"]').length).toBe(0)
    expect(wrapper.text()).toContain('暂无快捷 prompt')
  })

  it('读取已有存储并渲染', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(['已有的 prompt']))
    const wrapper = mount(PromptPresets)
    expect(wrapper.text()).toContain('已有的 prompt')
  })
})
