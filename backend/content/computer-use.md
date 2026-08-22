---
id: computer-use
name: 计算机操作代理
shortDesc: 看截图、移鼠标、点按钮、敲键盘——任何有界面的软件都能成为 Agent 的工具。
icon: monitor
difficulty: adv
tags: [Computer-Use, GUI-Agent, Operator, Claude-Code]
techFilters: [MCP]
accent: '#fb7185'
experience: false
---
## 为什么需要它

传统工具调用要求软件提供 API；Computer Use 让 Agent 直接"看屏幕截图 → 定位坐标 → 移动鼠标点击 → 键盘输入"，任何 GUI 软件都成了可操作工具。代表实践：Anthropic Computer Use（2024）、OpenAI Operator、Claude Code 深度编程代理。Agent 由此从"对话框里的助手"变成能在真实环境持续运行的数字员工。

## 怎么解决

难点在感知与行动的安全——多模态视觉理解 GUI、坐标定位误差、误操作风险。业界做法：沙箱隔离执行、权限隔离、关键操作人工确认、操作审计日志，让"能干活"与"不闯祸"兼得。

## 核心实现

```python
# Computer Use：视觉感知 → 坐标行动 → 观测反馈
async def gui_agent(goal, screenshot):
    action = await model.decide_action(
        screenshot=screenshot, goal=goal,
        action_space=["click", "type", "scroll", "done"],
    )
    if action["type"] == "click":
        await screen.click(action["x"], action["y"])   # 坐标点击
    elif action["type"] == "type":
        await screen.type(action["text"])
    return await screen.snapshot()                     # 新截图 → 下一轮
```

## 收益与边界

- GUI 即工具：无 API 的软件也能被操作
- 从"对话框助手"到"数字员工"，真实环境持续运行
- 沙箱 + 权限 + 审计，守住操作安全边界
