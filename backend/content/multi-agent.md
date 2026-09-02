---
id: multi-agent
name: 多智能体编排
shortDesc: Orchestrator 统一调度多个专业 Agent，各司其职协作完成复杂任务。
icon: network
difficulty: adv
completeLevel: 85
tags: [Orchestrator, Multi-Agent, Coordination, Agent]
techFilters: [LangGraph, MCP]
accent: '#7c5cff'
mode: multi_agent
enabledTools: [rag]
prompts:
  - 你是项目经理：把「上线一个 AI 助手网站」拆给研究员、开发者、测试员三个角色，分派任务并汇总执行方案。
  - 让「策划师 + 文案 + 设计师」三个角色协作，为新品咖啡出一份上市营销方案。
  - 新员工入职需要准备哪些材料？请分角色给出清单
---
## 为什么需要它

多智能体模式由一个编排者（Orchestrator）接收任务，分析后分派给不同专长的子 Agent（如研究员、程序员、分析师），各子 Agent 独立工作后将结果汇总给编排者。单个 Agent 的上限是"全能但容易过载的人"，多 Agent 让不同角色专精协作。

## 怎么解决

核心挑战在于子 Agent 之间的协作协议、上下文传递和结果汇总。我设计了标准化的任务分派格式（TaskTicket），每个子 Agent 有明确的能力描述和输入输出 schema，Orchestrator 根据能力路由任务。

## 核心实现

```python
# 多智能体编排器
class OrchestratorAgent:
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self.router = TaskRouter(agents)

    async def execute(self, task: str):
        # 1. 任务分解与路由
        subtasks = self.router.decompose(task)

        # 2. 并行/串行分派
        results = {}
        for subtask in subtasks:
            agent = self.agents[subtask.agent_id]
            results[subtask.id] = await agent.execute(
                subtask.payload
            )

        # 3. 结果汇总
        return self.synthesize(task, results)
```

## 收益与边界

- 标准化 TaskTicket 协议，子 Agent 可插拔替换
- 支持并行分派与依赖编排，提升复杂任务效率
- 每个子 Agent 有独立上下文，互不干扰
