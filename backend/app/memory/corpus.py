"""RAG 内嵌知识语料：关于 LangChain / LangGraph / Agent 的小型知识库。"""
KNOWLEDGE_CORPUS: list[str] = [
    "LangGraph 是 LangChain 推出的用于构建有状态、多步骤 AI Agent 的编排框架，核心是 StateGraph，"
    "通过节点和边定义状态转换，节点读写共享的 AgentState。",
    "StateGraph 中的状态用 TypedDict 定义，消息列表使用 add_messages 归约器实现消息累积；"
    "每个节点接收当前状态并返回部分状态更新。",
    "ReAct 模式由'思考(Thought)-行动(Action)-观察(Observation)'循环组成：模型先思考下一步，"
    "再选择并调用工具，把工具结果作为观察继续思考，直到产出最终答案。",
    "plan-and-execute 模式先把任务分解为多个子步骤计划，再逐条执行，执行后重新规划（replan）"
    "以决定继续或收尾，适合复杂、多步骤任务。",
    "reflection 模式先生成一份草稿答案，再由反思节点评估其不足并给出批评，据此修订生成更优答案，"
    "迭代直到批评为空或达到最大轮次。",
    "Human-in-the-loop 通过 LangGraph 的 interrupt() 在关键节点暂停图执行，等待人工审批或澄清，"
    "再用 Command(resume=...) 从暂停点继续，常用于工具执行前的人工确认。",
    "MCP（Model Context Protocol）是开放协议，把外部工具/数据源封装为 MCP Server，"
    "Agent 可通过 MCP Client 动态发现并调用这些工具，实现能力热插拔。",
    "Embedding 把文本映射为稠密向量，RAG 先对查询向量化，再从向量库检索最相关的文档片段，"
    "将其注入上下文后由大模型生成有依据的回答。",
    "长期记忆把关键事实向量化存入记忆库，后续对话按语义召回相关记忆，从而跨轮次记住用户信息。",
    "多智能体协作由 Orchestrator（监督者）负责任务分派，把子任务交给专门的 Worker 执行，"
    "再汇总各 Worker 结果形成最终输出。",
]
