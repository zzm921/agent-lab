/** 源码展示的行注释：每个 code_key → 后端真实文件名 + 高亮行号区间与说明。
 * 行号为包含区间 [start, end]，对应后端源码实际行号。
 */
export interface CodeNote {
  range: [number, number]
  text: string
}

export interface CodeNotes {
  file: string
  notes: CodeNote[]
}

export const LINE_NOTES: Record<string, CodeNotes> = {
  react: {
    file: 'agents/modes/react.py',
    notes: [
      { range: [9, 10], text: '构建工具执行节点并绑定工具集（来自能力池热插拔）' },
      { range: [14, 22], text: 'agent 节点：有工具调用则输出 thinking，否则直接输出 message' },
      { range: [24, 28], text: '路由决策：根据是否请求工具决定走向 tools 还是结束' },
    ],
  },
  plan_execute: {
    file: 'agents/modes/plan_execute.py',
    notes: [
      { range: [36, 39], text: 'planner：把任务拆解为步骤并 emit plan 事件（前端渲染计划时间线）' },
      { range: [41, 52], text: 'executor：逐条执行当前步骤，可调用工具，观察结果回填' },
      { range: [60, 64], text: 'replanner：推进当前步骤，判断计划是否完成' },
      { range: [74, 82], text: '构图：planner → executor ⇄ tools → replanner 循环直至结束' },
    ],
  },
  reflection: {
    file: 'agents/modes/reflection.py',
    notes: [
      { range: [13, 22], text: 'generate：生成草稿并 emit reflect(stage=draft)' },
      { range: [24, 31], text: 'reflect：评审草稿输出批评意见，emit reflect(critique=…)' },
      { range: [33, 40], text: 'revise：根据批评修订完整答案，emit revise 增量文本' },
      { range: [43, 47], text: '路由：批评非空且未达最大轮次则继续修订，否则结束' },
    ],
  },
  multi_agent: {
    file: 'agents/modes/multi_agent.py',
    notes: [
      { range: [13, 53], text: 'worker 内执行有界工具循环，含 HITL 审批与工具执行/事件推送' },
      { range: [68, 69], text: 'orchestrator：解析任务并 emit agent_event 分派' },
      { range: [71, 85], text: 'compute / analyze 两个 Worker 并行执行，各自 emit 完成事件' },
      { range: [87, 99], text: 'aggregate：整合各 Worker 子结论，输出最终回答' },
    ],
  },
  harness: {
    file: 'agents/harness.py',
    notes: [
      { range: [83, 100], text: 'stream：构建模式图并产出 meta + 流式事件' },
      { range: [102, 117], text: 'resume：审批后按 decision / modified_args 恢复图执行' },
      { range: [119, 149], text: '核心循环：跑图、排空事件队列、遇中断产出 approval_request 并暂停' },
    ],
  },
  registry: {
    file: 'capabilities/registry.py',
    notes: [
      { range: [36, 55], text: 'list：合并内置与 MCP 能力，按 Embedding 配置判断可用性' },
      { range: [63, 83], text: 'tool_for：把能力 id 解析为 LangChain 工具，不可用返回 None' },
    ],
  },
  mcp: {
    file: 'capabilities/mcp.py',
    notes: [
      { range: [20, 59], text: 'discover：逐个连接 MCP Server 并收集工具，失败标记「不适配」' },
      { range: [61, 100], text: '按配置类型加载工具：command→stdio、url→HTTP，连接保持存活' },
    ],
  },
  calculator: {
    file: 'tools/calculator.py',
    notes: [
      { range: [7, 16], text: '运算符白名单：仅允许 + - * / % ** 等安全运算' },
      { range: [19, 35], text: 'AST 白名单安全求值，防止任意代码执行' },
      { range: [38, 52], text: 'calculator 工具：归一化中文符号（× ÷ − （））后求值' },
    ],
  },
  time_now: {
    file: 'tools/time_now.py',
    notes: [{ range: [7, 10], text: 'time_now 工具：返回当前本地日期与时间' }],
  },
  web_search: {
    file: 'tools/web_search.py',
    notes: [
      { range: [13, 35], text: 'web_search：DuckDuckGo HTML 接口搜索，解析标题/链接/摘要' },
      { range: [36, 37], text: '网络失败时降级返回提示，不中断 Agent 流程' },
    ],
  },
  rag: {
    file: 'tools/rag_tool.py',
    notes: [
      { range: [8, 16], text: 'knowledge_search：向量检索 top-k 片段，可 emit retrieve 事件可视化' },
    ],
  },
  memory: {
    file: 'tools/memory_tool.py',
    notes: [
      { range: [8, 16], text: 'memory_write：写入长期记忆并 emit memory_write 事件' },
      { range: [17, 24], text: 'memory_recall：语义召回相关事实并 emit memory_read 事件' },
    ],
  },
}
