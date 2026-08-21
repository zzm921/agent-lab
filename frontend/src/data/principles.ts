/** 各模式与核心技术（MCP / 能力热插拔 / RAG / 记忆 / HITL / 提示词工程）的原理文案。 */
export interface Principle {
  title: string
  summary: string
  points: string[]
}

export const PRINCIPLES: Record<string, Principle> = {
  react: {
    title: 'ReAct 原理',
    summary: '让大模型交替执行推理（Reasoning）与行动（Action），借助外部工具扩展能力边界。',
    points: [
      '每轮 Agent 生成一段思考文本，同时决定是否调用工具。',
      '工具返回的观察结果作为新上下文，驱动下一轮思考，形成「思考→行动→观察」闭环。',
      '当模型不再请求工具时，循环结束并输出最终回答。',
      '工具集由「能力池」动态注入，这就是能力热插拔的落地方式。',
    ],
  },
  plan_execute: {
    title: 'Plan-and-Execute 原理',
    summary: '将复杂任务先分解为可验证的子步骤，再逐步执行，降低单步推理难度。',
    points: [
      '规划器把用户任务拆成 2-5 个有序子步骤，先建立整体蓝图。',
      '执行器一次只处理当前步骤，专注度高、出错率低。',
      '执行完一步后由 replanner 判断是否还有剩余步骤，动态推进。',
      '工具调用发生步骤内部，观察结果参与后续步骤的执行。',
    ],
  },
  reflection: {
    title: 'Reflection 原理',
    summary: '让模型扮演「生成者」与「评审者」两个角色，通过批评-修订迭代提升回答质量。',
    points: [
      '生成器先产出草稿，评审器以严格标准找出不足并给出建议。',
      '若批评非空，修订器按意见重写完整答案，再交给评审器复审。',
      '循环持续到批评为空或达到最大轮次（max_iterations），避免无限迭代。',
      '适用于质量要求高、一次成稿不满意的生成类任务。',
    ],
  },
  multi_agent: {
    title: 'Multi-Agent 原理',
    summary: '用多个各司其职的子智能体并行协作，再把子结论汇总成统一答案。',
    points: [
      'Orchestrator 负责理解任务并按性质分派给不同的 Worker。',
      '计算 Worker 专注数值运算（绑定计算器工具），分析 Worker 负责逻辑归纳，两者并行。',
      '每个 Worker 内部可独立执行有界的工具循环（含 HITL 审批）。',
      '汇总者整合各 Worker 的子结论，输出结构完整的最终答案。',
    ],
  },
  mcp: {
    title: 'MCP 集成原理',
    summary: 'Model Context Protocol 是连接大模型与外部工具/数据的开放协议，支持 stdio 与 HTTP 传输。',
    points: [
      '后端按 MCP_SERVERS 配置连接各 Server，自动发现其工具并暴露为「能力」。',
      '能力 id 形如 <server>:<tool>，来源标记为 mcp。',
      '连接失败或未配置的 Server 能力被标记为「不适配」并置灰，不注入 Agent。',
      '因此 MCP 是一种「外挂能力」：配置即出现，失败即降级。',
    ],
  },
  hotswap: {
    title: '能力热插拔原理',
    summary: '能力池把「工具」抽象为可开关的「能力卡片」，随请求动态组装 Agent 工具集。',
    points: [
      '每个能力卡片对应一个后端可解析的工具或工具组。',
      '开关状态写入 enabled_capabilities，随 /api/stream 请求体发送。',
      '后端按 enabled 列表组装工具集：启用即注入、关闭即移除（tools_builder）。',
      '同一会话可随时增减能力，无需重启，这就是「热插拔」。',
    ],
  },
  rag: {
    title: 'RAG 知识库检索原理',
    summary: '检索增强生成：先向量检索最相关片段，再让模型基于片段作答，降低幻觉。',
    points: [
      '查询文本经 Embedding 模型编码为向量，与内置知识库向量做相似度检索。',
      '命中片段带相关度分数，作为上下文注入模型。',
      '未配置 Embedding Key 时该能力标记「不适配」，前端置灰并给出原因。',
      '检索过程通过 retrieve 事件实时可视化展示。',
    ],
  },
  memory: {
    title: '长期记忆原理',
    summary: '跨轮次记住关键事实，后续对话按语义召回，让 Agent 具备「记忆」。',
    points: [
      'memory_write 把重要事实写入会话级向量记忆库。',
      'memory_recall 按查询语义召回相关事实，作为上下文参与推理。',
      '记忆与 RAG 共用 Embedding 能力，因此同样受 Key 配置影响。',
      '读写通过 memory_write / memory_read 事件实时展示。',
    ],
  },
  hitl: {
    title: 'HITL 人工审批原理',
    summary: 'Human-in-the-Loop：工具执行前可暂停等待人工批准、拒绝或修改参数。',
    points: [
      'approval_policy=always 时，工具调用前触发 GraphInterrupt 中断。',
      '后端发出 approval_request 事件，携带待审批的 tool_calls。',
      '前端弹窗提供「批准 / 拒绝 / 修改」三种决策。',
      '决策经 /api/approve 恢复执行：修改则用新参数替换原调用。',
    ],
  },
  prompting: {
    title: '提示词策略原理',
    summary: '通过调整系统提示词结构，影响模型的回答方式与质量，可切换对比。',
    points: [
      'standard：直接、准确地回答，不加额外结构。',
      'few_shot：给出输入-输出的示例格式，引导模型按结构作答。',
      'cot：要求先逐步思考（chain-of-thought）再给出结论。',
      '策略随请求体 prompt_strategy 字段生效，可在同一任务下对比效果。',
    ],
  },
}
