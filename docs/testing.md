# 测试说明与报告（Testing）

## 1. 测试策略

- **后端**（pytest，`backend/tests/`）：不联网，注入 `FakeChatModel` / `FakeEmbeddings`，MCP 用 mock。`asyncio_mode = auto`，`pythonpath = .`。
- **前端**（vitest + @vue/test-utils + jsdom，`frontend/tests/`）。

## 2. 运行方式

```bash
# 后端
cd backend
python -m pytest -q

# 前端
cd frontend
npm run test
```

## 3. 测试用例清单

### 3.1 后端（68 个用例，8 个文件）

**test_dashscope_chat.py（10）** — DashScope 原生 SDK 适配（reason/output 分离）
- `test_parse_message_distinguishes_reason_and_output`：响应解析明确分离 `reasoning_content`(reason) 与 `content`(output)
- `test_parse_message_with_tool_calls`：携带工具调用时 output/reasoning 可为空
- `test_to_lc_tool_calls` / `test_to_lc_tool_calls_bad_json_guards`：DashScope tool_calls → LangChain 格式（arguments 解析为 dict / 非法 JSON 兜底）
- `test_tool_call_chunks_accumulate_via_add`：流式增量 tool_calls 经 `AIMessageChunk.__add__` 合并为完整调用
- `test_to_dashscope_messages_mapping`：LangChain 消息 → DashScope 请求格式（system/user/assistant(tool_calls)/tool）
- `test_tool_to_ds_from_dict_passthrough`：dict 工具透传
- `test_to_message_carries_reasoning_in_kwargs` / `test_to_message_with_tool_calls`：DashScopeTurn → AIMessage（reasoning 入 additional_kwargs / tool_calls 转换）
- `test_payload_sets_thinking_and_tools`：`enable_thinking` 生效、tools 透传、temperature 缺省用实例值

**test_api_stream.py（9）** — API 层
- `test_health`：`/api/health` 返回状态、模型、MCP/Embedding 配置标记
- `test_capabilities_with_fake_registry`：`/api/capabilities` 返回能力目录
- `test_stream_sse_with_fake_runner`：`/api/stream` 产出 SSE 事件序列（meta/done）
- `test_approve_sse`：`/api/approve` 恢复执行并产出事件
- `test_source_returns_code` / `test_source_unknown_module`：源码接口（存在/未知模块）
- `test_sandbox_files_list_and_download`：`/api/sandbox/files` 列出 + `/files/download` 下载（含子目录）
- `test_sandbox_files_download_traversal_blocked`：路径穿越/绝对路径被拒绝（400）
- `test_sandbox_files_download_not_found`：下载不存在文件 → 404

**test_capabilities.py（3）** — 能力注册表
- `test_builtin_available`：内置能力可用（有 Embedding）
- `test_rag_memory_unavailable_without_embedding`：缺 Embedding Key → rag/memory「不适配」并给出原因
- `test_tool_for_unavailable_returns_none`：不可用能力 `tool_for` 返回 None

**test_mcp.py（3）** — MCP 集成
- `test_mcp_discover_success`：mock 连接成功 → 工具以 MCP 能力列出
- `test_mcp_discover_failure_marks_unavailable`：连接失败 → 标记「不适配」
- `test_mcp_partial_failure`：多 server 部分失败不影响其它

**test_tools.py（16）** — 工具与向量检索
- `test_calculator_basic` / `test_calculator_chinese_operators` / `test_calculator_unsafe_rejected`：计算器基本/中文符号/不安全表达式拒绝
- `test_time_now_format`：时间格式
- `test_run_command_echo`：沙箱命令执行返回输出
- `test_run_command_deny_dangerous` / `test_run_command_deny_empty`：危险命令/空命令拒绝
- `test_run_command_timeout`：超时硬杀并提示
- `test_run_command_output_truncated`：超长输出截断
- `test_run_command_opensandbox_backend_dispatch`：backend=opensandbox 时命令交给 OpenSandbox 执行器
- `test_run_command_opensandbox_sdk_missing`：SDK 未安装时优雅降级提示
- `test_run_command_local_persists_to_work_dir`：local 后端写入的文件持久化到工作目录（供下载）
- `test_run_command_opensandbox_mounts_work_volume`：opensandbox 后端把工作目录以 Volume 挂载进沙箱
- `test_run_command_opensandbox_volume_passes_sdk_converter`：回归——挂载卷能通过 SDK 真实转换器（规避 0.1.15 `Unset.claim_name` bug）
- `test_vector_store_search_topk` / `test_vector_store_empty`：余弦 top-k 检索 / 空库

**test_tools_builder.py（5）** — 能力热插拔组装
- `test_build_tools_by_enabled`：按 enabled 组装工具
- `test_build_tools_empty`：空列表 → 空工具集
- `test_build_tools_memory_two_tools`：memory 能力解析为写/读两个工具
- `test_build_tools_skips_unavailable`：跳过不可用能力
- `test_build_tools_dedupe`：按名称去重

**test_modes.py（7）** — 四种模式
- `test_react_direct_answer` / `test_react_tool_loop`：ReAct 直接回答 / 工具循环
- `test_plan_execute`：计划-执行-再计划流程
- `test_plan_execute_replan`：步骤工具失败触发 replanner 重规划后继续执行
- `test_reflection_revise_loop`：生成-反思-修订循环
- `test_multi_agent`：编排者分派 Worker 汇总
- `test_unknown_mode`：未知模式报错

**test_retry.py（14）** — 两层重试机制
- 单元：`test_is_retryable_exception_marks_transient` / `test_is_retryable_exception_marks_permanent`：瞬时（超时/连接/5xx/429）与确定性（参数/4xx/FileNotFound）错误分类
- 单元：`test_is_retryable_status`：状态码判定（429/5xx 可重试，400/404 不可）
- 单元：`test_backoff_delay_within_bounds`：指数退避+抖动在 `[base*0.5, cap]` 内
- 单元：`test_format_tool_error_retryable_exhausted` / `test_format_tool_error_permanent`：结构化错误文案（错误类型/详情/建议）
- 单元：`test_invoke_with_retry_succeeds_after_transient_failures`：瞬时错误直接重试后成功（发 2 次 tool_retry 事件）
- 单元：`test_invoke_with_retry_exhausted`：重试耗尽返回错误；`test_invoke_with_retry_skips_permanent`：确定性错误不重试
- 集成：`test_tool_layer_retry_transparent`：瞬时错误透明重试成功，模型只看到成功结果
- 集成：`test_tool_layer_retry_exhausted_returns_structured_error`：重试耗尽返回结构化错误给模型
- 集成：`test_permanent_error_no_direct_retry`：确定性错误不直接重试
- 集成：`test_agent_layer_retry_cap_gives_up`：同工具连续失败达上限 →「请改用其它工具」短路
- 集成：`test_harness_agent_retry_cap_and_reset`：harness 连续失败计数与成功后清零
- 故障注入目录：`test_fault_catalog_classification`（13 种类型按 retryable/permanent 分类）、`test_fault_spec_classification`（规格含重试分类、off 清除）、`test_fault_unknown_type_rejected`（未知类型 ValueError）

**test_modes.py 故障注入类型**（新增）
- `test_fault_transient_type_triggers_direct_retry`：注入 `http_500` → 发 `tool_retry` 事件（工具层直接重试），耗尽返回结构化错误，且短路审批
- `test_fault_permanent_type_goes_to_model`：注入 `http_400` → 无 `tool_retry`，错误直接返回给模型

**test_api_stream.py 故障注入 API**（新增）
- `test_fault_types_endpoint`：`GET /api/faults/types` 返回类型目录及重试分类
- `test_fault_unknown_mode_returns_400`：未知注入类型返回 400

**test_approval_flow.py（14）** — HITL 审批
- `test_approve_flow` / `test_reject_flow` / `test_modify_flow` / `test_modify_flow_by_name_fallback`：批准 / 拒绝 / 修改（按 id / 按名称兜底）后恢复
- `test_tool_count_accumulates_across_approvals`：同轮多次审批工具数累计
- `test_plan_execute_approve_flow` / `test_plan_execute_modify_flow` / `test_plan_execute_modify_flow_by_name_fallback`：plan_execute（StateGraph 版）审批
- `test_resume_unknown_approval`：未知审批号 → 错误事件
- `test_command_tool_forced_hitl_even_when_never` / `test_command_tool_reject_flow` / `test_plan_execute_command_forced_hitl`：run_command 强制 HITL（react / plan_execute）
- `test_react_multi_forced_tools_batch_approval` / `test_react_multi_forced_tools_batch_reject`：一步内多个需审批工具批量审批 / 批量拒绝

### 3.2 前端（20 个用例，4 个文件）

**sse.test.ts（7）** — SSE 解析器
- 单条事件 / chunk 内多事件 / 跨 chunk 拼接 / `\r\n` 兼容 / 多 `data:` 行合并 / 非 data 行忽略 / HTTP 错误抛出状态码

**ModeSelector.test.ts（3）**
- 渲染 4 种模式 / 点击切换发出 `update:modelValue` / 当前选中高亮

**CapabilityCard.test.ts（5）**
- 可用能力渲染与开关 toggle / 启用状态 aria-checked / 不可用置灰「不适配」且开关禁用 / 「示例」按钮发出 example 事件 / 不可用不渲染示例按钮

**ApprovalDialog.test.ts（5）**
- 渲染待审批工具与参数 / 批准 / 拒绝 / 修改参数提交 / 非法 JSON 报错不提交

## 4. 测试结果

| 项目 | 用例数 | 结果 |
|---|---|---|
| 后端 pytest | 97（含 21 个重试/故障注入相关用例） | ✅ 94 通过；3 个 opensandbox 用例需安装 `opensandbox` SDK（环境缺失） |
| 前端 vitest | 20 | ✅ 全部通过（20 passed） |
| 前端构建 | — | ✅ `vite build` 成功 |

## 5. 冒烟验证（无真实 Key 环境）

- `GET /api/health` → 200，`{"status":"ok","model":"qwen-plus","mcp_configured":false,"embedding_configured":false}`
- `GET /api/capabilities` → 内置能力可用；rag/memory 因缺 Embedding Key 标记 `unavailable`（含原因）
- `GET /` → 200 返回前端构建页面
- `GET /api/source/react` → 200 返回后端真实源码
- `POST /api/stream`（未配 `LLM_API_KEY`）→ 500 且 `detail` 明确引导配置

## 5.1 真实 SSE 冒烟（配 DashScope Key 后，原生 SDK 流式）

- `POST /api/stream`（react + calculator，`approval_policy=never`）事件序列验证通过：
  `meta(1) → thinking(270 段真实思考流) → tool_start(calculator) → tool_end(result=32.15, success=true) → message(36 段最终输出) → done`
- 后端控制台按段打印 `[model-stream] reason: ...` 与 `[model-stream] output: ...`，确认思考与输出分流传给前端（思考 581 字符 / 输出 161 字符）。

## 6. 端到端验证清单（配真实 Key 后）

- [ ] 能力池：内置可用；未配 Embedding Key 时 rag/memory 显示「不适配」；配置的 MCP Server 工具自动出现，连接失败显示「不适配」
- [ ] 点击能力开关 → 热插拔即时生效；不可用能力开关禁用
- [ ] 点击示例按钮 → 示例 prompt 自动填入输入框且能力已启用 → 发送后真实调用该能力并流式展示
- [ ] 四种模式均可运行并可视化（ReAct 循环 / plan 分解 / reflection 修订 / multi-agent 分派）
- [ ] HITL 审批弹窗批准/拒绝/修改后继续；提示词策略切换生效
- [ ] 对比视图同任务多模式并排；代码片段展示真实源码；错误提示与重试可用
- [ ] 移动端窄屏正常、流式渲染无卡顿
- [ ] 生产：`npm run build` 后由 FastAPI 托管，`http://localhost:8000` 全流程可用
