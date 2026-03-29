# Dify Workflow DSL 规范

本文档汇总 `/Users/air/Downloads/DifyAIA` 仓库内全部 38 份 Workflow YAML（DSL）文件与官方文档 `/Users/air/Downloads/dify-docs/zh-hans/guides/workflow/` 所描述的规则，形成一份完整的 Dify 工作流 DSL 参考。适用于 Workflow 与 Chatflow 两类应用的离线建模、校验、调试和自定义生成。

## 1. 基础概念

- **Workflow / Chatflow**：Workflow 面向一次性自动化任务，Chatflow 面向多轮对话。两者共用 DSL 结构，差异体现在节点可用性（Answer、Memory 等）与系统变量。
- **节点 (Node)**：工作流的原子组件，通过连线组合执行逻辑。每个节点有唯一 `id`，以 `data.type` 标识语义类型。
- **变量 (Variable)**：节点间传递数据的途径，包括系统变量（`sys.*`）、环境变量（`env.*`）、会话变量（`conversation.*`）、以及节点输出引用（`<node_id>.<field>`）。
- **画布 (Graph)**：以 `workflow.graph` 保存节点、连线、画布坐标，保证可视化编辑与可重复执行。
- **DSL 版本**：当前样本均为 `version: 0.2.0`，`kind: app`，如版本升级需关注兼容性。

## 2. YAML 顶层结构

| 键 | 类型 | 说明 |
| --- | --- | --- |
| `app` | dict | 应用元数据（名称、描述、图标等）。
| `dependencies` | list | Marketplace 插件依赖列表。每项包含 `type: marketplace` 与 `value.marketplace_plugin_unique_identifier`。
| `model_config` | dict? | 仅部分文件存在，用于 Agent/Chatflow 额外配置（详见 3.3）。
| `workflow` | dict | 工作流主体配置，包括变量、功能开关、图结构。
| `kind` | str | 恒为 `app`。
| `version` | str | 当前为 `0.2.0`。

### 2.1 app

字段：`name`、`description`、`icon`、`icon_background`、`mode` (`workflow` 或 `chat`)、`use_icon_as_answer_icon`。

### 2.2 dependencies

每项：
```
- type: marketplace
  current_identifier: null
  value:
    marketplace_plugin_unique_identifier: <provider>/<plugin>:<version>@<sha>
```
用于声明工具、模型、Agent 等远端插件。

### 2.3 workflow

- `conversation_variables`：数组，定义对话/会话变量初始值。字段：`id`、`name`、`description`、`value`、`value_type`、可选 `selector`。
- `environment_variables`：数组，保护敏感信息。字段与会话变量一致，`value_type` 支持 `string`、`number`、`secret`。
- `features`：附加功能开关（见 6 章）。
- `graph`：节点、连线与视图内容。

## 3. 模型与 Agent 配置 (`model_config`)

`model_config` 可选，用于 Chatflow/Agent 应用的默认行为。

| 键 | 说明 |
| --- | --- |
| `model` | 默认模型描述：`mode`、`name`、`provider`、`completion_params`（温度、stop 词等）。
| `pre_prompt` / `opening_statement` | 应用级系统提示与欢迎语。
| `agent_mode` | 开启 Agent 策略，字段含 `enabled`、`strategy`（如 `react`、`function_calling`）、`max_iteration`、`tools` 数组（工具 ID、参数）。
| `dataset_configs` | 知识库检索策略，如 `retrieval_model`、`reranking_enable`、`top_k`。
| `file_upload` | Chatflow 附加文件上传设置（类型、大小限制、图片 detail 等）。
| `annotation_reply`、`more_like_this`、`sensitive_word_avoidance` | 其它附加能力开关。
| `user_input_form` | 自定义开场表单，各项以控件名称为键（如 `text-input`、`paragraph`、`select`），字段含 `label`、`default`、`required`、`max_length`、`options`、`variable`。

## 4. 变量与引用机制

### 4.1 变量类型

| 类型 | 前缀 | 写权限 | 说明 |
| --- | --- | --- | --- |
| 系统变量 | `sys.*` | 只读 | 由系统自动提供。Workflow: `sys.files` (legacy)、`sys.user_id`、`sys.app_id`、`sys.workflow_id`、`sys.workflow_run_id`；Chatflow 额外提供 `sys.query`、`sys.dialogue_count`、`sys.conversation_id`。|
| 环境变量 | `env.*` | 只读 | 保护密钥、API Key 等。需在应用配置中预设。|
| 会话变量 | `conversation.*` | 读写 | Chatflow 专用，通过变量赋值节点更新。数据类型支持 `string`、`number`、`boolean`、`object` 及对应数组。|
| 节点输出 | `<node_id>.<field>` | 只读 | 某节点执行结果。通过引用语法注入。|

### 4.2 引用语法

- **Mustache 占位符**：`{{#<selector>#}}`，用于模板文本或参数，例如 `{{#sys.query#}}`、`{{#1730959461085.text#}}`。
- **选择器数组**：`['scope','key']` 或 `['<node_id>','field']`。用于 `variables`、`tool_parameters`、`value_selector` 等结构化字段。
- 支持多重引用：聚合器节点使用 `[['node','field'], ...]` 表示多个来源。

### 4.3 变量管理

- 环境变量需唯一命名，只读引用。
- 会话变量只能由 Chatflow 使用，必须通过变量赋值节点写入。
- 调试时可通过变量检查器查看节点执行后的变量状态（官方调试文档 `debug-and-preview`）。

## 5. Graph 结构

### 5.1 节点对象

每个节点（`workflow.graph.nodes`）包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | str | 唯一标识，常为时间戳 + 随机串或语义名。
| `type` | str | UI 用途（样本为 `custom`）。
| `data` | dict | 语义配置，必须包含 `title`、`type` 等。
| `position` / `positionAbsolute` | dict | 画布坐标 (`x`,`y`)。
| `width` / `height` | number | 节点尺寸。
| `sourcePosition` / `targetPosition` | str | 连接点方位 (`left`/`right` 等)。
| `selected` | bool | 画布状态。
| 可选 `parentId` | str | 指向父节点（迭代体）。
| 可选 `extent` | str | 限制拖拽范围（如 `parent`）。
| 可选 `zIndex`、`selectable`、`draggable`。

`data` 字段中核心键：`type`（节点语义，详见第 7 章），`title`、`desc`、`selected`、`variables`（输入引用）、及特定节点配置。

### 5.2 连线对象

`workflow.graph.edges` 的字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | str | 唯一标识。
| `type` | str | 恒为 `custom`。
| `source` / `target` | str | 源节点/目标节点 ID。
| `sourceHandle` / `targetHandle` | str | 连接端口 ID。
| `selected` | bool | UI 状态。
| `data` | dict | 包含 `sourceType`、`targetType`、`isInIteration`、`isInLoop`、可选 `iteration_id`。
| `zIndex` | int | 层级。

### 5.3 迭代与父子节点

- `iteration` 节点通过 `parentId` 将子节点纳入迭代容器。
- `iteration-start` 节点标记迭代入口，`data.isInIteration: true`。
- `iteration.start_node_id` 指向入口节点 ID，`startNodeType` 指定入口类型（如 `code`）。

## 6. 附加功能 (`features`)

来自官方“附加功能”与 YAML 样本：

| 键 | 说明 | 典型字段 |
| --- | --- | --- |
| `file_upload` | Chatflow/Workflow 图片或多类型文件上传。Workflow 仅推荐自定义文件变量。字段包括 `enabled`、`allowed_file_extensions`、`allowed_file_types`、`allowed_file_upload_methods`、`number_limits`、`image.enabled/number_limits/transfer_methods`、`fileUploadConfig`（大小限制）。文件通过 `sys.files` 提供给后续节点。|
| `opening_statement` | 开场白文本（Chatflow 常用）。|
| `suggested_questions` | 数组，显示下一步建议。|
| `suggested_questions_after_answer` | `enabled` 布尔与 `items` 配置。|
| `retriever_resource` | 控制知识检索入口是否开启。|
| `sensitive_word_avoidance` | 敏感词过滤开关与策略。|
| `speech_to_text` / `text_to_speech` | 语音输入输出开关及语言、声音配置。|

官方建议：Workflow 应用若需文件上传，应在开始节点自定义 `file` 类型变量替代 legacy 功能。

## 7. 节点类型规范

以下按照 YAML 样本与 `guides/workflow/node/*.mdx` 汇总。字段列表包含必填与常见可选字段；若无特殊说明则字段类型为字符串或布尔。

### 7.1 start

- 用途：定义应用输入表单。Chatflow 默认包含 `sys.query`、`sys.files` 等系统变量；Workflow 可配置表单项。
- 字段：
  - `variables`: 表单字段数组。字段键：
    - `label`、`variable`、`type`（`text-input`、`paragraph`、`select`、`file` 等）、`required`、`max_length`、`options`（select 用）、可选 `allowed_file_extensions`、`allowed_file_types`、`allowed_file_upload_methods`。
  - `desc`、`title`、`selected`、`type: start`。

### 7.2 llm

- 用途：调用大语言模型。支持上下文、记忆、视觉、结构化输出。
- 数据字段：
  - `model`: `{ mode, name, provider, completion_params }`。`completion_params` 包含 `temperature`、`stop`、`max_tokens` 等。
  - `prompt_template`: 数组，元素含 `role` (`system`/`user`/`assistant`)、`text`、可选 `edition_type`、`id`。
  - `context`: `{ enabled, variable_selector }` 控制上下文引用。
  - `memory`: `{ window.size, window.enabled, query_prompt_template, role_prefix }`（Chatflow 支持）。
  - `vision`: `{ enabled, configs.detail, configs.variable_selector }`。（若模型支持图像输入）。
  - `variables`: 节点输入变量数组，结构与 start 类似或 `value_selector` 形式。
  - 迭代相关：`isInIteration`、`iteration_id`、`isIterationStart`。
  - 可选 `structured_output_enabled`、`output_schema`（JSON Schema 定义，结合结构化输出功能）。

### 7.3 answer（Chatflow 专用）

- 用途：在流程任意位置返回回答，支持流式输出。
- 字段：`answer`（模板字符串或引用）、`variables`（少见）、`title`、`desc`。

### 7.4 end（Workflow 专用）

- 用途：终止流程并返回最终变量。
- 字段：`outputs`: 数组，每项 `variable`（返回名）+ `value_selector`（源）；`title`、`desc`。

### 7.5 code

- 用途：执行自定义脚本（主要为 Python）。
- 字段：
  - `code_language`: 示例为 `python3`。
  - `code`: 脚本内容。
  - `variables`: 输入变量映射，数组项 `{ variable, value_selector }`。
  - `outputs`: 映射输出名称到类型定义 `{ type, children }`。
  - 迭代标记：`isInIteration`、`iteration_id`、`isIterationStart`（若在循环内）。

### 7.6 tool

- 用途：调用内建或 Marketplace 工具。
- 字段：
  - `provider_id`、`provider_name`、`provider_type`、`tool_name`、`tool_label`、`tool_description`。
  - `tool_parameters`: 参数字典，键为参数名，值 `{ type: mixed, value: <模板> }`。
  - `paramSchemas`: 参数元数据数组，每项含 `name`、`label`（多语言）、`type`、`form`、`required`、`human_description`、`options`、`min`/`max` 等。
  - `tool_configurations`: 额外配置。
  - `is_team_authorization`（布尔）决定团队授权。
  - 可选 `retry_config` `{ retry_enabled, max_retries, retry_interval }`。

### 7.7 http-request

- 用途：发起 HTTP API 调用。
- 字段：
  - `method`（`get`、`post` 等）、`url`、`headers`（字符串，支持模板）、`params`（查询字符串）、`body`（对象，含 `type`=`json`/`raw-text`/`none`、`data`）。
  - `authorization`: `{ type: 'no-auth'|'api-key', config }`。
  - `timeout`: `{ max_connect_timeout, max_read_timeout, max_write_timeout }`。
  - `variables`: 输入变量数组。
  - 支持迭代标记。

### 7.8 doc-extractor（Document Extractor）

- 用途：解析上传文档，输出纯文本或结构化内容。
- 字段：`variable_selector`（通常 `['sys','files']`）、`is_array_file`（是否批量）。

### 7.9 template-transform

- 用途：根据模板拼接文本。
- 字段：`template`（字符串，可含 Mustache）、`variables`（所需变量选择器数组）。

### 7.10 assigner（Variable Assigner）

- 用途：写入会话或对话变量。
- 字段：
  - `items`: 数组，每项包含：
    - `input_type`（样本为 `variable`）。
    - `operation`: `over-write`。
    - `value`: 选择器数组（数据来源）。
    - `variable_selector`: 目标变量（如 `['conversation','language']`）。
    - `write_mode`: `over-write`。
  - `version`: `'2'`（样本）。

### 7.11 variable-aggregator

- 用途：聚合多个变量为单一输出。
- 字段：`output_type`（`string`、`array[string]` 等）、`variables`（二维数组，每项 `[node_id, field]`）。

### 7.12 list-operator

- 用途：对列表数据过滤、排序、限制。常用于文件列表。
- 字段：
  - `variable`: 数据来源选择器。
  - `var_type`: 列表类型（如 `array[file]`）。
  - `item_var_type`: 单项类型。
  - `filter_by`: `{ enabled, conditions }`，每个条件含 `comparison_operator` (`in`/`contains`)、`key`、`value`（数组或字符串）。
  - `order_by`: `{ enabled, key, value }`。
  - `limit`: `{ enabled, size }`。

### 7.13 if-else

- 用途：基于条件分支。
- 字段：
  - `cases`: 数组，每项含 `case_id`、`logical_operator` (`or`/`and`)、`conditions`。
  - 条件字段：`variable_selector`、`comparison_operator`（`empty`、`not empty`、`=`、`>`、`contains`、`not contains`、`in`、`exists`）、`value`（字符串或列表）、`varType`（`string`、`number`、`file`、`array[file]`）。

### 7.14 iteration

- 用途：对数组或文件集合进行循环，支持并行。
- 字段：
  - `iterator_selector`：循环输入来源。
  - `iterator_input_type` / `is_array_input`: 标明是否数组输入。
  - `input_parameters`: 可选列表，在进入循环体前注入变量（每项含 `variable`、`value_selector`）。
  - `start_node_id`: 迭代起始节点（通常 `iteration-start`）。
  - `startNodeType`: 可显式声明起始节点类型（如 `code`）。
  - `output_selector`、`output_type`: 定义聚合输出。
  - `is_parallel`: 是否并行，`parallel_nums` 控制并发度。
  - `error_handle_mode`: 错误策略（样本为 `terminated`）。

### 7.15 iteration-start

- 作为迭代容器入口。字段仅包含 `type: iteration-start`、`isInIteration: true`、`title`、`desc`。

### 7.16 parameter-extractor

- 用途：利用 LLM 从文本提取结构化参数。
- 字段：
  - `model` 与 `completion_params`。
  - `query`: 选择器（文本来源）。
  - `parameters`: 数组，定义 `name`、`type`、`description`、`required`。
  - `instruction`/`instructions`: 指令文本。
  - `reasoning_mode`: `prompt`（样本）。

### 7.17 question-classifier

- 用途：多分支意图分类。
- 字段：
  - `model`、`query_variable_selector`。
  - `classes`: 数组，含 `id`、`name`。
  - `topics`: 可选数组。
  - `instructions`（可空）。
  - `vision`: 结构同 LLM。

### 7.18 knowledge-retrieval

- 用途：接入 Dify 知识库检索。
- 字段：`dataset_ids`、`retrieval_mode` (`single`/`multiple`)、`multiple_retrieval_config`（`top_k`、`score_threshold`、`reranking_enable`、`reranking_mode`、`reranking_model`）、`query_variable_selector`。

### 7.19 agent

- 用途：调用 Agent 策略，实现多步自主工具调用。与官方文档一致。
- 字段：
  - `agent_strategy_name`/`label`、`agent_strategy_provider_name`、`plugin_unique_identifier`。
  - `agent_parameters`: 字典，键对应策略定义的参数（如 `instruction`、`model`、`tools`、`query`），每项 `{ type: constant|variable, value }`。
  - `output_schema`: 可选（样本为 `null`）。
  - 支持 `memory`、`desc`、`title` 等通用字段。

### 7.20 variable-assigner（文档中“变量赋值”节点）

- DSL 中同 `assigner`，参照 7.10。

### 7.21 loop / 其它

- 文档中提到的 Loop 节点与目前样本迭代实现一致（`iteration` + `iteration-start`）。如未来引入新的 `loop` 类型，应参照官方节点文档补充。

### 7.22 注释节点（`data.type` 为空字符串）

- 用途：画布说明，不参与执行。
- 字段：`author`、`text`（Lexical JSON 字符串）、`theme`、`showAuthor`、`width`、`height`。

## 8. 错误处理与重试

- 节点可通过 `retry_config` 启用失败重试（样本在 Tool 节点中出现）。配置项：`retry_enabled`、`max_retries`、`retry_interval`（毫秒）。
- 复杂流程可结合 `if-else`、`error_handle_mode`（迭代节点）与调试面板进行异常分支处理（参考官方 `error-handling` 文档）。

## 9. 文件上传与文档解析

- Workflow 官方建议通过开始节点添加自定义 `file` 类型变量，以替代 legacy `features.file_upload`。
- 上传文件根据类型不同需配合 `document-extractor`、`list-operator`、`vision` 功能等节点。
- `sys.files` 在每轮对话更新，若需长期记忆需写入会话变量或外部存储。

## 10. 结构化输出

- LLM 节点可通过启用 `structured_output_enabled` 并配置 `output_schema` 实现 JSON Schema 约束。
- Tool 插件亦可在自身 Schema 中声明返回结构（参阅官方 “结构化输出” 文档中的方式一）。
- 处理失败建议：
  1. 选用支持 JSON Schema 的模型（如 Gemini Flash、GPT-4o、o 系列）。
  2. 强化系统提示词以匹配 Schema。
  3. 配置重试或异常分支。

## 11. 调试与发布建议

- 工作流应通过调试器检查变量、节点输出（参见 `debug-and-preview` 文档）。
- 发布前确认 `features` 与 `environment_variables` 配置完整，必要时设置默认值。
- Chatflow 应注意会话变量写入逻辑，避免循环内重复覆盖。

## 12. 命名与兼容约束

- 节点 `id` 在同一工作流内不可重复，推荐语义化或时间戳命名。
- 变量命名应避免冲突，特别是系统变量与自定义变量。
- 依赖项需保证 Marketplace 插件已安装且版本匹配。
- DSL 当前基于 0.2.0 版本格式，如引入新的节点类型或字段，需同步更新此规范。

## 13. 常见模式总结

- **文件解析流程**：`start` (file input) → `document-extractor` → `llm` → `template-transform` → `answer/end`。
- **Agent 工具链**：`question-classifier` → `agent` (function_calling 或 ReAct) → `tool/llm` → `answer/end`。
- **批量迭代**：`start` (array input) → `iteration` + 内部 `code/llm/tool` → `variable-aggregator` → `template-transform` → `end`。
- **变量记忆**：`if-else` 判断会话变量 → `llm` 提取 → `assigner` 写入 → 后续节点引用。

## 14. 参考资料

- Dify 官方文档：`/zh-hans/guides/workflow/` 下的 `key-concept.mdx`、`variables.mdx`、`node/*.mdx`、`structured-outputs.mdx`、`additional-feature.mdx`、`orchestrate-node.mdx` 等。
- 样本 DSL：`/Users/air/Downloads/DifyAIA` 中的 38 个 `.yml` 工作流文件。

以上规范可直接用于 DSL 校验、脚本生成或代码审查，确保离线编辑与前端画布行为保持一致。
