# 模块开发指南

本文档说明每个模块的职责、输入输出、扩展点和测试要求。

## core

路径：

```text
src/core/
```

关键文件：

- `schema.py`：所有阶段共享的 Pydantic 数据模型。
- `config.py`：环境变量读取与运行配置。
- `llm_client.py`：DeepSeek/OpenAI-compatible chat JSON 客户端。
- `utils.py`：JSON 提取、去重、JSONL 写入等工具。

开发规则：

- 新增跨模块数据结构必须先在 `schema.py` 定义。
- Schema 字段应有稳定语义，避免为了某个脚本临时加字段。
- 配置项必须进入 `.env.example`，并在 `Settings` 中提供默认值。

测试要求：

- Schema 变动需要补充或更新相关模块测试。
- LLM 相关逻辑必须能在无 API key 的本地测试中降级或隔离。

## utr_generation

路径：

```text
src/utr_generation/
```

关键文件：

- `pipeline.py`：UTR 生成统一入口。
- `utr_core.py`：LLM prompt、UTR 构造、依赖归一化。

输入：

- 自然语言任务描述。

输出：

- `PipelineOutput`
- 其中包含 `UTR`、`UTRValidationReport`、`meta`

开发规则：

- UTR 只表达元数据，不表达执行控制流。
- `implicit_dependencies` 必须只引用存在的 action_id。
- 自循环、重复边、悬空引用必须在进入 Skeleton 前过滤。

测试要求：

- 修改依赖处理逻辑时更新 `tests/test_utr_generator.py`。
- Prompt 调整后应重新跑 UTR 批处理和 UTR 评测。

## skeleton_planning

路径：

```text
src/skeleton_planning/
```

关键文件：

- `skeleton_planner.py`

输入：

- `UTR`

输出：

- `SequentialBlock`

开发规则：

- 拓扑排序是确定性主路径。
- 并行来自同层无依赖动作。
- 条件和循环属于 Skeleton 结构，但应有明确依据。
- 环形依赖必须报错。

测试要求：

- 修改拓扑或控制流逻辑时更新 `tests/test_skeleton_planner.py`。
- 需要覆盖顺序、并行、条件、循环、环检测。

## dsl_generation

路径：

```text
src/dsl_generation/
```

关键文件：

- `pipeline.py`：DSL 生成流水线。
- `validators.py`：输入合法性校验。
- `normalizer.py`：结构归一化。
- `compiler.py`：Dify workflow 编译器。
- `node_mapper.py`：动作到 Dify 节点类型的映射器。
- `node_mapping_rules.py`：规则词典与节点优先级。
- `semantic_retriever.py`：TF-IDF、embedding、hybrid 语义检索。
- `workflow_validator.py`：Dify workflow 输出校验。

输入：

- `UTR`
- `SequentialBlock`

输出：

- `DSLCompileOutput`
- Dify workflow payload

开发规则：

- Validator 只做校验，不做修复。
- Normalizer 可以做结构补全，例如默认 else 分支。
- Compiler 只消费归一化结果，不直接读取原始文件。
- Compiler 必须按归一化后的上下游槽位绑定输入 selector；不能把业务节点输入全部硬编码到 `start`。
- 控制流 join 是条件、并行、循环之后继续执行的唯一汇合点，后续动作和 end 节点应绑定 join 输出。
- 循环块必须生成 `iteration` 容器和内部 `iteration-start` 起点；循环体节点、循环体内部边必须带 `isInIteration` 与 `iteration_id`。
- 循环体首步如果消费的就是迭代对象，应绑定 `iteration_x.item`；循环结束后通过 `iteration_x.output -> loop_join.result` 进入后续链路。
- 生成 Dify 节点数据时优先贴近外部 Dify DSL 样本，例如 HTTP timeout/authorization、tool provider 元信息、variable-aggregator selector 结构。
- WorkflowValidator 必须检查 selector 来源节点、输出字段、edge 端点和 if-else 分支 handle；`value_selector` 字段和 `{{#node.field#}}` 模板引用都要纳入校验。
- NodeMapper 的每个决策必须可追踪，保留 scoring 与 trace。
- 不支持的 Dify 能力应降级到 `code`，并标记 `degraded`。

测试要求：

- 输入校验：`tests/test_dsl_input_validator.py`
- 编译输出：`tests/test_dsl_compiler.py`
- 节点映射：`tests/test_node_mapper.py`
- 语义检索：`tests/test_semantic_retriever.py`

## workflow_pipeline

路径：

```text
src/workflow_pipeline.py
```

职责：

- 编排单条自然语言任务的端到端链路。
- 按 `utr`、`skeleton`、`dsl` 三个阶段返回中间产物。
- 为 `main.py --stage` 和 `POST /workflow/build` 提供统一入口。

开发规则：

- 只负责阶段编排，不把 UTR、Skeleton 或 DSL 的内部逻辑写进该模块。
- 新增端到端输出字段时先更新 `WorkflowBuildOutput`。
- 无 API key 的 fallback 场景必须能跑通到 DSL，保证本地调试可用。

测试要求：

- 修改编排行为时更新 `tests/test_workflow_pipeline.py`。

## scripts

路径：

```text
scripts/
```

脚本分组：

- `01` 到 `03`：主链路生成。
- `04` 到 `05`：UTR 真值与评测。
- `07` 到 `12`：节点映射样本和评测。
- `13` 到 `15`：外部 Dify 数据集构造、转评测、分析。
- `16`：项目健康检查与科研推进基线。

开发规则：

- 脚本必须是可重复运行的批处理入口。
- 默认输入输出必须写在 README 或本文件中。
- 新增脚本必须说明属于哪条链路。
- 一次性报告生成脚本不要放在 `scripts/` 根目录。
- 每轮大改动至少运行 `16_project_healthcheck.py`；涉及 DSL 编译器时加 `--run-smoke`，发布前加 `--run-tests`。

## tests

路径：

```text
tests/
```

开发规则：

- 测试应优先覆盖阶段边界，而不是只测内部实现细节。
- 需要外部网络或真实 API key 的行为必须可跳过或使用 fallback。
- 新增映射规则时至少补一个正例和一个冲突/降级例。

## dataset 与 generated_data

`dataset/` 存放可复用输入样本。当前核心文件是：

```text
dataset/dataset.jsonl
```

`generated_data/` 存放批处理输出和实验结果。开发时可以生成新结果，但稳定代码不应依赖某个临时 run 目录。需要复用的结果应使用固定路径，例如：

```text
generated_data/utr_generation/utrs.jsonl
generated_data/skeleton_planning/iter2/skeletons.jsonl
generated_data/dsl_generation/dsls.jsonl
```
