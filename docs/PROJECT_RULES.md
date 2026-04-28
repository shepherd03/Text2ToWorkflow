# 项目规则

本文档定义项目长期维护规则。新增代码、脚本和文档都应遵守这些约束。

## 1. Schema-first

跨模块交互必须以 `src/core/schema.py` 中的 Pydantic 模型为契约。

允许：

- 在 Schema 中新增稳定字段。
- 用 `model_validate` 解析外部输入。
- 在报告模型中保留 errors、warnings、issues。

不允许：

- 在脚本中临时拼接跨阶段私有结构。
- 让下游模块依赖上游输出中的非 Schema 字段。

## 2. 阶段职责不可混用

UTR 阶段：

- 只抽取任务目标、动作、资源、变量、潜在依赖。
- 不判断并行、条件、循环。
- 不输出 Dify 节点类型。

Skeleton 阶段：

- 负责顺序、并行、条件、循环等执行结构。
- 不决定 Dify 节点类型。
- 不生成 Dify workflow payload。

DSL 阶段：

- 负责 Dify 节点映射、graph 编译和 workflow 校验。
- 可以做 Dify 特定降级，但必须记录 `degraded` 与原因。

评测阶段：

- 只消费已有产物。
- 不反向修改生成逻辑。

## 3. 脚本规则

脚本编号含义：

```text
01-03  主生成链路
04-05  UTR 真值与评测
07-12  节点映射评测
13-15  外部 Dify 数据集
16     项目健康检查
```

规则：

- 新脚本必须有明确输入、输出和默认路径。
- 批处理脚本默认写入 `generated_data/`。
- 一次性报告生成脚本不放在 `scripts/` 根目录。
- 脚本不应在 import 时执行批处理逻辑。
- 可被测试复用的逻辑应拆成函数。
- 科研推进脚本必须输出机器可读 JSON，方便追踪实验变化。

## 4. 文档规则

`docs/` 只保留稳定文档：

```text
ARCHITECTURE.md
MODULE_GUIDE.md
DEVELOPMENT.md
PROJECT_RULES.md
dify-dsl-spec-codex.md
```

规则：

- 阶段性汇报、周报、一次性分析不进入 `docs/` 根目录。
- 稳定结论应合并进架构、开发或规则文档。
- 每次改变链路、脚本编号、默认输出路径，都要更新 README 和相关 docs。

## 5. generated_data 规则

`generated_data/` 是实验产物区。

规则：

- 固定产物路径可以被脚本默认读取。
- 时间戳 run 目录只作追溯，不作为代码依赖。
- 大型缓存和临时分析不应进入代码逻辑。
- 评测摘要可以保留，阶段性解释应沉淀到文档。

## 6. LLM 使用规则

LLM 是辅助能力，不是结构正确性的唯一来源。

规则：

- 没有 API key 时，测试链路必须可运行。
- 研究批处理必须记录 `generation_source`、`llm_call_count`、`llm_model` 或等价 manifest 字段。
- 声称“真实生成”的实验必须有 `used_real_llm=true` 且 `llm_call_count > 0`。
- LLM 输出必须经过 Schema 或 JSON 解析。
- UTR 依赖边必须归一化。
- Skeleton 控制流注入失败时应降级为基础动作节点。
- 评测脚本使用 LLM-as-a-Judge 时必须有 fallback 或明确失败信息。

## 7. 节点映射规则

NodeMapper 必须输出可解释结果。

必须保留：

- `candidate_node_types`
- `candidate_scores`
- `chosen_score`
- `runner_up_score`
- `confidence`
- `confidence_score`
- `confidence_margin`
- `trace`
- `degraded`
- `degrade_reason`

新增节点映射能力时：

- 更新 `DifyNodeType`。
- 更新 `node_mapping_rules.py`。
- 更新 `NodeMapper` 参数覆盖和降级策略。
- 更新 `DifyWorkflowValidator`。
- 新增或更新测试。

DSL 编译规则：

- 所有 `value_selector` 必须指向真实存在的节点输出、start 变量、系统变量或环境变量。
- 所有 `{{#node.field#}}` 模板引用也必须纳入 selector 校验，不能只检查结构化 `value_selector` 字段。
- 业务节点输入优先来自上游动作输出；即使输入名同名存在于 start 变量中，也应优先使用上游输出。只有无法解析时才退回到已有 start 变量或 `sys.query`。
- 如果输入名是明确的 start 变量，而上游动作没有同名或语义匹配输出，不能因为上游只有单个输出就强行绑定到上游。
- 条件、并行、循环之后必须通过 join 节点汇合，不能让后续动作隐式绑定到某个分支末端。
- if-else 分支出边必须使用对应 `case_id` 或 `false`；普通顺序出边使用 `source`。
- LoopBlock 必须落为 Dify 风格的 `iteration` 容器加内部 `iteration-start` 起点，循环体节点和内部边必须标记 `isInIteration`、`iteration_id`。
- 循环体消费迭代对象时应绑定 `iteration_x.item`，循环结束后的后续动作只能通过 `iteration_x.output` 或 `loop_join.result` 继续读取。
- 调整 Dify 节点字段形状时，应参考 `generated_data/dify_external_dataset/` 中的外部样本，并补充编译测试。

## 8. 测试规则

提交前至少运行：

```bash
python -m pytest tests/
```

大改动后运行：

```bash
python scripts/16_project_healthcheck.py --run-tests --run-smoke
```

`quality_gates.passed` 必须为 `true`，否则不得把本轮结果写成稳定实验结论。

最小测试要求：

- UTR 变更跑 `tests/test_utr_generator.py`。
- Skeleton 变更跑 `tests/test_skeleton_planner.py`。
- DSL 变更跑 `tests/test_dsl_input_validator.py` 与 `tests/test_dsl_compiler.py`。
- 节点映射变更跑 `tests/test_node_mapper.py` 与节点映射评测工具测试。
- 修改节点映射可信度后，必须重跑 `scripts/08_evaluate_node_mapping_generalization.py` 和 `scripts/12_evaluate_dify_external_node_mapping.py`，并检查 `confidence_ece`。

## 9. 命名规则

文件命名：

- Python 文件使用 `snake_case.py`。
- 文档使用英文大写主题名或稳定中文名，不使用日期作为主要文件名。
- 评测产物文件可以带语义后缀，不在 docs 中堆叠日期报告。

模型命名：

- 数据模型使用名词，例如 `UTR`、`DSLCompileOutput`。
- 处理器使用职责后缀，例如 `Generator`、`Planner`、`Validator`、`Compiler`。

## 10. 变更纪律

开发时优先保持小步提交和清晰边界。

规则：

- 不把无关重构混进功能变更。
- 不直接改历史评测产物来“修分数”。
- 不删除仍被测试引用的脚本。
- 删除旧文档时，需要有新的稳定文档承接有效内容。
