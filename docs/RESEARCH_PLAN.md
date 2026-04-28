# 研究推进计划

本文档把项目定义为一个可持续实验系统，而不是一次性工程交付。

## 研究目标

核心问题：

```text
自然语言任务能否稳定转化为可导入、可验证、可评测的 Dify Workflow DSL？
```

当前主假设：

- UTR 只表达任务语义，能降低下游控制流规划的噪声。
- Skeleton 用确定性拓扑为主、LLM 判断为辅，能提升结构可控性。
- DSL 编译器用规则映射、语义检索和严格 validator 组合，能比纯 LLM 生成更稳定。
- 外部 Dify 样本可以作为节点形状和映射泛化能力的主要参照。

## 研究问题

R1：UTR 抽取质量如何影响 Skeleton 和 DSL 的最终成功率？

R2：动作到 Dify 节点类型的映射，规则、语义检索、混合检索各自的边界在哪里？

R3：控制流结构中，条件、并行、循环的最小可导入 DSL 形态是什么？

R4：项目内 validator 能否提前发现 Dify 导入失败风险？

R5：外部 DSL 样本能否持续转化为高质量评测集，反向推动节点映射和字段形状改进？

## 实验闭环

每轮推进按以下顺序执行：

1. 选择一个明确链路问题，例如 selector 绑定、循环输出、节点映射混淆。
2. 写下假设和预期指标变化。
3. 修改最小必要代码。
4. 补充单元测试或链路测试。
5. 运行健康检查和 smoke。
6. 记录结果、失败样本和下一轮假设。

## 基线指标

必须长期跟踪：

- 全量测试通过数。
- `scripts/03_compile_dify_workflows.py` 的成功数和错误数。
- 节点映射 accuracy、macro F1、degradation accuracy。
- 降级样本要拆分看三类指标：`degradation_accuracy` 表示类型正确且识别降级，`degradation_detection_accuracy` 表示是否识别需要降级，`degradation_type_accuracy` 表示原始节点类型是否映射正确。
- 外部 Dify 样本节点类型覆盖数。
- validator 捕获的 selector、edge、node payload 错误数。

当前基础门槛：

```text
python -m pytest tests/
python scripts/03_compile_dify_workflows.py --output-file $TEMP --error-file $TEMP
```

第三阶段 smoke 基线：

```text
31 条 Skeleton 样本编译成功，Errors: 0
```

健康检查基线：

```powershell
python scripts/16_project_healthcheck.py --run-tests --run-smoke
```

`quality_gates.passed` 必须为 `true` 才能把本轮结果当作稳定实验基线。

## 失败样本规则

- 不直接删除失败样本。
- 失败先归因到 UTR、Skeleton、DSL mapping、payload shape、validator 或数据集噪声。
- 可复现失败要沉淀为测试。
- 修复后保留错误归因，作为后续论文或报告材料。

## 近期路线

P0：端到端入口稳定化。

- CLI 支持 `utr`、`skeleton`、`dsl` 三阶段输出。
- API 支持 `/workflow/build`。
- 无 API key fallback 能跑通完整链路。

P1：健康检查自动化。

- 用单个脚本汇总测试、smoke、产物存在性和评测摘要。
- 输出机器可读 JSON，作为每轮实验记录。
- 用 `quality_gates` 明确当前结果是否达到基础门槛，避免半截产物或退化指标被误记为结论。

P2：节点映射泛化能力。

- 扩大外部 Dify 样本覆盖。
- 对混淆矩阵中高频误判建立专门 hard set。
- 比较 rule、tfidf、hybrid、embedding backend。

P3：DSL 可导入性。

- 对照真实 Dify DSL 字段形状继续收敛 HTTP、tool、iteration、parameter-extractor、variable-aggregator。
- 增强 validator 对模板引用、分支 handle、iteration 内部结构的检查。

P4：论文材料沉淀。

- 把每轮基线、消融和失败案例转化为稳定实验表。
- 建立“方法、指标、结果、威胁”四段式实验记录。
