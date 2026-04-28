# 实验记录

本文档只记录稳定实验结论和可复现实验命令。临时分析和一次性输出放在 `generated_data/`。

## 2026-04-29：端到端入口与健康基线

目标：

- 让单条自然语言任务能直接跑到 UTR、Skeleton 或 Dify DSL。
- 建立下一步自动健康检查的输入基线。

改动：

- 新增 `src/workflow_pipeline.py`，统一编排 `UTRGenerationPipeline -> SkeletonPlanner -> DSLGenerationPipeline`。
- `main.py` 增加 `--stage utr|skeleton|dsl`。
- `api.py` 增加 `POST /workflow/build`。
- `WorkflowBuildOutput` 统一承载 UTR、Skeleton、DSL、success、errors、warnings、meta。

验证：

```text
python -m pytest tests/ -q
147 passed
```

```text
python scripts/03_compile_dify_workflows.py --output-file "$env:TEMP\utr_smoke_e2e_dsls.jsonl" --error-file "$env:TEMP\utr_smoke_e2e_errors.json"
Success: 31, Errors: 0
```

观察：

- 无 API key fallback 能跑通端到端链路。
- CLI 完整 DSL 输出需要强制 UTF-8，避免 Windows GBK 控制台无法输出图标字符。
- 库层不应直接 `print` LLM 判断，否则会污染 CLI JSON 输出。

下一步假设：

- 如果建立 `scripts/16_project_healthcheck.py`，每轮迭代可以用统一 JSON 记录测试、smoke、产物和评测摘要，从而更像科研实验而不是手工检查。

## 2026-04-29：健康门禁与降级指标拆分

目标：

- 把健康检查从“收集信息”升级为“可判定基线是否有效”的质量门禁。
- 拆分节点映射的降级指标，避免把“识别需要降级”和“节点类型仍然正确”混在一个数里。

改动：

- `scripts/16_project_healthcheck.py` 增加 `quality_gates`、pytest 摘要解析和 Stage 3 smoke 摘要解析。
- `NodeMappingEvalMetrics` 增加 `degradation_detection_accuracy` 与 `degradation_type_accuracy`。
- 内部和外部节点映射评测统一输出严格降级准确率、降级识别准确率、降级类型准确率。

验证：

```text
python -m pytest tests/test_project_healthcheck.py tests/test_node_mapping_generalization_eval.py tests/test_dify_external_node_mapping_eval_data.py -q
22 passed
```

当前观察：

- 内部 tfidf 测试集 accuracy 变为 1.0，hard accuracy 为 0.9556，hard 降级三项指标均为 0.5。
- 外部 Dify 全量 587 个节点 accuracy 为 0.9830，macro F1 为 0.9729。
- 外部 4 个 expected degraded 样本中，降级识别准确率为 0.75，但类型准确率只有 0.25，严格降级准确率为 0.0，下一轮应集中分析这 4 个样本。

下一步假设：

- 先不要盲目提高 `degradation_accuracy`，应把外部样本的 expected degraded 标注规则和内部 hard set 口径对齐，再决定 mapper 是否需要改变。
