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

## 2026-04-29：可信度校准与真实 LLM 链路

目标：

- 让节点映射不只报告 accuracy，还报告置信度是否可信。
- 证明项目能够使用已配置的 DeepSeek API 跑真实 UTR/Skeleton/DSL 链路。
- 构造一批新的研究数据，避免只依赖旧基线。

改动：

- `NodeMappingEvalPrediction` 增加 `confidence_score` 与 `confidence_margin`。
- 内部、外部节点映射评测增加 `confidence_ece`、`confidence_brier`、`confidence_bucket_accuracy`。
- `UTRGenerationPipeline` 的 meta 记录 `generation_source`、`llm_call_count`、`llm_model` 和 `llm_usage`。
- 新增 `scripts/17_run_llm_workflow_research_batch.py`，强制使用真实 LLM 端到端生成。
- 新增 `scripts/18_build_research_dataset_from_existing_dsl.py`，从外部 DSL 样本池构造新的 LLM instruction 数据。

验证：

```text
python -m pytest tests/ -q
156 passed
```

```text
python scripts/03_compile_dify_workflows.py --output-file "$env:TEMP\utr_healthcheck_dsls.jsonl" --error-file "$env:TEMP\utr_healthcheck_errors.json"
Success: 31, Errors: 0
```

真实 LLM 链路：

```text
python scripts/17_run_llm_workflow_research_batch.py --max-records 6 --stage dsl
success_count=6, error_count=0, llm_call_count=21, used_real_llm=true
```

新增研究数据：

```text
python scripts/18_build_research_dataset_from_existing_dsl.py --max-records 8 --min-novel-ratio 0.10
workflow_count=8, error_count=0, llm_call_count=8, used_real_llm=true
```

当前观察：

- 内部 tfidf test accuracy 为 1.0，confidence ECE 为 0.0228，Brier 为 0.0031。
- 外部 587 节点 accuracy 为 0.9830，confidence ECE 为 0.0323，Brier 为 0.0187。
- 外部高置信桶 `0.85-1.00` 有 515 条，准确率 1.0，平均置信度 0.9900。
- 外部低置信桶 `0.00-0.50` 有 24 条，准确率 0.75，说明后续主动学习应优先采样低置信节点。
- 重联网搜索外部 DSL 的 `scripts/13_build_dify_external_dataset.py` 仍偏重，快速迭代应先用 `18` 脚本；后续需要把 `13` 拆成候选抓取和 LLM instruction 生成两个可断点阶段。

下一步假设：

- 以低置信桶和 degraded 样本为主动学习池，补 hard set，比盲目扩大普通样本更能提升可信度。
