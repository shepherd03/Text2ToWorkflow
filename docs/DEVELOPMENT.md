# 开发与运行手册

本文档面向日常开发、调试和评测。

## 环境准备

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

关键依赖：

- `pydantic`：Schema 与模型校验。
- `httpx`：LLM 与 embedding 请求。
- `fastapi`、`uvicorn`：API 服务。
- `networkx`：UTR 依赖评测。
- `PyYAML`：读取 Dify DSL 样本。
- `pytest`：测试。

## 环境变量

```env
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
SEMANTIC_BACKEND=tfidf
SEMANTIC_EMBEDDING_PROVIDER=openai-compatible
SEMANTIC_EMBEDDING_MODEL=embedding-default
SEMANTIC_EMBEDDING_API_KEY=
SEMANTIC_EMBEDDING_BASE_URL=https://api.openai.com/v1
SEMANTIC_EMBEDDING_CACHE_PATH=generated_data/semantic_cache/embeddings.json
UTR_STRICT_COMPLETENESS=false
```

推荐默认：

- 本地开发使用 `SEMANTIC_BACKEND=tfidf`。
- 需要比较语义检索能力时再使用 `embedding` 或 `hybrid`。
- 没有 API key 时仍应能运行大部分单元测试。

## 单条任务调试

```bash
python main.py --text "读取文章并生成摘要" --pretty
```

调试完整链路：

```bash
python main.py --text "读取文章并生成摘要" --stage dsl --pretty
```

`--stage` 支持：

- `utr`：只生成统一任务表示。
- `skeleton`：生成 UTR 与执行骨架。
- `dsl`：生成 UTR、Skeleton 与 Dify workflow payload。

启动 API：

```bash
uvicorn api:app --reload
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/utr/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"读取文章并生成摘要\"}"
```

端到端接口：

```bash
curl -X POST http://127.0.0.1:8000/workflow/build ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"读取文章并生成摘要\",\"stage\":\"dsl\"}"
```

## 主链路批处理

生成 UTR：

```bash
python scripts/01_generate_utrs.py
```

可选环境变量：

```env
UTR_SAMPLE_SIZE=50
UTR_OUTPUT_FILE=generated_data/utr_generation/utrs.jsonl
UTR_ERROR_FILE=generated_data/utr_generation/utr_errors.jsonl
```

规划 Skeleton：

```bash
python scripts/02_plan_skeletons.py
```

常用参数：

```bash
python scripts/02_plan_skeletons.py ^
  --input-file generated_data/utr_generation/utrs.jsonl ^
  --output-dir generated_data/skeleton_planning/iter2
```

编译 Dify workflow：

```bash
python scripts/03_compile_dify_workflows.py
```

常用参数：

```bash
python scripts/03_compile_dify_workflows.py ^
  --utr-file generated_data/utr_generation/utrs.jsonl ^
  --skeleton-file generated_data/skeleton_planning/iter2/skeletons.jsonl ^
  --output-file generated_data/dsl_generation/dsls.jsonl
```

第三阶段临时烟测建议写入系统临时目录，避免覆盖稳定产物：

```powershell
python scripts/03_compile_dify_workflows.py `
  --output-file "$env:TEMP\utr_smoke_dsls.jsonl" `
  --error-file "$env:TEMP\utr_smoke_errors.json"
```

当前验证基线：

- `python -m pytest tests/` 应全部通过。
- `scripts/03_compile_dify_workflows.py` 对 `generated_data/skeleton_planning/iter2/skeletons.jsonl` 的 31 条样本应输出 `Success: 31, Errors: 0`。
- 若修改 DSL 编译器或 validator，至少检查 selector 是否存在、控制流 join 是否绑定、if-else 出边 handle 是否为 `case_id`、`false` 或普通 `source`。

## 项目健康检查

轻量收集当前产物、评测摘要和链路状态：

```bash
python scripts/16_project_healthcheck.py
```

完整健康检查：

```powershell
python scripts/16_project_healthcheck.py --run-tests --run-smoke
```

输出：

```text
generated_data/project_health/latest.json
```

`quality_gates.passed` 为 `true` 时，本轮结果才可作为稳定基线。每次较大改动后，应把 healthcheck 结果和关键观察写入 `docs/EXPERIMENT_LOG.md`。

## UTR 评测

构建真值：

```bash
python scripts/04_build_utr_ground_truth.py
```

评测预测 UTR：

```bash
python scripts/05_evaluate_utrs.py
```

可选环境变量：

```env
UTR_GT_FILE=generated_data/utr_generation/utr_ground_truth.jsonl
UTR_PRED_FILE=generated_data/utr_generation/utrs.jsonl
```

指标：

- `M_pur`：元数据纯度。
- `M_dep`：依赖关系合理性。
- `M_comp`：实体提取完备性。
- `M_sch`：Schema 依从性。

## 节点映射评测

内部样本：

```bash
python scripts/07_prepare_node_mapping_eval_data.py
python scripts/08_evaluate_node_mapping_generalization.py
```

外部样本：

```bash
python scripts/11_prepare_dify_external_node_mapping_eval_data.py
python scripts/12_evaluate_dify_external_node_mapping.py
```

外部 Dify 数据集：

```bash
python scripts/13_build_dify_external_dataset.py
python scripts/14_prepare_dify_external_eval_from_dataset.py
python scripts/12_evaluate_dify_external_node_mapping.py
python scripts/15_analyze_dify_external_dataset.py
```

主要输出：

```text
generated_data/dsl_generation/node_mapping_eval/
generated_data/dsl_generation/dify_external_node_mapping_eval/
generated_data/dify_external_dataset/
```

降级指标口径：

- `degradation_accuracy`：降级样本同时满足节点类型正确和预测降级。
- `degradation_detection_accuracy`：降级样本中预测为降级的比例。
- `degradation_type_accuracy`：降级样本中节点类型仍映射正确的比例。

## 测试

运行全部测试：

```bash
python -m pytest tests/
```

按模块运行：

```bash
python -m pytest tests/test_utr_generator.py
python -m pytest tests/test_skeleton_planner.py
python -m pytest tests/test_dsl_input_validator.py
python -m pytest tests/test_dsl_compiler.py
python -m pytest tests/test_node_mapper.py
```

## 常见问题

`rg` 不可用时，Windows 环境可用 PowerShell：

```powershell
Get-ChildItem -Recurse -File | Select-String -Pattern "NodeMapper"
```

LLM 调用失败：

- 检查 `.env` 是否配置 `DEEPSEEK_API_KEY`。
- 只跑结构测试时可留空 API key。

节点映射结果过度降级：

- 查看 `NodeMappingResult.trace`。
- 检查 `required_params` 与 `available_params`。
- 为规则或语义检索新增测试样例。
