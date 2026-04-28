# UTR Workflow Compiler

本项目用于把自然语言任务或已有 Dify DSL 数据转化为可验证、可评测的工作流表示。当前主链路是：

```text
dataset/instruction -> UTR -> Skeleton -> Dify Workflow DSL -> Evaluation
```

它不是单个脚本项目，而是一个分阶段的工作流编译系统：前两阶段负责抽取任务语义与规划结构，第三阶段负责映射 Dify 节点并生成 workflow graph，评测链路负责量化 UTR 与节点映射质量。当前重点已推进到第三阶段 DSL 可落地性：业务节点输入会按 Skeleton 上游绑定，条件和并行通过 join 汇合，循环会生成 iteration 容器、iteration-start 内部起点和 loop join，Dify 节点数据结构按外部样本持续对齐。

## 项目结构

```text
api.py                         FastAPI 服务入口
main.py                        单条自然语言任务的 CLI 入口
dataset/                       原始样本，包含 instruction 与 Dify DSL
src/core/                      全局配置、LLM 客户端、Pydantic Schema
src/utr_generation/            自然语言 -> UTR
src/skeleton_planning/         UTR -> Skeleton
src/dsl_generation/            UTR + Skeleton -> Dify Workflow DSL
scripts/                       批处理、评测、外部数据集构造脚本
tests/                         单元测试与链路测试
generated_data/                批处理输出与评测产物
docs/                          稳定开发文档
```

## 快速开始

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

如需调用 LLM 生成真实 UTR，在 `.env` 中配置：

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

不配置 API key 时，UTR 生成会走测试 fallback，适合本地测试结构链路。

## 主链路命令

1. 生成 UTR：

```bash
python scripts/01_generate_utrs.py
```

默认输出：

```text
generated_data/utr_generation/utrs.jsonl
generated_data/utr_generation/run_<timestamp>/utrs.jsonl
```

2. 规划 Skeleton：

```bash
python scripts/02_plan_skeletons.py
```

默认读取 `generated_data/utr_generation/utrs.jsonl`，输出：

```text
generated_data/skeleton_planning/iter2/skeletons.jsonl
generated_data/skeleton_planning/iter2/errors.json
```

3. 编译 Dify Workflow：

```bash
python scripts/03_compile_dify_workflows.py
```

默认读取 UTR 与 Skeleton，输出：

```text
generated_data/dsl_generation/dsls.jsonl
generated_data/dsl_generation/errors.json
```

当前第三阶段验证基线：`generated_data/skeleton_planning/iter2/skeletons.jsonl` 中 31 条样本应编译为 `Success: 31, Errors: 0`。临时验证请把输出写入 `$env:TEMP`，避免覆盖稳定产物。

4. 构建 UTR 真值并评测：

```bash
python scripts/04_build_utr_ground_truth.py
python scripts/05_evaluate_utrs.py
```

5. 节点映射评测：

```bash
python scripts/07_prepare_node_mapping_eval_data.py
python scripts/08_evaluate_node_mapping_generalization.py
```

6. 外部 Dify 数据集链路：

```bash
python scripts/13_build_dify_external_dataset.py
python scripts/14_prepare_dify_external_eval_from_dataset.py
python scripts/12_evaluate_dify_external_node_mapping.py
python scripts/15_analyze_dify_external_dataset.py
```

7. 项目健康检查：

```bash
python scripts/16_project_healthcheck.py
```

如需同时运行测试和第三阶段 smoke：

```powershell
python scripts/16_project_healthcheck.py --run-tests --run-smoke
```

默认输出：

```text
generated_data/project_health/latest.json
```

健康检查会输出 `quality_gates`。只有 `quality_gates.passed=true` 时，本轮测试、smoke、产物和指标才作为稳定实验基线。

8. 真实 LLM 研究批处理：

```powershell
python scripts/17_run_llm_workflow_research_batch.py --max-records 6 --stage dsl
```

该脚本强制要求 `DEEPSEEK_API_KEY`，输出 manifest 会记录 `used_real_llm`、`llm_call_count`、模型和成功数。

9. 新增研究数据构造：

```powershell
python scripts/18_build_research_dataset_from_existing_dsl.py --max-records 8
```

该脚本从外部 Dify DSL 样本池中抽取新样本，并用真实 LLM 重新生成中文 instruction。

## 服务入口

启动 API：

```bash
uvicorn api:app --reload
```

接口：

```text
GET  /health
POST /utr/generate
POST /workflow/build
```

单条 CLI：

```bash
python main.py --text "读取文章并生成摘要" --pretty
```

单条端到端 DSL 编译：

```bash
python main.py --text "读取文章并生成摘要" --stage dsl --pretty
```

`--stage` 可选 `utr`、`skeleton`、`dsl`，用于调试单条任务在不同阶段的产物。

## 测试

```bash
python -m pytest tests/
```

测试覆盖 UTR 依赖归一化、Skeleton 拓扑规划、DSL 输入校验、结构归一化、Dify 编译、节点映射、语义检索与外部数据集工具。

涉及 `src/dsl_generation/` 的改动还应运行：

```powershell
python scripts/03_compile_dify_workflows.py `
  --output-file "$env:TEMP\utr_smoke_dsls.jsonl" `
  --error-file "$env:TEMP\utr_smoke_errors.json"
```

## 文档索引

- [架构与链路](docs/ARCHITECTURE.md)
- [模块开发指南](docs/MODULE_GUIDE.md)
- [开发与运行手册](docs/DEVELOPMENT.md)
- [项目规则](docs/PROJECT_RULES.md)
- [研究推进计划](docs/RESEARCH_PLAN.md)
- [实验记录](docs/EXPERIMENT_LOG.md)
- [Dify DSL 参考](docs/dify-dsl-spec-codex.md)

## 维护原则

项目默认采用 Schema-first 的分阶段设计。UTR 只描述任务元数据，Skeleton 只负责执行结构，DSL 模块只处理 Dify 适配与节点映射。阶段性汇报和一次性分析不要放入 `docs/` 根目录；可复现的评测结果放在 `generated_data/`，稳定结论沉淀到 `docs/`。
