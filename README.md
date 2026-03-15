
## 5. 评估与验证

本项目提供了基于 Dify DSL 的自动化评估套件，用于验证 UTR 生成的质量。

### 5.1 评估原理
1. **Ground Truth (GT)**: 将 Dify 的 YAML 工作流文件转换为标准 UTR 格式，作为标准答案。
2. **Generation (Gen)**: 使用 UTR 生成流水线（基于 LLM）处理 Dify 工作流的描述，生成预测 UTR。
3. **Scoring**: 对比 GT 和 Gen，计算动作匹配度、参数完整性、逻辑准确率等指标。

### 5.2 核心组件
- `dify_dsl_converter.py`: Dify DSL -> Standard UTR 转换器。
- `utr_evaluator.py`: UTR 评分器，支持模糊匹配和语义打分。
- `batch_evaluator.py`: 批量评估入口，支持自动遍历 `dify_workflows` 目录。

### 5.3 运行评估
```powershell
# 运行批量评估（默认评估前 10 个样本）
.\venv\Scripts\python.exe utr_generator/batch_evaluator.py
```
评估结果将生成在 `evaluation_results/batch_evaluation_summary.csv` 和 `evaluation_results/summary_report.md`。

### 5.4 最新评估结果 (2026-03-15)
- **样本数**: 10
- **平均分**: 38.97
- **亮点**: 简单任务（如翻译、代码执行）动作匹配度可达 100%。
- **待改进**: 复杂工作流的粒度对齐和参数提取。
