from pathlib import Path

from src.core.schema import Action
from src.dsl_generation.node_mapper import NodeMapper


en_cases = [
    {
        "id": "llm",
        "name": "generate_summary",
        "description": "generate a concise summary for the article",
        "inputs": ["article_text"],
        "outputs": ["summary_text"],
        "extra": {},
    },
    {
        "id": "http_ok",
        "name": "request_weather_api",
        "description": "call weather api with get method and url",
        "inputs": ["url", "method", "city"],
        "outputs": ["weather_result"],
        "extra": {},
    },
    {
        "id": "http_missing",
        "name": "request_external_service",
        "description": "call external api and return data",
        "inputs": ["payload"],
        "outputs": ["result"],
        "extra": {},
    },
    {
        "id": "template",
        "name": "format_report_template",
        "description": "format a report with a template",
        "inputs": ["title", "content", "template"],
        "outputs": ["formatted_text"],
        "extra": {},
    },
    {
        "id": "iteration",
        "name": "iterate_files",
        "description": "iterate files one by one",
        "inputs": ["files"],
        "outputs": ["processed_files"],
        "extra": {"parent_block_type": "Loop"},
    },
    {
        "id": "if_else",
        "name": "check_condition",
        "description": "check a condition and branch",
        "inputs": ["status"],
        "outputs": ["decision"],
        "extra": {"parent_block_type": "Conditional"},
    },
    {
        "id": "tool",
        "name": "invoke_search_tool",
        "description": "execute search tool with query",
        "inputs": ["query"],
        "outputs": ["result"],
        "extra": {"available_resources": ["search_service"]},
    },
]

cn_cases = [
    {
        "id": "中文_llm",
        "name": "生成摘要",
        "description": "请根据文章内容生成摘要",
        "inputs": ["文章内容"],
        "outputs": ["摘要"],
        "extra": {},
    },
    {
        "id": "中文_http",
        "name": "调用天气接口",
        "description": "调用天气API，使用url和get方法获取结果",
        "inputs": ["url", "method", "城市"],
        "outputs": ["天气结果"],
        "extra": {},
    },
    {
        "id": "中文_template",
        "name": "套用模板生成通知",
        "description": "根据模板生成通知文本",
        "inputs": ["template", "标题", "内容"],
        "outputs": ["通知文本"],
        "extra": {},
    },
    {
        "id": "中文_iteration",
        "name": "遍历文件列表",
        "description": "遍历文件列表并逐个处理",
        "inputs": ["files"],
        "outputs": ["处理结果"],
        "extra": {"parent_block_type": "Loop"},
    },
    {
        "id": "中文_if",
        "name": "条件判断分流",
        "description": "如果状态异常则进入告警分支",
        "inputs": ["状态"],
        "outputs": ["分支结果"],
        "extra": {"parent_block_type": "Conditional"},
    },
    {
        "id": "中文_tool",
        "name": "调用搜索工具",
        "description": "调用搜索工具查询资料",
        "inputs": ["query"],
        "outputs": ["结果"],
        "extra": {"available_resources": ["search_service"]},
    },
    {
        "id": "中文_http_missing",
        "name": "请求外部服务",
        "description": "调用外部接口返回数据",
        "inputs": ["payload"],
        "outputs": ["结果"],
        "extra": {},
    },
]

mapper = NodeMapper()
report = [
    "# 第四阶段节点匹配评分实验报告",
    "",
    "## 1. 实验目的",
    "",
    "本实验用于验证当前节点映射模块中各分项得分是否已经实际参与计算，并对典型动作样例给出详细分数拆解。",
    "",
    "## 2. 当前评分公式",
    "",
    "当前实现中的总分为：",
    "",
    "\\[",
    "S = 0.40\\cdot rule + 0.32\\cdot semantic + 0.20\\cdot coverage + 0.05\\cdot context + 0.03\\cdot priority",
    "\\]",
    "",
    "其中：",
    "",
    "- `rule`：规则召回得分；",
    "- `semantic`：语义检索得分；",
    "- `coverage`：参数覆盖得分；",
    "- `context`：上下文得分；",
    "- `priority`：优先级微调项。",
    "",
    "## 3. 英文典型实验样例",
    "",
    "本组实验选取 7 个英文动作，覆盖 LLM、HTTP 请求、模板转换、循环、条件分支、工具调用，以及缺参降级场景。",
    "",
]


def append_case_block(report_lines, idx, case, result):
    report_lines.extend(
        [
            f"### {idx} {case['id']}",
            "",
            f"- 动作名：`{case['name']}`",
            f"- 描述：{case['description']}",
            f"- 最终节点：`{result.chosen_node_type.value}`",
            f"- 是否降级：`{result.degraded}`",
            f"- 置信度：`{result.confidence.value}`",
            "",
            "| candidate | rule | semantic | coverage | context | priority | total |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for score in result.candidate_scores:
        report_lines.append(
            f"| `{score.node_type.value}` | {score.rule_score:.3f} | {score.semantic_score:.3f} | {score.coverage_score:.3f} | {score.context_score:.3f} | {score.priority_bonus:.3f} | {score.total_score:.3f} |"
        )
    report_lines.extend(
        [
            "",
            f"- 决策原因：{result.decision_reason}",
            f"- Trace：`{' | '.join(result.trace)}`",
            "",
        ]
    )


for idx, case in enumerate(en_cases, start=1):
    action = Action(
        action_id=case["id"],
        action_name=case["name"],
        description=case["description"],
        inputs=case["inputs"],
        outputs=case["outputs"],
    )
    extra = case["extra"]
    result = mapper.map_action(
        action,
        parent_block_type=extra.get("parent_block_type", "Sequential"),
        upstream_actions=extra.get("upstream_actions", []),
        downstream_actions=extra.get("downstream_actions", []),
        available_variables=extra.get("available_variables", []),
        available_resources=extra.get("available_resources", []),
    )
    append_case_block(report, f"3.{idx}", case, result)

report.extend(
    [
        "## 4. 中文典型实验样例",
        "",
        "本组实验选取 7 个中文动作，用于观察当前规则集、语义检索和参数抽取在中文输入下的表现。",
        "",
    ]
)

for idx, case in enumerate(cn_cases, start=1):
    action = Action(
        action_id=case["id"],
        action_name=case["name"],
        description=case["description"],
        inputs=case["inputs"],
        outputs=case["outputs"],
    )
    extra = case["extra"]
    result = mapper.map_action(
        action,
        parent_block_type=extra.get("parent_block_type", "Sequential"),
        upstream_actions=extra.get("upstream_actions", []),
        downstream_actions=extra.get("downstream_actions", []),
        available_variables=extra.get("available_variables", []),
        available_resources=extra.get("available_resources", []),
    )
    append_case_block(report, f"4.{idx}", case, result)

report.extend(
    [
        "## 5. 实验观察",
        "",
        "### 5.1 分项得分已经实际参与计算",
        "",
        "从代码和实验结果可以确认：规则、语义、参数覆盖、上下文和优先级五项分数都已经进入总分计算，而不是只存在于设计层面。",
        "",
        "### 5.2 当前主导项仍然是规则与语义",
        "",
        "从权重上看，规则和语义共占 0.72，因此当前系统仍然主要依赖“规则召回 + 语义检索”进行候选排序。",
        "",
        "### 5.3 参数覆盖是排序项，也是落地约束",
        "",
        "参数覆盖不仅影响总分，还会进一步影响 landing check。以 `http_missing` 和 `中文_http_missing` 为例，`http-request` 的总分最高，但因缺少关键参数，最终仍会降级到 `code`。",
        "",
        "### 5.4 上下文分当前属于轻量修正项",
        "",
        "当前上下文权重只有 0.05，能够帮助 `Loop -> iteration`、`Conditional -> if-else` 这类结构化场景，但还不能算强上下文模型。",
        "",
        "### 5.5 语义分当前未做 0 到 1 归一化",
        "",
        "实验中多次出现 semantic 分数大于 1 的情况，例如英文 `tool` 样例中达到 1.987，中文 `template-transform` 样例中达到 1.492。这说明当前语义分更适合作为相对排序信号，而不应直接解释为概率或百分制分数。",
        "",
        "### 5.6 中文测试结果说明当前系统具备基础中文能力，但参数抽取仍偏弱",
        "",
        "从中文样例看，`llm`、`http-request`、`template-transform`、`iteration` 等类型已经可以在中文描述下被识别出来；但 `if-else` 和 `tool` 在中文场景下更容易因为关键参数识别不足而触发降级。这说明当前中文规则和中文参数线索识别还可以继续增强。",
        "",
        "## 6. 结论",
        "",
        "当前节点映射模块内部已经具备完整的分项评分机制，只是此前没有结构化暴露到 `NodeMappingResult` 中。现在代码已补充分数字段，可以直接用于后续实验记录、可解释性展示和报告撰写；同时中文测试表明，当前系统已具有基础中文可用性，但中文参数抽取和中文规则覆盖仍有进一步优化空间。",
        "",
    ]
)

output_path = Path(r"e:\Desktop\论文\工作流\utr\docs\第四阶段节点匹配评分实验报告.md")
output_path.write_text("\n".join(report), encoding="utf-8")
print(output_path)
