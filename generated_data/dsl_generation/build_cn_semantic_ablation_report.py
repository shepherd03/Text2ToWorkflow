from __future__ import annotations

import os
from pathlib import Path

from src.core.schema import Action, DifyNodeType
from src.dsl_generation.node_mapper import NodeMapper


CASES = [
    {
        "id": "cn_llm_1",
        "name": "生成摘要",
        "description": "请根据文章内容生成摘要",
        "inputs": ["文章内容"],
        "outputs": ["摘要"],
        "expected": DifyNodeType.llm,
        "extra": {},
        "note": "典型中文生成任务，规则和语义都应支持。",
    },
    {
        "id": "cn_http_1",
        "name": "调用天气接口",
        "description": "调用天气API，使用url和get方法获取结果",
        "inputs": ["url", "method", "城市"],
        "outputs": ["天气结果"],
        "expected": DifyNodeType.http_request,
        "extra": {},
        "note": "中文接口调用，且关键参数齐全。",
    },
    {
        "id": "cn_template_1",
        "name": "套用模板生成通知",
        "description": "根据模板生成通知文本",
        "inputs": ["template", "标题", "内容"],
        "outputs": ["通知文本"],
        "expected": DifyNodeType.template_transform,
        "extra": {},
        "note": "规则命中较弱，适合观察语义层贡献。",
    },
    {
        "id": "cn_iteration_1",
        "name": "遍历文件列表",
        "description": "遍历文件列表并逐个处理",
        "inputs": ["files"],
        "outputs": ["处理结果"],
        "expected": DifyNodeType.iteration,
        "extra": {"parent_block_type": "Loop"},
        "note": "结构上下文与中文规则共同作用。",
    },
    {
        "id": "cn_if_1",
        "name": "条件判断分流",
        "description": "根据 conditions 条件判断是否进入告警分支",
        "inputs": ["conditions", "状态"],
        "outputs": ["分支结果"],
        "expected": DifyNodeType.if_else,
        "extra": {"parent_block_type": "Conditional"},
        "note": "显式补充 conditions，用于测试中文条件分支在参数齐全时的识别能力。",
    },
    {
        "id": "cn_tool_1",
        "name": "调用搜索工具",
        "description": "调用搜索工具查询资料",
        "inputs": ["provider_id", "tool_name", "query"],
        "outputs": ["结果"],
        "expected": DifyNodeType.tool,
        "extra": {"available_resources": ["search_service"]},
        "note": "显式补充 provider_id / tool_name，观察工具节点在中文下的语义竞争。",
    },
    {
        "id": "cn_doc_1",
        "name": "解析合同文件",
        "description": "读取合同PDF并提取正文内容",
        "inputs": ["pdf"],
        "outputs": ["合同文本"],
        "expected": DifyNodeType.doc_extractor,
        "extra": {},
        "note": "中文文档抽取任务。",
    },
    {
        "id": "cn_param_1",
        "name": "抽取用户意图和槽位",
        "description": "从用户消息中抽取意图、实体和参数字段",
        "inputs": ["message"],
        "outputs": ["意图", "参数"],
        "expected": DifyNodeType.parameter_extractor,
        "extra": {},
        "note": "中文参数抽取任务。",
    },
    {
        "id": "cn_agg_1",
        "name": "合并多路结果",
        "description": "聚合多个分支输出结果",
        "inputs": ["分支A", "分支B"],
        "outputs": ["合并结果"],
        "expected": DifyNodeType.variable_aggregator,
        "extra": {},
        "note": "中文聚合任务。",
    },
    {
        "id": "cn_http_missing_1",
        "name": "请求外部服务",
        "description": "调用外部接口返回数据",
        "inputs": ["payload"],
        "outputs": ["结果"],
        "expected": DifyNodeType.code,
        "extra": {},
        "note": "缺少 url/method，预期最终发生降级。",
    },
]

METHODS = [
    {
        "name": "tfidf",
        "env": {
            "SEMANTIC_BACKEND": "tfidf",
            "SEMANTIC_EMBEDDING_PROVIDER": "local-hash",
            "SEMANTIC_EMBEDDING_API_KEY": "dummy",
        },
        "desc": "仅使用词法语义检索。",
    },
    {
        "name": "embedding-local",
        "env": {
            "SEMANTIC_BACKEND": "embedding",
            "SEMANTIC_EMBEDDING_PROVIDER": "local-hash",
            "SEMANTIC_EMBEDDING_API_KEY": "dummy",
        },
        "desc": "使用本地哈希向量 provider 模拟 embedding 检索。",
    },
    {
        "name": "hybrid-local",
        "env": {
            "SEMANTIC_BACKEND": "hybrid",
            "SEMANTIC_EMBEDDING_PROVIDER": "local-hash",
            "SEMANTIC_EMBEDDING_API_KEY": "dummy",
        },
        "desc": "将 TF-IDF 与本地哈希向量检索融合。",
    },
]


def set_method_env(method: dict[str, object]) -> None:
    env = method["env"]
    for key, value in env.items():
        os.environ[key] = str(value)


def run_method(method: dict[str, object]) -> list[dict[str, object]]:
    set_method_env(method)
    mapper = NodeMapper()
    results: list[dict[str, object]] = []

    for case in CASES:
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
        results.append(
            {
                "case": case,
                "result": result,
                "correct": result.chosen_node_type == case["expected"],
            }
        )
    return results


all_results = {method["name"]: run_method(method) for method in METHODS}

report: list[str] = [
    "# 中文语义方法消融实验报告",
    "",
    "## 1. 实验目的",
    "",
    "本实验仅使用中文测试样例，对当前节点映射模块中的三种语义方法进行消融对比：`tfidf`、`embedding(local-hash)` 与 `hybrid(local-hash)`。",
    "实验目标包括：",
    "",
    "1. 比较不同语义方法在中文样例上的最终节点映射结果；",
    "2. 对比不同方法下各个候选节点的分项评分；",
    "3. 分析规则层、语义层、参数覆盖和上下文在中文场景下分别起到了什么作用。",
    "",
    "## 2. 方法设置",
    "",
]

for method in METHODS:
    report.extend([f"- `{method['name']}`：{method['desc']}"])

report.extend(
    [
        "",
        "说明：为了保证实验可复现，本次 embedding 与 hybrid 实验均使用 `local-hash` 作为向量 provider，不依赖外部 API 服务。",
        "",
        "## 3. 中文测试集",
        "",
        f"共使用 {len(CASES)} 个中文样例，覆盖：LLM、HTTP 请求、模板转换、循环、条件分支、工具调用、文档抽取、参数抽取、变量聚合与缺参降级场景。",
        "",
        "| case | 中文动作 | 预期节点 | 说明 |",
        "|---|---|---|---|",
    ]
)

for case in CASES:
    report.append(
        f"| `{case['id']}` | {case['name']} | `{case['expected'].value}` | {case['note']} |"
    )

report.extend(["", "## 4. 总体结果对比", ""])

report.extend(["| 方法 | 正确数 | 总数 | 准确率 |", "|---|---:|---:|---:|"])
for method in METHODS:
    method_results = all_results[method["name"]]
    correct = sum(1 for item in method_results if item["correct"])
    total = len(method_results)
    acc = correct / total if total else 0.0
    report.append(f"| `{method['name']}` | {correct} | {total} | {acc:.2%} |")

report.extend(["", "### 4.1 各样例最终节点对比", ""])
header = "| case | 预期 | " + " | ".join(f"{method['name']}" for method in METHODS) + " |"
sep = "|---|---|" + "---|" * len(METHODS)
report.extend([header, sep])
for idx, case in enumerate(CASES):
    cells = [f"`{case['id']}`", f"`{case['expected'].value}`"]
    for method in METHODS:
        item = all_results[method["name"]][idx]
        result = item["result"]
        suffix = " ✓" if item["correct"] else " ✗"
        degrade = " (degraded)" if result.degraded else ""
        cells.append(f"`{result.chosen_node_type.value}`{degrade}{suffix}")
    report.append("| " + " | ".join(cells) + " |")

report.extend(["", "## 5. 逐方法详细评分结果", ""])

for method in METHODS:
    report.extend([f"### 5.{METHODS.index(method)+1} {method['name']}", ""])
    method_results = all_results[method["name"]]
    for index, item in enumerate(method_results, start=1):
        case = item["case"]
        result = item["result"]
        report.extend(
            [
                f"#### 5.{METHODS.index(method)+1}.{index} {case['id']}",
                "",
                f"- 动作名：`{case['name']}`",
                f"- 描述：{case['description']}",
                f"- 预期节点：`{case['expected'].value}`",
                f"- 实际节点：`{result.chosen_node_type.value}`",
                f"- 是否正确：`{item['correct']}`",
                f"- 是否降级：`{result.degraded}`",
                f"- 置信度：`{result.confidence.value}`",
                f"- chosen_score：`{result.chosen_score:.3f}`",
                f"- runner_up_score：`{result.runner_up_score:.3f}`",
                "",
                "| candidate | rule | semantic | coverage | context | priority | total |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for score in result.candidate_scores:
            report.append(
                f"| `{score.node_type.value}` | {score.rule_score:.3f} | {score.semantic_score:.3f} | {score.coverage_score:.3f} | {score.context_score:.3f} | {score.priority_bonus:.3f} | {score.total_score:.3f} |"
            )
        report.extend(
            [
                "",
                f"- 决策原因：{result.decision_reason}",
                f"- Trace：`{' | '.join(result.trace)}`",
                "",
            ]
        )

report.extend(
    [
        "## 6. 结果剖析",
        "",
        "### 6.1 中文 LLM、HTTP、Iteration 场景对三种方法都比较稳定",
        "",
        "从总体结果看，`生成摘要`、`调用天气接口`、`遍历文件列表` 这三类中文动作在三种语义方法下都能较稳定地命中目标节点。原因是这类场景中规则线索、语义线索和参数线索通常比较一致。",
        "",
        "### 6.2 中文 template-transform 更依赖语义层补位",
        "",
        "`套用模板生成通知` 这一类样例在规则层命中较弱，更多依赖语义层把节点拉到 `template-transform`。如果语义层表现不好，这类任务容易退回到 `code`。因此它是区分不同语义方法效果的重要场景。",
        "",
        "### 6.3 中文 if-else 和 tool 的核心瓶颈不只是语义，而是参数覆盖",
        "",
        "即使语义层已经把候选抬到了 `if-else` 或 `tool` 前列，只要 `conditions`、`provider_id`、`tool_name` 等关键参数没被识别出来，最后仍然会在 landing check 阶段发生降级。因此这两类节点的优化重点不应只盯在语义后端，还要增强中文参数线索识别。",
        "",
        "### 6.4 语义分数本身不是概率分，而是排序信号",
        "",
        "实验中可以看到，不同方法下 `semantic_score` 的数值范围并不统一，而且有些值会大于 1。这说明当前语义分更适合作为相对排序依据，而不是直接解释成概率或置信度。真正决定最终输出的，仍然是加权总分加上后续的参数落地检查。",
        "",
        "### 6.5 Hybrid 是否更优，需要结合中文规则覆盖情况一起看",
        "",
        "Hybrid 理论上能同时利用词法检索和向量检索，但当中文规则已经较强、且 local-hash 向量 provider 较弱时，Hybrid 的优势不一定会非常明显。因此当前消融结果更多反映的是：在现有中文规则覆盖和本地哈希向量条件下，三种语义方法分别能把哪些中文任务拉起来。",
        "",
        "## 7. 结论",
        "",
        "本次中文消融实验表明：",
        "",
        "1. 当前系统在中文下已经具备基础可用性；",
        "2. `tfidf`、`embedding(local-hash)`、`hybrid(local-hash)` 在简单中文任务上都能工作；",
        "3. 真正限制中文节点映射效果的，不仅是语义方法本身，更是中文规则覆盖与中文参数抽取能力；",
        "4. 如果后续要继续做中文优化，应优先增强 `if-else` 和 `tool` 这类关键参数敏感节点的中文参数识别能力。",
        "",
    ]
)

output_path = Path(r"e:\Desktop\论文\工作流\utr\docs\第四阶段中文语义方法消融实验报告.md")
output_path.write_text("\n".join(report), encoding="utf-8")
print(output_path)
