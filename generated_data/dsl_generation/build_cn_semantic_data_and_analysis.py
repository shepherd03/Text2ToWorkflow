from __future__ import annotations
import json, os, random
from collections import defaultdict
from pathlib import Path
from src.core.schema import Action, DifyNodeType
from src.dsl_generation.node_mapper import NodeMapper

SEED = 20260420
N = 30
METHODS = [
    ("tfidf", {"SEMANTIC_BACKEND": "tfidf", "SEMANTIC_EMBEDDING_PROVIDER": "local-hash", "SEMANTIC_EMBEDDING_API_KEY": "dummy"}, "仅使用词法语义检索"),
    ("embedding-local", {"SEMANTIC_BACKEND": "embedding", "SEMANTIC_EMBEDDING_PROVIDER": "local-hash", "SEMANTIC_EMBEDDING_API_KEY": "dummy"}, "使用本地哈希向量 provider 模拟 embedding 检索"),
    ("hybrid-local", {"SEMANTIC_BACKEND": "hybrid", "SEMANTIC_EMBEDDING_PROVIDER": "local-hash", "SEMANTIC_EMBEDDING_API_KEY": "dummy"}, "将 TF-IDF 与本地哈希向量检索融合"),
]
SPECS = {
    "llm": (DifyNodeType.llm, ["生成摘要", "改写说明", "撰写回复"], ["请根据{obj}生成摘要", "把{obj}改写成更清晰的说明", "根据{obj}撰写回复"], ["文章内容", "会议纪要", "用户反馈"], [["文章内容"], ["草稿文本"]], [["摘要"], ["回复文本"]], {}, "中文生成任务"),
    "http-request": (DifyNodeType.http_request, ["调用天气接口", "请求订单服务", "同步库存接口"], ["调用{obj}API，使用url和get方法获取结果", "向{obj}接口发送post请求，并带上url和method", "请求{obj}外部服务，使用url、method和payload返回数据"], ["天气", "订单", "库存"], [["url", "method", "payload"], ["url", "method", "城市"]], [["接口结果"], ["服务响应"]], {}, "中文接口调用任务"),
    "template-transform": (DifyNodeType.template_transform, ["套用模板生成通知", "渲染报告模板", "填充邮件模板"], ["根据template把{obj}生成固定格式通知", "使用模板渲染{obj}，输出结构化文本", "把{obj}填入template并形成最终消息"], ["活动信息", "报告数据", "邮件正文"], [["template", "标题", "内容"], ["template", "报告数据"]], [["通知文本"], ["渲染文本"]], {}, "中文模板转换任务"),
    "iteration": (DifyNodeType.iteration, ["遍历文件列表", "逐条处理记录", "批量处理消息"], ["遍历{obj}并逐个处理", "对{obj}中的每一项循环执行步骤", "批量处理{obj}，每条记录生成结果"], ["文件列表", "记录列表", "消息列表"], [["files"], ["records"], ["items"]], [["处理结果"], ["批量结果"]], {"parent_block_type": "Loop"}, "中文循环任务"),
    "if-else": (DifyNodeType.if_else, ["条件判断分流", "按风险等级路由", "检查权限条件"], ["根据conditions判断{obj}是否进入告警分支", "检查{obj}，当条件满足时进入高风险分支", "根据条件规则对{obj}进行分支选择"], ["订单", "用户", "任务"], [["conditions", "状态"], ["conditions", "风险等级"]], [["分支结果"], ["路由结果"]], {"parent_block_type": "Conditional"}, "中文条件分支任务，显式包含 conditions"),
    "tool": (DifyNodeType.tool, ["调用搜索工具", "执行日历工具", "使用邮件工具"], ["调用{obj}工具查询资料", "使用{obj}工具执行外部能力", "运行{obj}插件并返回结果"], ["搜索", "日历", "邮件"], [["provider_id", "tool_name", "query"], ["provider_id", "tool_name", "参数"]], [["工具结果"], ["执行结果"]], {"available_resources": ["search_service", "calendar_tool"]}, "中文工具调用任务，显式包含 provider_id/tool_name"),
    "doc-extractor": (DifyNodeType.doc_extractor, ["解析合同文件", "读取发票PDF", "提取简历内容"], ["读取{obj}PDF并提取正文内容", "从{obj}文件中解析文本和关键字段", "扫描{obj}附件，抽取可读文本"], ["合同", "发票", "简历"], [["pdf"], ["file"], ["document"]], [["文档文本"], ["解析结果"]], {}, "中文文档抽取任务"),
    "parameter-extractor": (DifyNodeType.parameter_extractor, ["抽取用户意图和槽位", "识别实体参数", "标注关键词类别"], ["从{obj}中抽取意图、实体和参数字段", "识别{obj}里的关键词、类别和槽位", "解析{obj}并提取参数信息"], ["用户消息", "对话文本", "客服记录"], [["message"], ["文本"]], [["意图", "参数"], ["实体", "标签"]], {}, "中文参数抽取任务"),
    "variable-aggregator": (DifyNodeType.variable_aggregator, ["合并多路结果", "聚合分支输出", "汇总指标数据"], ["聚合多个分支输出结果", "合并{obj}并形成统一输出", "收集多个{obj}并汇总成总结果"], ["分支结果", "指标", "响应"], [["分支A", "分支B"], ["结果1", "结果2"]], [["合并结果"], ["汇总结果"]], {}, "中文变量聚合任务"),
    "degrade-case": (DifyNodeType.code, ["请求外部服务", "调用外部接口", "触发远端服务"], ["调用外部接口返回数据", "请求远端服务并处理响应", "触发第三方接口获取信息"], ["服务", "接口", "网关"], [["payload"], ["请求体"], ["参数"]], [["结果"], ["响应"]], {}, "缺少 url/method，预期发生降级"),
}
ORDER = list(SPECS)

def build_cases():
    r = random.Random(SEED); out = []
    for i in range(N):
        cat = ORDER[i % len(ORDER)]
        exp, names, descs, objs, ins, outs, extra, note = SPECS[cat]
        obj = r.choice(objs)
        out.append({
            "id": f"cn_rand_{i+1:02d}_{cat.replace('-', '_')}", "name": r.choice(names), "description": r.choice(descs).format(obj=obj),
            "inputs": list(r.choice(ins)), "outputs": list(r.choice(outs)), "expected": exp, "extra": dict(extra), "category": cat,
            "note": note, "generation": {"seed": SEED, "index": i + 1, "object": obj, "category": cat},
        })
    return out

def run_method(method, cases):
    name, env, _ = method
    for k, v in env.items(): os.environ[k] = v
    m = NodeMapper(); rows = []
    for c in cases:
        a = Action(action_id=c["id"], action_name=c["name"], description=c["description"], inputs=c["inputs"], outputs=c["outputs"])
        e = c["extra"]
        r = m.map_action(a, parent_block_type=e.get("parent_block_type", "Sequential"), upstream_actions=e.get("upstream_actions", []), downstream_actions=e.get("downstream_actions", []), available_variables=e.get("available_variables", []), available_resources=e.get("available_resources", []))
        rows.append({"case_id": c["id"], "case_name": c["name"], "category": c["category"], "description": c["description"], "inputs": c["inputs"], "outputs": c["outputs"], "expected_node": c["expected"].value, "predicted_node": r.chosen_node_type.value, "correct": r.chosen_node_type == c["expected"], "degraded": r.degraded, "confidence": r.confidence.value, "chosen_score": r.chosen_score, "runner_up_score": r.runner_up_score, "score_margin": r.chosen_score - r.runner_up_score, "decision_reason": r.decision_reason, "trace": r.trace, "candidate_scores": [x.model_dump(mode='json') for x in r.candidate_scores]})
    return name, rows

def summary(name, rows):
    return {"method": name, "correct": sum(1 for x in rows if x["correct"]), "total": len(rows), "accuracy": sum(1 for x in rows if x["correct"]) / len(rows), "degraded_count": sum(1 for x in rows if x["degraded"]), "high_confidence_count": sum(1 for x in rows if x["confidence"] == 'high'), "medium_confidence_count": sum(1 for x in rows if x["confidence"] == 'medium'), "low_confidence_count": sum(1 for x in rows if x["confidence"] == 'low'), "avg_chosen_score": sum(x["chosen_score"] for x in rows) / len(rows), "avg_margin": sum(x["score_margin"] for x in rows) / len(rows)}

cases = build_cases()
results = dict(run_method(m, cases) for m in METHODS)
method_summary = [summary(name, results[name]) for name, _, _ in METHODS]
category_matrix = defaultdict(dict)
for cat in sorted({c['category'] for c in cases}):
    for name, _, _ in METHODS:
        rs = [x for x in results[name] if x['category'] == cat]
        category_matrix[cat][name] = {"correct": sum(1 for x in rs if x['correct']), "total": len(rs), "degraded": sum(1 for x in rs if x['degraded'])}
disagreement_cases = []
for i, c in enumerate(cases):
    preds = {name: results[name][i]['predicted_node'] for name, _, _ in METHODS}
    corrs = {name: results[name][i]['correct'] for name, _, _ in METHODS}
    if len(set(preds.values())) > 1 or len(set(corrs.values())) > 1:
        disagreement_cases.append({"case_id": c['id'], "case_name": c['name'], "category": c['category'], "expected": c['expected'].value, "predictions": preds, "correctness": corrs})

payload = {"experiment": "cn_semantic_ablation_random_30", "description": "中文语义方法消融实验结构化数据。样例由固定随机种子合成，报告只给出数据和字段说明，不附加固定评价结论。", "random_seed": SEED, "case_count": len(cases), "methods": [{"name": n, "env": e, "description": d} for n, e, d in METHODS], "cases": [{"id": c['id'], "name": c['name'], "category": c['category'], "description": c['description'], "inputs": c['inputs'], "outputs": c['outputs'], "expected": c['expected'].value, "note": c['note'], "generation": c['generation']} for c in cases], "method_summary": method_summary, "category_matrix": category_matrix, "disagreement_cases": disagreement_cases, "results": results}
json_path = Path(r"e:\Desktop\论文\工作流\utr\generated_data\dsl_generation\cn_semantic_ablation_random30_data.json")
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

lines = ["# 中文语义方法消融实验数据说明报告", "", "## 1. 数据生成说明", "", f"- 数据文件：`{json_path.name}`", f"- 样例数量：{len(cases)}`, 随机种子：`{SEED}`", "- 样例由固定随机种子从中文任务模板中合成生成。", "", "## 2. 对比方法说明", "", "| 方法 | 说明 |", "|---|---|"]
for n, _, d in METHODS: lines.append(f"| `{n}` | {d} |")
lines += ["", "## 3. 测试样例清单", "", "| case_id | 类别 | 动作名 | 描述 | 输入 | 预期节点 |", "|---|---|---|---|---|---|"]
for c in cases: lines.append(f"| `{c['id']}` | `{c['category']}` | {c['name']} | {c['description']} | `{', '.join(c['inputs'])}` | `{c['expected'].value}` |")
lines += ["", "## 4. 方法总体数据", "", "| 方法 | 正确数 | 总数 | 准确率 | 降级数 | high | medium | low | 平均 chosen_score | 平均领先幅度 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for s in method_summary: lines.append(f"| `{s['method']}` | {s['correct']} | {s['total']} | {s['accuracy']:.2%} | {s['degraded_count']} | {s['high_confidence_count']} | {s['medium_confidence_count']} | {s['low_confidence_count']} | {s['avg_chosen_score']:.3f} | {s['avg_margin']:.3f} |")
lines += ["", "## 5. 类别维度数据", "", "每个单元格格式为：`正确数/总数，降级数=x`。", "", "| 类别 | tfidf | embedding-local | hybrid-local |", "|---|---|---|---|"]
for cat in sorted(category_matrix):
    vals = [f"`{cat}`"]
    for n, _, _ in METHODS:
        v = category_matrix[cat][n]; vals.append(f"{v['correct']}/{v['total']}，降级数={v['degraded']}")
    lines.append("| " + " | ".join(vals) + " |")
lines += ["", "## 6. 各方法逐样例预测结果", "", "| case_id | 预期节点 | tfidf | embedding-local | hybrid-local |", "|---|---|---|---|---|"]
for i, c in enumerate(cases):
    vals = [f"`{c['id']}`", f"`{c['expected'].value}`"]
    for n, _, _ in METHODS:
        r = results[n][i]; tag = 'correct' if r['correct'] else 'wrong'; deg = '; degraded' if r['degraded'] else ''
        vals.append(f"`{r['predicted_node']}` ({tag}{deg})")
    lines.append("| " + " | ".join(vals) + " |")
lines += ["", "## 7. 方法间不一致样例", ""]
if disagreement_cases:
    lines += ["| case_id | 类别 | 预期节点 | tfidf | embedding-local | hybrid-local |", "|---|---|---|---|---|---|"]
    for c in disagreement_cases:
        vals = [f"`{c['case_id']}`", f"`{c['category']}`", f"`{c['expected']}`"]
        for n, _, _ in METHODS:
            vals.append(f"`{c['predictions'][n]}` ({'correct' if c['correctness'][n] else 'wrong'})")
        lines.append("| " + " | ".join(vals) + " |")
else:
    lines.append("无方法间预测不一致样例。")
lines += ["", "## 8. 评分字段说明", "", "结构化数据中的 `candidate_scores` 字段记录每个候选节点的分项得分：", "", "- `rule_score`：规则召回得分；", "- `semantic_score`：语义检索得分；", "- `coverage_score`：参数覆盖得分；", "- `context_score`：上下文得分；", "- `priority_bonus`：节点优先级微调项；", "- `total_score`：加权后的候选总分。", "", "总分公式为：", "", "\\[", "S = 0.40\\cdot rule + 0.32\\cdot semantic + 0.20\\cdot coverage + 0.05\\cdot context + 0.03\\cdot priority", "\\]", "", "## 9. 数据文件结构说明", "", "JSON 顶层字段包括：`methods`、`cases`、`method_summary`、`category_matrix`、`disagreement_cases`、`results`。", "", "本报告仅提供数据、表格和字段说明，不包含固定评价结论。", ""]
report_path = Path(r"e:\Desktop\论文\工作流\utr\docs\第四阶段中文语义方法数据说明报告.md")
report_path.write_text("\n".join(lines), encoding='utf-8')
print(json.dumps({"json": str(json_path), "report": str(report_path)}, ensure_ascii=False))
