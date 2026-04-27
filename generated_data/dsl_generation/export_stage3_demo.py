import json
from pathlib import Path

from src.core.schema import Action, ActionSlot, ConditionalBlock, SequentialBlock, UTR, UTRMetadata, Variable, VariableType
from src.dsl_generation.pipeline import DSLGenerationPipeline

utr = UTR(
    task_id="task_stage_demo_20260407",
    task_desc="根据风险等级决定摘要策略并输出结果",
    create_time="2026-04-07T12:00:00Z",
    metadata=UTRMetadata(
        task_goal="按风险等级选择处理分支",
        core_actions=[
            Action(action_id="act_high", action_name="generate_risk_summary", description="生成高风险摘要", inputs=["article_text"], outputs=["risk_summary"]),
            Action(action_id="act_low", action_name="transform_safe_result", description="转换低风险结构化结果", inputs=["article_text"], outputs=["safe_result"]),
        ],
        core_variables=[
            Variable(var_id="var_1", name="article_text", type=VariableType.string, source="user"),
            Variable(var_id="var_2", name="risk_level", type=VariableType.number, source="system"),
        ],
    ),
)

skeleton = SequentialBlock(
    children=[
        ActionSlot(action_id="start_node", action_name="start"),
        ConditionalBlock(
            condition_description="如果风险分数大于等于 80 走高风险分支，否则走低风险分支",
            branches={
                "high": SequentialBlock(children=[ActionSlot(action_id="act_high", action_name="generate_risk_summary")]),
                "else": SequentialBlock(children=[ActionSlot(action_id="act_low", action_name="transform_safe_result")]),
            },
        ),
        ActionSlot(action_id="end_node", action_name="end"),
    ]
)

pipeline = DSLGenerationPipeline()
output = pipeline.run_step3_minimal(utr, skeleton)

result = {
    "utr": utr.model_dump(mode="json"),
    "skeleton": skeleton.model_dump(mode="json"),
    "compile_report": output.compile_report.model_dump(mode="json"),
    "compiled_graph": output.compiled_graph.model_dump(mode="json") if output.compiled_graph else None,
    "workflow": output.workflow.model_dump(mode="json") if output.workflow else None,
}

path = Path(r"e:\Desktop\论文\工作流\utr\generated_data\dsl_generation\stage3_demo_output_2026_04_07.json")
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(path)
