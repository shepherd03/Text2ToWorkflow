from src.core.schema import (
    Action,
    ActionSlot,
    ConditionalBlock,
    LoopBlock,
    ParallelBlock,
    SequentialBlock,
    UTR,
    UTRMetadata,
)
from src.dsl_generation.pipeline import DSLGenerationPipeline
from src.dsl_generation.validators import DSLInputValidator


def _build_sample_utr() -> UTR:
    return UTR(
        task_id="task_1",
        task_desc="测试任务",
        create_time="2026-03-29T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="完成测试",
            core_actions=[
                Action(action_id="act_1", action_name="generate_summary"),
                Action(action_id="act_2", action_name="transform_json"),
            ],
        ),
    )


def _build_valid_skeleton() -> SequentialBlock:
    return SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ActionSlot(action_id="act_1", action_name="generate_summary"),
            ActionSlot(action_id="act_2", action_name="transform_json"),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )


def test_validate_inputs_success():
    validator = DSLInputValidator()
    output = validator.validate(_build_sample_utr(), _build_valid_skeleton())
    assert output.report.passed is True
    assert output.context is not None
    assert output.context.block_stats["ActionSlot"] == 4


def test_validate_inputs_action_ref_not_found():
    validator = DSLInputValidator()
    skeleton = SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ActionSlot(action_id="act_404", action_name="unknown"),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )
    output = validator.validate(_build_sample_utr(), skeleton)
    assert output.report.passed is False
    assert output.report.action_ref_valid is False
    assert output.context is None


def test_validate_inputs_empty_parallel_branch():
    validator = DSLInputValidator()
    skeleton = SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ParallelBlock(branches=[SequentialBlock(children=[])]),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )
    output = validator.validate(_build_sample_utr(), skeleton)
    assert output.report.passed is False
    assert output.report.structure_valid is False


def test_validate_inputs_reject_unknown_block_type():
    validator = DSLInputValidator()
    skeleton = SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ActionSlot(action_id="act_1", action_name="generate_summary", type="Unknown"),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )
    output = validator.validate(_build_sample_utr(), skeleton)
    assert output.report.passed is False
    assert output.report.node_type_valid is False


def test_pipeline_validate_inputs_adapter():
    pipeline = DSLGenerationPipeline()
    output = pipeline.validate_inputs(_build_sample_utr(), _build_valid_skeleton())
    assert output.report.passed is True


def test_run_step2_normalize_success():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step2(_build_sample_utr(), _build_valid_skeleton())
    assert output.precheck_report.passed is True
    assert output.normalization_report.success is True
    assert output.context is not None
    assert len(output.context.root_entry_slot_ids) == 1
    assert len(output.context.root_exit_slot_ids) == 1
    start_slot_id = output.context.root_entry_slot_ids[0]
    end_slot_id = output.context.root_exit_slot_ids[0]
    slots = {slot.slot_id: slot for slot in output.context.action_slots}
    assert slots[start_slot_id].action_id == "start_node"
    assert slots[end_slot_id].action_id == "end_node"


def test_run_step2_auto_fill_else_branch():
    pipeline = DSLGenerationPipeline()
    skeleton = SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ConditionalBlock(
                condition_description="当数据有效时继续",
                branches={
                    "true": SequentialBlock(
                        children=[ActionSlot(action_id="act_1", action_name="generate_summary")]
                    )
                },
            ),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )
    output = pipeline.run_step2(_build_sample_utr(), skeleton)
    assert output.normalization_report.success is True
    assert output.context is not None
    conditional_blocks = [block for block in output.context.blocks if block.block_type == "Conditional"]
    assert len(conditional_blocks) == 1
    assert conditional_blocks[0].has_default_else is True
    assert len(output.normalization_report.warnings) > 0


def test_run_step2_parallel_domain_and_join():
    pipeline = DSLGenerationPipeline()
    skeleton = SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ParallelBlock(
                branches=[
                    SequentialBlock(children=[ActionSlot(action_id="act_1", action_name="generate_summary")]),
                    SequentialBlock(children=[ActionSlot(action_id="act_2", action_name="transform_json")]),
                ]
            ),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )
    output = pipeline.run_step2(_build_sample_utr(), skeleton)
    assert output.normalization_report.success is True
    assert output.context is not None
    parallel_blocks = [block for block in output.context.blocks if block.block_type == "Parallel"]
    assert len(parallel_blocks) == 1
    assert parallel_blocks[0].needs_join is True
    branch_slots = [
        slot
        for slot in output.context.action_slots
        if slot.action_id in {"act_1", "act_2"}
    ]
    assert len(branch_slots) == 2
    for slot in branch_slots:
        assert any(domain.domain_type == "parallel" for domain in slot.control_domains)


def test_run_step2_loop_domain():
    pipeline = DSLGenerationPipeline()
    skeleton = SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            LoopBlock(
                loop_condition="遍历列表",
                body=SequentialBlock(
                    children=[ActionSlot(action_id="act_1", action_name="generate_summary")]
                ),
            ),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )
    output = pipeline.run_step2(_build_sample_utr(), skeleton)
    assert output.normalization_report.success is True
    assert output.context is not None
    loop_slots = [slot for slot in output.context.action_slots if slot.action_id == "act_1"]
    assert len(loop_slots) == 1
    assert any(domain.domain_type == "loop" for domain in loop_slots[0].control_domains)
