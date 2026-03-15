from copy import deepcopy
from pathlib import Path

from utr_generator.config import Settings
from utr_generator.pipeline import UTRGenerationPipeline
from utr_generator.rulebook import load_rulebook
from utr_generator.schema import Action, ControlIntent, ControlIntentType
from utr_generator.utr_combiner import UTRCombiner


def build_pipeline() -> UTRGenerationPipeline:
    settings = Settings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        strict_completeness=False,
    )
    return UTRGenerationPipeline(settings=settings)


def test_pipeline_basic_generation() -> None:
    pipeline = build_pipeline()
    text = "先抓取订单数据，同时清洗并分析，如果失败就通知运维，最后把结果发送到ops@example.com"
    output = pipeline.run(text)
    assert len(output.utr.actions) >= 3
    assert len(output.utr.resources) >= 1
    assert len(output.utr.variables) >= 1
    assert output.report.schema_valid
    assert output.report.logic_valid


def test_action_order_is_consecutive() -> None:
    pipeline = build_pipeline()
    output = pipeline.run("下载日志并校验后发送邮件")
    orders = [item.order for item in output.utr.actions]
    assert orders == list(range(1, len(orders) + 1))


def test_pipeline_uses_runtime_rules_override() -> None:
    settings = Settings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        strict_completeness=False,
    )
    rules = deepcopy(load_rulebook())
    rules["action_extractor"]["verb_map"]["归档"] = "archive_result"
    pipeline = UTRGenerationPipeline(settings=settings, rules=rules)
    output = pipeline.run("归档历史记录")
    action_names = [item.action_name for item in output.utr.actions]
    assert "archive_result" in action_names


def test_load_rulebook_from_split_directory() -> None:
    rulebook = load_rulebook()
    assert "preprocessor" in rulebook
    assert "action_extractor" in rulebook
    assert "resource_extractor" in rulebook
    assert "control_intent_extractor" in rulebook
    assert "variable_extractor" in rulebook


def test_load_rulebook_from_custom_directory() -> None:
    rule_dir = Path(__file__).resolve().parent.parent / "configs" / "rules"
    rulebook = load_rulebook(str(rule_dir))
    assert "verb_map" in rulebook["action_extractor"]


def test_variable_extraction_does_not_overmatch_url_and_email() -> None:
    pipeline = build_pipeline()
    output = pipeline.run("调用https://api.example.com/report并把结果发给ops@example.com")
    values = [item.value for item in output.utr.variables if isinstance(item.value, str)]
    assert "https://api.example.com/report" in values
    assert "ops@example.com" in values


def test_rule_fallback_does_not_convert_condition_to_action() -> None:
    pipeline = build_pipeline()
    output = pipeline.run("先获取客户清单，若接口超时则告警，最后发送结果")
    action_names = [item.action_name for item in output.utr.actions]
    assert "接口超时则告_task" not in action_names


def test_condition_from_failure_trigger_is_normalized() -> None:
    pipeline = build_pipeline()
    output = pipeline.run("下载报表，失败则重试并通知负责人")
    conditional = [item for item in output.utr.control_intents if item.type.value == "conditional"]
    assert conditional
    assert conditional[0].condition == "失败"


def test_if_else_actions_are_both_extracted() -> None:
    pipeline = build_pipeline()
    output = pipeline.run("如果金额大于1000则走人工审批，否则自动放行")
    action_names = [item.action_name for item in output.utr.actions]
    assert "approve_request" in action_names
    assert "release_request" in action_names


def test_iteration_parallel_sentence_extracts_sync_and_send() -> None:
    pipeline = build_pipeline()
    output = pipeline.run("遍历每个租户同步配置并分别发送通知")
    action_names = [item.action_name for item in output.utr.actions]
    assert "sync_data" in action_names
    assert "send_message" in action_names


def test_combiner_drops_unknown_control_targets() -> None:
    combiner = UTRCombiner()
    actions = [
        Action(action_name="fetch_data", order=1),
        Action(action_name="send_message", order=2),
    ]
    intents = [
        ControlIntent(
            type=ControlIntentType.sequential,
            target_actions=["fetch_data", "missing_action", "act_9"],
        )
    ]
    utr = combiner.combine(actions=actions, resources=[], control_intents=intents, variables=[])
    assert utr.control_intents[0].target_actions == ["act_1"]
