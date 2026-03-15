import re

from .llm_client import DeepSeekClient
from .schema import ControlIntent, ControlIntentType


class ControlIntentExtractor:
    def __init__(self, llm_client: DeepSeekClient | None, rules: dict) -> None:
        self.llm_client = llm_client
        section = rules.get("control_intent_extractor", {})
        self.allowed_types = set(section.get("allowed_types", ["sequential", "conditional", "parallel", "iteration"]))
        self.conditional_triggers = section.get("conditional_triggers", [])
        self.parallel_triggers = section.get("parallel_triggers", [])
        self.iteration_triggers = section.get("iteration_triggers", [])
        self.condition_pattern = section.get("condition_extract_pattern", r"(如果|若|当)([^，,；;]+)")
        self.loop_target_pattern = section.get("loop_target_extract_pattern", r"(对所有|每个|逐个)([^，,；;]+)")

    def extract(self, normalized_text: str) -> list[ControlIntent]:
        if self.llm_client:
            try:
                return self._extract_by_llm(normalized_text)
            except Exception:
                return self._extract_by_rules(normalized_text)
        return self._extract_by_rules(normalized_text)

    def _extract_by_llm(self, text: str) -> list[ControlIntent]:
        result = self.llm_client.chat_json(
            system_prompt=(
                "你是控制结构识别引擎。"
                "只输出一个JSON对象，不要输出解释、不要markdown、不要代码块。"
            ),
            user_prompt=(
                "任务：识别流程中的控制结构。\n"
                "输出格式必须是：\n"
                '{"control_intents":[{"type":"sequential|conditional|parallel|iteration","condition":"","target_actions":[],"loop_target":"","loop_condition":""}]}\n'
                "约束：\n"
                "1) type只能取sequential、conditional、parallel、iteration。\n"
                "2) 必须至少输出一个sequential意图。\n"
                "3) conditional仅在存在明确条件时输出，condition填中文条件语句。\n"
                "4) parallel仅在存在“同时/并行/分别”等并发语义时输出。\n"
                "5) iteration仅在存在“每个/遍历/循环”等语义时输出，并填写loop_target和loop_condition。\n"
                "6) target_actions填写动作名列表，优先使用英文snake_case动作名；无法确定可为空数组。\n"
                "7) 不要编造不存在的条件或循环目标。\n"
                f"任务：{text}"
            ),
        )
        items = result.get("control_intents", [])
        intents: list[ControlIntent] = []
        for item in items:
            raw_type = str(item.get("type", "sequential"))
            if raw_type not in self.allowed_types:
                raw_type = "sequential"
            intents.append(
                ControlIntent(
                    type=ControlIntentType(raw_type),
                    condition=str(item.get("condition", "")),
                    target_actions=[str(x) for x in item.get("target_actions", [])],
                    loop_target=str(item.get("loop_target", "")),
                    loop_condition=str(item.get("loop_condition", "")),
                )
            )
        return intents

    def _extract_by_rules(self, text: str) -> list[ControlIntent]:
        intents: list[ControlIntent] = [ControlIntent(type=ControlIntentType.sequential)]
        if self._contains_trigger(text, self.conditional_triggers):
            condition = self._extract_condition(text)
            intents.append(ControlIntent(type=ControlIntentType.conditional, condition=condition))
        if self._contains_trigger(text, self.parallel_triggers):
            intents.append(ControlIntent(type=ControlIntentType.parallel))
        if self._contains_trigger(text, self.iteration_triggers):
            target = self._extract_loop_target(text)
            intents.append(
                ControlIntent(
                    type=ControlIntentType.iteration,
                    loop_target=target,
                    loop_condition=f"for each {target}" if target else "",
                )
            )
        return intents

    def _extract_condition(self, text: str) -> str:
        m = re.search(self.condition_pattern, text)
        if not m:
            return ""
        trigger = m.group(1).strip()
        if trigger == "失败则":
            return "失败"
        if trigger == "成功则":
            return "成功"
        return m.group(2).strip()

    def _extract_loop_target(self, text: str) -> str:
        m = re.search(self.loop_target_pattern, text)
        if not m:
            return ""
        return m.group(2).strip()

    def _contains_trigger(self, text: str, triggers: list[str]) -> bool:
        return any(token and token in text for token in triggers)
