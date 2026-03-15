import json
import re

from .llm_client import DeepSeekClient
from .schema import Variable, VariableType


class VariableExtractor:
    def __init__(self, llm_client: DeepSeekClient | None, rules: dict) -> None:
        self.llm_client = llm_client
        section = rules.get("variable_extractor", {})
        self.allowed_types = set(section.get("allowed_types", ["string", "number", "boolean", "list", "object"]))
        patterns = section.get("patterns", {})
        self.email_pattern = patterns.get("email", r"[\w\.-]+@[\w\.-]+\.\w+")
        self.url_pattern = patterns.get("url", r"https?://[^\s，,]+")
        self.number_pattern = patterns.get("number", r"\d+(?:\.\d+)?")
        self.boolean_pattern = patterns.get("boolean", r"(开启|关闭|是|否|true|false)")
        self.json_like_pattern = patterns.get("json_like", r"\{[^\}]+\}|\[[^\]]+\]")
        self.bool_true_values = {str(item).lower() for item in section.get("bool_true_values", ["true", "是", "开启"])}
        self.fallback_raw_text_name = section.get("fallback_raw_text_name", "raw_text")
        self.structured_input_name = section.get("structured_input_name", "structured_input")

    def extract(self, normalized_text: str) -> list[Variable]:
        if self.llm_client:
            try:
                return self._extract_by_llm(normalized_text)
            except Exception:
                return self._extract_by_rules(normalized_text)
        return self._extract_by_rules(normalized_text)

    def _extract_by_llm(self, text: str) -> list[Variable]:
        result = self.llm_client.chat_json(
            system_prompt=(
                "你是变量提取引擎。"
                "只输出一个JSON对象，不要输出解释、不要markdown、不要代码块。"
            ),
            user_prompt=(
                "任务：提取任务描述中的可复用变量。\n"
                "输出格式必须是：\n"
                '{"variables":[{"name":"string","type":"string|number|boolean|list|object","value":null,"source":"user_input"}]}\n'
                "约束：\n"
                "1) name用英文snake_case，表达变量语义。\n"
                "2) type只能是string、number、boolean、list、object。\n"
                "3) value必须与type一致：number为数值，boolean为true/false，list/object为JSON结构。\n"
                "4) source固定为user_input。\n"
                "5) 提取邮箱、URL、数字、布尔开关、JSON片段等显式值。\n"
                "6) 不要输出空变量；确实没有时返回空数组。\n"
                f"任务：{text}"
            ),
        )
        variables: list[Variable] = []
        for item in result.get("variables", []):
            raw_type = str(item.get("type", "string"))
            if raw_type not in self.allowed_types:
                raw_type = "string"
            variables.append(
                Variable(
                    name=str(item.get("name", "")).strip() or "variable",
                    type=VariableType(raw_type),
                    value=item.get("value"),
                    source=str(item.get("source", "user_input")),
                )
            )
        return variables

    def _extract_by_rules(self, text: str) -> list[Variable]:
        variables: list[Variable] = []
        for email in re.findall(self.email_pattern, text):
            variables.append(
                Variable(name="email_recipient", type=VariableType.string, value=email, source="user_input")
            )
        for url in re.findall(self.url_pattern, text):
            variables.append(Variable(name="url", type=VariableType.string, value=url, source="user_input"))
        for number in re.findall(self.number_pattern, text):
            value = float(number) if "." in number else int(number)
            variables.append(Variable(name=f"number_{number}", type=VariableType.number, value=value, source="user_input"))
        for bool_token in re.findall(self.boolean_pattern, text, flags=re.IGNORECASE):
            bool_value = bool_token.lower() in self.bool_true_values
            variables.append(
                Variable(name=f"flag_{bool_token}", type=VariableType.boolean, value=bool_value, source="user_input")
            )
        json_like = re.findall(self.json_like_pattern, text)
        for item in json_like:
            try:
                parsed = json.loads(item)
            except Exception:
                continue
            var_type = VariableType.list if isinstance(parsed, list) else VariableType.object
            variables.append(Variable(name=self.structured_input_name, type=var_type, value=parsed, source="user_input"))
        if not variables:
            variables.append(
                Variable(name=self.fallback_raw_text_name, type=VariableType.string, value=text, source="user_input")
            )
        return variables
