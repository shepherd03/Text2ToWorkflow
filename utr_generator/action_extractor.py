import re

from .llm_client import DeepSeekClient
from .schema import Action
from .utils import unique_keep_order


class ActionExtractor:
    def __init__(self, llm_client: DeepSeekClient | None, rules: dict) -> None:
        self.llm_client = llm_client
        section = rules.get("action_extractor", {})
        self.verb_map = section.get("verb_map", {})
        self.fallback_pattern = section.get("fallback_pattern", r"([\u4e00-\u9fa5]{2,6})")
        self.fallback_suffix = section.get("fallback_suffix", "_task")
        self.skip_clause_prefixes = section.get("skip_clause_prefixes", [])
        self.skip_clause_contains = section.get("skip_clause_contains", [])

    def extract(self, normalized_text: str, clauses: list[str]) -> list[Action]:
        if self.llm_client:
            try:
                return self._extract_by_llm(normalized_text)
            except Exception:
                return self._extract_by_rules(clauses)
        return self._extract_by_rules(clauses)

    def _extract_by_llm(self, text: str) -> list[Action]:
        result = self.llm_client.chat_json(
            system_prompt=(
                "你是工作流动作抽取引擎。"
                "只输出一个JSON对象，不要输出解释、不要markdown、不要代码块。"
            ),
            user_prompt=(
                "任务：从下面中文描述提取按执行顺序排列的动作序列，并提取每个动作的参数（args）。\n"
                "输出格式必须是：\n"
                '{"actions":[{"action_name":"string","description":"string","order":1,"args":{"key":"value"}}]}\n'
                "约束：\n"
                "1) action_name必须是英文snake_case，只能包含小写字母、数字、下划线。\n"
                "2) 推荐使用标准动作名称：web_search, image_generation, llm_generation, code_execution, http_request, translation, email_service, web_scraper。\n"
                "3) 禁止出现中文action_name，禁止空字符串。\n"
                "4) order从1开始连续递增，不要跳号。\n"
                "5) description使用中文简短描述该动作做什么，不超过20字。\n"
                "6) 仅保留可执行动作，不要把条件语句本身当动作。\n"
                "7) 动作去重，语义重复只保留一次。\n"
                "8) args 字段必须是对象，尽可能从描述中提取参数键值对，若无参数则为空对象。\n"
                "9) 若无法判断，使用task_step_n作为action_name。\n"
                f"任务描述：{text}"
            ),
        )
        items = result.get("actions", [])
        actions: list[Action] = []
        for index, item in enumerate(items, start=1):
            order = int(item.get("order", index))
            action_name = str(item.get("action_name", f"task_step_{index}")).strip() or f"task_step_{index}"
            actions.append(
                Action(
                    action_name=action_name,
                    description=str(item.get("description", "")),
                    order=order,
                    args=item.get("args", {})
                )
            )
        actions.sort(key=lambda x: x.order)
        deduped = self._dedupe_actions(actions)
        return self._renumber(deduped)

    def _extract_by_rules(self, clauses: list[str]) -> list[Action]:
        verbs: list[str] = []
        for clause in clauses:
            matched = False
            first_hit_index = len(clause) + 1
            selected_action: str | None = None
            for zh_verb, normalized in self.verb_map.items():
                pos = clause.find(zh_verb)
                if pos >= 0 and pos < first_hit_index:
                    first_hit_index = pos
                    selected_action = normalized
            if selected_action:
                verbs.append(selected_action)
                matched = True
            if not matched and not self._should_skip_clause(clause):
                m = re.match(self.fallback_pattern, clause)
                if m:
                    verbs.append(f"{m.group(1)}{self.fallback_suffix}")
        verbs = unique_keep_order(verbs)
        return [Action(action_name=name, description="", order=i) for i, name in enumerate(verbs, start=1)]

    def _dedupe_actions(self, actions: list[Action]) -> list[Action]:
        seen: set[str] = set()
        result: list[Action] = []
        for action in actions:
            key = action.action_name.strip().lower()
            if key not in seen:
                seen.add(key)
                result.append(action)
        return result

    def _renumber(self, actions: list[Action]) -> list[Action]:
        for index, action in enumerate(actions, start=1):
            action.order = index
        return actions

    def _should_skip_clause(self, clause: str) -> bool:
        content = clause.strip()
        if any(content.startswith(prefix) for prefix in self.skip_clause_prefixes):
            return True
        return any(token and token in content for token in self.skip_clause_contains)
