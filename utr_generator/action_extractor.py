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
                "你是高级工作流架构师。你的任务是将用户的简短自然语言指令，扩展、拆解为生产级（Production-Grade）的详细工作流步骤。\n"
                "用户指令通常是高层次的（例如'把代码翻译成另一种语言'），但真实的工作流需要处理多步逻辑，例如：接收输入 -> 验证/预处理 -> 执行核心操作（如LLM生成） -> 后处理（如代码提取/清理） -> 输出结果。\n"
                "请你运用 Chain-of-Thought 思考，想象在一个真实的类似 Dify 的工作流引擎中，需要哪些具体节点来稳健地完成这个任务。补充缺失的必要环节，使工作流变得完整、健壮。只输出一个JSON对象，不要输出解释、不要markdown、不要代码块。"
            ),
            user_prompt=(
                "任务：从下面中文描述提取按执行顺序排列的动作序列，并提取每个动作的参数（args）。你需要把简单任务扩展为完整的生产级工作流。\n\n"
                "输出格式必须是：\n"
                '{"actions":[{"action_name":"string","description":"string","order":1,"args":{"key":"value"}}]}\n\n'
                "【Few-Shot 示例 1 - 简单任务复杂化】\n"
                "用户指令: \"我想把一种编程语言的代码翻译成另一种语言。\"\n"
                "你的输出:\n"
                '{"actions": [\n'
                '  {"action_name": "variable_assignment", "description": "定义源语言、目标语言和输入代码的变量", "order": 1, "args": {"variables": ["source_lang", "target_lang", "code"]}},\n'
                '  {"action_name": "llm_generation", "description": "调用大模型进行代码翻译", "order": 2, "args": {"prompt_template": "将以下{{source_lang}}代码翻译为{{target_lang}}:\\n{{code}}", "model_config": "gpt-4"}},\n'
                '  {"action_name": "code_execution", "description": "使用Python脚本从大模型回复中提取纯代码块", "order": 3, "args": {"script": "def main(llm_output): return extract_code(llm_output)"}}\n'
                ']}\n\n'
                "【Few-Shot 示例 2 - 包含工具调用和后处理】\n"
                "用户指令: \"我想做一个能自动读取CSV文件内容的工具。\"\n"
                "你的输出:\n"
                '{"actions": [\n'
                '  {"action_name": "variable_assignment", "description": "接收用户输入的文件路径或文件对象", "order": 1, "args": {"variables": ["file_path"]}},\n'
                '  {"action_name": "code_execution", "description": "执行Python脚本读取CSV文件并解析为JSON数组", "order": 2, "args": {"script": "import pandas as pd\\ndef main(file_path): return pd.read_csv(file_path).to_dict(orient=\'records\')"}},\n'
                '  {"action_name": "condition_branch", "description": "检查CSV数据是否为空", "order": 3, "args": {"condition": "len(csv_data) > 0"}}\n'
                ']}\n\n'
                "【约束】\n"
                "1) action_name必须是英文snake_case，只能包含小写字母、数字、下划线。\n"
                "2) 推荐使用标准动作名称：llm_generation, code_execution, http_request, knowledge_retrieval, web_search, variable_assignment, condition_branch, image_generation, list_operation。\n"
                "3) 极力避免生造工具（如 web_scraper 读取本地文件是错误的，应该用 code_execution 写代码读取）。\n"
                "4) order从1开始连续递增。\n"
                "5) description使用中文详细描述该动作做什么。\n"
                "6) args 字段必须是对象，尽可能详细地推断参数（如 prompt_template, script, url 等），若无参数则为空对象。\n"
                "7) 思考：这个任务在真实世界中可能会遇到什么异常？是否需要前置变量定义？是否需要后置数据格式化？把这些思考转化为额外的 action。\n\n"
                f"当前任务描述：{text}"
            ),
            temperature=0.3
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
