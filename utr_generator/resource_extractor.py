import re

from .llm_client import DeepSeekClient
from .schema import Resource, ResourceType
from .utils import unique_keep_order


class ResourceExtractor:
    def __init__(self, llm_client: DeepSeekClient | None, rules: dict) -> None:
        self.llm_client = llm_client
        section = rules.get("resource_extractor", {})
        self.valid_types = set(section.get("valid_types", ["data", "file", "service", "target", "variable"]))
        self.regex_rules = section.get("regex_rules", [])
        self.keyword_type_map = section.get("keyword_type_map", {})
        self.fallback_resource = section.get("fallback_resource", {"name": "task_context", "type": "data"})

    def extract(self, normalized_text: str) -> list[Resource]:
        if self.llm_client:
            try:
                return self._extract_by_llm(normalized_text)
            except Exception:
                return self._extract_by_rules(normalized_text)
        return self._extract_by_rules(normalized_text)

    def _extract_by_llm(self, text: str) -> list[Resource]:
        result = self.llm_client.chat_json(
            system_prompt=(
                "你是资源抽取引擎。"
                "只输出一个JSON对象，不要输出解释、不要markdown、不要代码块。"
            ),
            user_prompt=(
                "任务：从描述中提取被动作使用或产生的资源。\n"
                "输出格式必须是：\n"
                '{"resources":[{"name":"string","type":"data|file|service|target|variable","description":"string"}]}\n'
                "分类规则：\n"
                "- data: 数据集、表、记录、文本内容\n"
                "- file: 文件、路径、报表文件\n"
                "- service: API、HTTP接口、外部系统服务\n"
                "- target: 接收方或目标对象（如邮箱、负责人、群组）\n"
                "- variable: 被显式命名并跨步骤复用的变量\n"
                "约束：\n"
                "1) name必须具体，不要用resource、unknown等泛化词。\n"
                "2) 去重后输出，同名同类只保留一次。\n"
                "3) description用中文简要说明资源用途，不超过20字。\n"
                "4) 不要把纯动作词当资源。\n"
                f"任务：{text}"
            ),
        )
        resources: list[Resource] = []
        for item in result.get("resources", []):
            r_type = str(item.get("type", "data"))
            if r_type not in self.valid_types:
                r_type = "data"
            resources.append(
                Resource(
                    name=str(item.get("name", "")).strip() or "unknown_resource",
                    type=ResourceType(r_type),
                    description=str(item.get("description", "")),
                )
            )
        return self._dedupe(resources)

    def _extract_by_rules(self, text: str) -> list[Resource]:
        found: list[tuple[str, ResourceType]] = []
        for item in self.regex_rules:
            pattern = str(item.get("pattern", ""))
            if pattern and re.search(pattern, text):
                name = str(item.get("name", "resource"))
                type_name = str(item.get("type", "data"))
                found.append((name, ResourceType(type_name if type_name in self.valid_types else "data")))
        for token, type_name in self.keyword_type_map.items():
            if token in text:
                found.append((token, ResourceType(type_name if type_name in self.valid_types else "data")))
        unique_names = unique_keep_order([item[0] for item in found])
        resources: list[Resource] = []
        for name in unique_names:
            r_type = next(item[1] for item in found if item[0] == name)
            resources.append(Resource(name=name, type=r_type, description=""))
        if not resources:
            fallback_name = str(self.fallback_resource.get("name", "task_context"))
            fallback_type = str(self.fallback_resource.get("type", "data"))
            resources.append(
                Resource(
                    name=fallback_name,
                    type=ResourceType(fallback_type if fallback_type in self.valid_types else "data"),
                    description="",
                )
            )
        return resources

    def _dedupe(self, resources: list[Resource]) -> list[Resource]:
        seen: set[str] = set()
        result: list[Resource] = []
        for res in resources:
            key = f"{res.name}|{res.type}"
            if key not in seen:
                seen.add(key)
                result.append(res)
        return result
