import uuid
import datetime
import json
from typing import Any
from src.core.schema import UTR, UTRMetadata, Action, Resource, Variable
from src.core.llm_client import DeepSeekClient
from src.core.config import Settings, load_settings

class UTRGenerator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.llm_client = DeepSeekClient(self.settings) if self.settings.llm_enabled else None
        self.last_generation_meta: dict[str, Any] = {}

    def generate_utr(self, task_desc: str) -> UTR:
        """
        生成 UTR 基础元数据，作为后续骨架规划的参考依据。
        不包含具体的骨架控制逻辑，仅提取核心元素和潜在的依赖关系。
        """
        metadata = self._extract_core_elements(task_desc)
        
        return UTR(
            task_id=str(uuid.uuid4()),
            task_desc=task_desc,
            metadata=metadata,
            create_time=datetime.datetime.now().isoformat()
        )

    def _normalize_dependencies(
        self,
        actions: list[Action],
        deps: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        valid_action_ids = {action.action_id for action in actions if action.action_id}
        normalized: list[dict[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for dep in deps:
            from_id = str(dep.get("from", "")).strip()
            to_id = str(dep.get("to", "")).strip()
            reason = str(dep.get("reason", "")).strip()

            # Drop malformed, dangling, or self-referential edges before planning.
            if not from_id or not to_id:
                continue
            if from_id not in valid_action_ids or to_id not in valid_action_ids:
                continue
            if from_id == to_id:
                continue

            pair = (from_id, to_id)
            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)
            normalized.append({
                "from": from_id,
                "to": to_id,
                "reason": reason,
            })

        return normalized

    def _extract_core_elements(self, task_desc: str) -> UTRMetadata:
        if not self.llm_client:
            # Fallback mock for testing without LLM
            self.last_generation_meta = {
                "generation_source": "fallback",
                "llm_enabled": False,
                "llm_call_count": 0,
                "llm_model": "",
                "llm_usage": {},
            }
            return UTRMetadata(
                task_goal="mock task",
                core_actions=[Action(action_id="act_1", action_name="mock_action", description="mock")],
                core_resources=[],
                core_variables=[],
                implicit_dependencies=[]
            )

        system_prompt = (
            "你是一个需求分析师。你的任务是分析用户的任务描述，并提取其中的核心元素（目标、核心动作、涉及资源、潜在依赖）。\n"
            "请注意：你不需要设计具体的工作流执行步骤，不需要判断并行、条件分支等控制流逻辑，"
            "你只需要将自然语言中的实体和意图提取出来，供后续的骨架规划器参考。\n"
            "只输出一个合法的 JSON 对象，不要输出任何额外的解释或 Markdown 代码块符号。"
        )

        user_prompt = (
            "任务描述：\n"
            f"{task_desc}\n\n"
            "请提取以下信息，并严格按照 JSON 格式输出：\n"
            "{\n"
            '  "task_goal": "任务总体目标描述",\n'
            '  "core_actions": [\n'
            '    {"action_id": "唯一ID(如act_1)", "action_name": "英文动作名(snake_case)", "description": "中文动作描述", "inputs": ["所需变量名"], "outputs": ["产生变量名"]}\n'
            '  ],\n'
            '  "core_resources": [\n'
            '    {"resource_id": "唯一ID(如res_1)", "name": "资源名", "type": "data|file|service|target|variable|tool", "description": "资源描述"}\n'
            '  ],\n'
            '  "core_variables": [\n'
            '    {"var_id": "唯一ID(如var_1)", "name": "变量名", "type": "string|number|boolean|list|object|file|data(只能选一个)", "source": "来源描述"}\n'
            '  ],\n'
            '  "implicit_dependencies": [\n'
            '    {"from": "前置动作action_id", "to": "后置动作action_id", "reason": "描述为什么有依赖"}\n'
            '  ]\n'
            "}"
        )

        try:
            result = self.llm_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3
            )
            
            # Parse the response to UTRMetadata
            actions = [Action(**act) for act in result.get("core_actions", [])]
            resources = [Resource(**res) for res in result.get("core_resources", [])]
            variables = [Variable(**var) for var in result.get("core_variables", [])]
            deps = self._normalize_dependencies(actions, result.get("implicit_dependencies", []))
            self.last_generation_meta = {
                "generation_source": "llm",
                "llm_enabled": True,
                "llm_call_count": self.llm_client.call_count if self.llm_client else 0,
                "llm_model": self.llm_client.last_model if self.llm_client else "",
                "llm_usage": self.llm_client.last_usage if self.llm_client else {},
            }

            return UTRMetadata(
                task_goal=result.get("task_goal", ""),
                core_actions=actions,
                core_resources=resources,
                core_variables=variables,
                implicit_dependencies=deps
            )
        except Exception as e:
            raise RuntimeError(f"Error extracting core elements: {e}") from e
