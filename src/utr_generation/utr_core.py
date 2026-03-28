import uuid
import datetime
import json
from src.core.schema import UTR, UTRMetadata, Action, Resource, Variable
from src.core.llm_client import DeepSeekClient
from src.core.config import Settings, load_settings

class UTRGenerator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.llm_client = DeepSeekClient(self.settings) if self.settings.llm_enabled else None

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

    def _extract_core_elements(self, task_desc: str) -> UTRMetadata:
        if not self.llm_client:
            # Fallback mock for testing without LLM
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
            deps = result.get("implicit_dependencies", [])

            return UTRMetadata(
                task_goal=result.get("task_goal", ""),
                core_actions=actions,
                core_resources=resources,
                core_variables=variables,
                implicit_dependencies=deps
            )
        except Exception as e:
            print(f"Error extracting core elements: {e}")
            return UTRMetadata(task_goal="error", core_actions=[], core_resources=[], core_variables=[], implicit_dependencies=[])
