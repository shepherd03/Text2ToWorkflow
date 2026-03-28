import uuid
from typing import List, Union, Optional, Any
from pydantic import BaseModel, Field
import json

from src.core.schema import Action, UTR
from src.core.config import load_settings
from src.core.llm_client import DeepSeekClient

class Block(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str

class ActionSlot(Block):
    type: str = "ActionSlot"
    action_id: str
    action_name: str = ""

class SequentialBlock(Block):
    type: str = "Sequential"
    children: List['BlockType'] = Field(default_factory=list)

class ParallelBlock(Block):
    type: str = "Parallel"
    branches: List[SequentialBlock] = Field(default_factory=list)

class ConditionalBlock(Block):
    type: str = "Conditional"
    condition_description: str = ""
    branches: dict[str, SequentialBlock] = Field(default_factory=dict)

class LoopBlock(Block):
    type: str = "Loop"
    loop_condition: str = ""
    body: SequentialBlock = Field(default_factory=SequentialBlock)

BlockType = Union[ActionSlot, SequentialBlock, ParallelBlock, ConditionalBlock, LoopBlock]

class SkeletonPlanner:
    """
    步骤2：工作流骨架规划 (Workflow Skeleton Planning)
    采用“算法主导拓扑，大模型辅助判断”的混合校验架构 (LLM-Assisted Validation):
    1. 拓扑维护（算法端）：基于图遍历与拓扑排序算法，维护骨架树，控制最大嵌套深度。
    2. 条件判定（模型端）：遍历 UTR 时，根据动作的输入输出依赖，向大模型发起局部查询（True/False）。
    3. 节点派生：算法根据大模型返回结果，派生 ConditionalBlock 或 LoopBlock 并挂载节点。
    """
    
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        self.llm_client = DeepSeekClient(self.settings) if self.settings.llm_enabled else None
        self.max_depth = 3  # 控制最大嵌套深度
    
    def plan(self, utr: UTR) -> SequentialBlock:
        actions = utr.metadata.core_actions
        deps = utr.metadata.implicit_dependencies
        
        # Step 1: 拓扑排序与分层 (Topological Sort & Layering)
        layers = self._build_dependency_layers(actions, deps)
        
        # Step 2: 遍历节点并结合大模型判断进行节点派生
        root_block = SequentialBlock()
        root_block.children.append(ActionSlot(action_id="start_node", action_name="start"))
        
        for layer in layers:
            if len(layer) == 1:
                action = layer[0]
                node = self._process_action_with_llm(utr, action, deps, current_depth=1)
                root_block.children.append(node)
            else:
                p_block = ParallelBlock()
                for action in layer:
                    branch = SequentialBlock()
                    node = self._process_action_with_llm(utr, action, deps, current_depth=2)
                    branch.children.append(node)
                    p_block.branches.append(branch)
                root_block.children.append(p_block)
                
        root_block.children.append(ActionSlot(action_id="end_node", action_name="end"))
        return root_block

    def _build_dependency_layers(self, actions: List[Action], deps: list[dict[str, str]]) -> List[List[Action]]:
        """
        基于 implicit_dependencies 进行简单的拓扑分层。
        同一层的 action 互相没有依赖，可以并行执行。
        """
        in_degree = {a.action_id: 0 for a in actions}
        adj_list = {a.action_id: [] for a in actions}
        action_map = {a.action_id: a for a in actions}
        
        for dep in deps:
            from_node = dep.get("from")
            to_node = dep.get("to")
            if from_node in adj_list and to_node in in_degree:
                adj_list[from_node].append(to_node)
                in_degree[to_node] += 1
                    
        layers = []
        queue = [a_id for a_id, deg in in_degree.items() if deg == 0]
        
        while queue:
            current_layer = queue
            layers.append([action_map[a_id] for a_id in current_layer])
            queue = []
            
            for node_id in current_layer:
                for neighbor in adj_list[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
                        
        # 兜底：处理循环依赖导致的未入队节点，防止死循环
        processed = {a.action_id for layer in layers for a in layer}
        unprocessed = [a for a in actions if a.action_id not in processed]
        if unprocessed:
            layers.append(unprocessed)
            
        return layers

    def _wrap_action(self, action: Action) -> BlockType:
        return ActionSlot(action_id=action.action_id, action_name=action.action_name)

    def _process_action_with_llm(self, utr: UTR, action: Action, deps: list[dict[str, str]], current_depth: int) -> BlockType:
        """
        模型端与算法端结合：
        向大模型发起局部查询，判断该动作是否需要条件或循环。
        根据返回的 True/False，算法端负责派生 Block。
        """
        base_node = self._wrap_action(action)
        
        if not self.llm_client or current_depth >= self.max_depth:
            return base_node
            
        # 获取当前动作的上下文依赖，用于传递数据流环境
        in_deps = [d for d in deps if d.get("to") == action.action_id]
        out_deps = [d for d in deps if d.get("from") == action.action_id]
        
        system_prompt = """
你是一个大模型判断器（Judge）。你的任务是根据总任务描述和当前动作的输入输出依赖，判断当前动作是否处于特定条件下，或者是否需要循环执行。
请严格返回如下格式的 JSON 对象，不要包含其他文本：
{
    "is_conditional": false,
    "condition_expr": "",
    "is_loop": false,
    "loop_expr": ""
}
如果 is_conditional 为 true，请在 condition_expr 提供条件表达式（如 "当文件存在时"）。
如果 is_loop 为 true，请在 loop_expr 提供循环表达式（如 "遍历所有文件"）。
"""
        
        user_prompt = f"""
任务描述：{utr.task_desc}

当前分析的动作：
- ID: {action.action_id}
- 名称: {action.action_name}
- 依赖于它的动作 (Input Deps): {json.dumps(in_deps, ensure_ascii=False)}
- 它依赖的动作 (Output Deps): {json.dumps(out_deps, ensure_ascii=False)}

请判断该动作是否需要被条件分支（If）或循环逻辑（Loop）包裹？
"""
        
        try:
            result_dict = self.llm_client.chat_json(system_prompt, user_prompt)
            print(f"\n[LLM Decision for {action.action_id}]: {json.dumps(result_dict, ensure_ascii=False)}")
            
            is_cond = result_dict.get("is_conditional", False)
            cond_expr = result_dict.get("condition_expr", "")
            is_loop = result_dict.get("is_loop", False)
            loop_expr = result_dict.get("loop_expr", "")
            
            current_node = base_node
            
            # 算法端介入：派生 LoopBlock
            if is_loop and loop_expr:
                loop_block = LoopBlock(loop_condition=loop_expr)
                seq = SequentialBlock()
                seq.children.append(current_node)
                loop_block.body = seq
                current_node = loop_block
                
            # 算法端介入：派生 ConditionalBlock
            if is_cond and cond_expr:
                cond_block = ConditionalBlock(condition_description=cond_expr)
                seq = SequentialBlock()
                seq.children.append(current_node)
                cond_block.branches["true"] = seq
                current_node = cond_block
                
            return current_node
            
        except Exception as e:
            print(f"[Warning] LLM validation failed for action {action.action_id}: {e}")
            return base_node
