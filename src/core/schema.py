import uuid
from enum import Enum
from typing import Any, List, Union, Optional

from pydantic import BaseModel, Field


class ResourceType(str, Enum):
    data = "data"
    file = "file"
    service = "service"
    target = "target"
    variable = "variable"
    tool = "tool"


class VariableType(str, Enum):
    string = "string"
    number = "number"
    boolean = "boolean"
    list = "list"
    object = "object"
    file = "file"
    data = "data"


class Action(BaseModel):
    action_id: str = Field(default="", description="动作的唯一标识符")
    action_name: str = Field(description="动作的名称")
    description: str = Field(default="", description="动作的具体描述")
    inputs: list[str] = Field(default_factory=list, description="该动作所需的变量或资源名称列表")
    outputs: list[str] = Field(default_factory=list, description="该动作产生的变量或资源名称列表")

class Resource(BaseModel):
    resource_id: str = ""
    name: str
    type: ResourceType
    description: str = ""

class Variable(BaseModel):
    var_id: str = ""
    name: str
    type: VariableType
    source: str = ""

class UTRMetadata(BaseModel):
    task_goal: str = Field(default="", description="任务的总体目标")
    core_actions: list[Action] = Field(default_factory=list, description="任务涉及的核心动作/意图")
    core_resources: list[Resource] = Field(default_factory=list, description="任务涉及的关键资源")
    core_variables: list[Variable] = Field(default_factory=list, description="任务涉及的关键变量")
    implicit_dependencies: list[dict[str, str]] = Field(default_factory=list, description="自然语言中暗示的依赖关系，如 [{'from': 'action_A', 'to': 'action_B', 'reason': '...'}]")

class UTR(BaseModel):
    task_id: str = Field(description="任务唯一标识")
    task_desc: str = Field(description="原始任务描述")
    metadata: UTRMetadata = Field(description="提取的基础元数据")
    create_time: str = Field(description="生成时间")


class UTRValidationReport(BaseModel):
    schema_valid: bool
    logic_valid: bool
    completeness_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.schema_valid and self.logic_valid and self.completeness_valid

class PipelineOutput(BaseModel):
    utr: UTR
    report: UTRValidationReport
    meta: dict[str, str | bool]


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

