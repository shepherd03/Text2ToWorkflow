from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResourceType(str, Enum):
    data = "data"
    file = "file"
    service = "service"
    target = "target"
    variable = "variable"
    tool = "tool"


class ControlIntentType(str, Enum):
    sequential = "sequential"
    conditional = "conditional"
    parallel = "parallel"
    iteration = "iteration"


class VariableType(str, Enum):
    string = "string"
    number = "number"
    boolean = "boolean"
    list = "list"
    object = "object"


class Action(BaseModel):
    action_id: str = ""
    action_name: str
    description: str = ""
    order: int
    args: dict[str, Any] = Field(default_factory=dict)


class Resource(BaseModel):
    resource_id: str = ""
    name: str
    type: ResourceType
    description: str = ""


class ControlIntent(BaseModel):
    intent_id: str = ""
    type: ControlIntentType
    condition: str = ""
    target_actions: list[str] = Field(default_factory=list)
    loop_target: str = ""
    loop_condition: str = ""


class Variable(BaseModel):
    var_id: str = ""
    name: str
    type: VariableType
    value: Any = None
    source: str = ""


class UTR(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)
    control_intents: list[ControlIntent] = Field(default_factory=list)
    variables: list[Variable] = Field(default_factory=list)


class UTRValidationReport(BaseModel):
    schema_valid: bool
    logic_valid: bool
    completeness_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.schema_valid and self.logic_valid and self.completeness_valid
