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


class DifyNodeType(str, Enum):
    start = "start"
    end = "end"
    llm = "llm"
    code = "code"
    if_else = "if-else"
    iteration = "iteration"
    iteration_start = "iteration-start"
    template_transform = "template-transform"
    http_request = "http-request"
    variable_aggregator = "variable-aggregator"
    tool = "tool"
    doc_extractor = "doc-extractor"
    parameter_extractor = "parameter-extractor"


class MappingConfidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


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
    meta: dict[str, Any]


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



class NodeMappingContext(BaseModel):
    action: Action
    parent_block_type: str = "Sequential"
    upstream_actions: list[str] = Field(default_factory=list)
    downstream_actions: list[str] = Field(default_factory=list)
    available_variables: list[str] = Field(default_factory=list)
    available_resources: list[str] = Field(default_factory=list)


class NodeCandidateScore(BaseModel):
    node_type: DifyNodeType
    rule_score: float = 0.0
    semantic_score: float = 0.0
    coverage_score: float = 0.0
    context_score: float = 0.0
    priority_bonus: float = 0.0
    total_score: float = 0.0


class NodeScoringWeights(BaseModel):
    rule_weight: float = 0.40
    semantic_weight: float = 0.32
    coverage_weight: float = 0.20
    context_weight: float = 0.05
    priority_weight: float = 0.03


class NodeMappingResult(BaseModel):
    source_action_id: str
    chosen_node_type: DifyNodeType
    candidate_node_types: list[DifyNodeType] = Field(default_factory=list)
    confidence: MappingConfidence = MappingConfidence.low
    decision_reason: str = ""
    required_params: list[str] = Field(default_factory=list)
    available_params: dict[str, bool] = Field(default_factory=dict)
    fallback_node_type: DifyNodeType = DifyNodeType.code
    needs_human_fill: bool = False
    degraded: bool = False
    degrade_reason: str = ""
    trace: list[str] = Field(default_factory=list)
    scoring_weights: NodeScoringWeights = Field(default_factory=NodeScoringWeights)
    chosen_score: float = 0.0
    runner_up_score: float = 0.0
    candidate_scores: list[NodeCandidateScore] = Field(default_factory=list)


class NodeMappingEvalSample(BaseModel):
    sample_id: str
    workflow_id: str = ""
    source_node_id: str = ""
    source: str = "dataset"
    split: str = ""
    expected_node_type: DifyNodeType
    action_name: str
    description: str = ""
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    available_resources: list[str] = Field(default_factory=list)
    parent_block_type: str = "Sequential"
    difficulty: str = "standard"
    expected_degraded: bool = False
    tags: list[str] = Field(default_factory=list)
    text_signature: str = ""


class NodeMappingEvalPrediction(BaseModel):
    sample_id: str
    workflow_id: str = ""
    backend: str = ""
    source: str = "dataset"
    split: str = ""
    expected_node_type: DifyNodeType
    predicted_node_type: DifyNodeType
    correct: bool
    expected_degraded: bool = False
    predicted_degraded: bool = False
    confidence: MappingConfidence = MappingConfidence.low
    confidence_score: float = 0.0
    confidence_margin: float = 0.0
    chosen_score: float = 0.0
    runner_up_score: float = 0.0
    seen_in_train: bool = False
    difficulty: str = "standard"
    tags: list[str] = Field(default_factory=list)


class NodeMappingEvalMetrics(BaseModel):
    sample_count: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    degradation_accuracy: Optional[float] = None
    degradation_detection_accuracy: Optional[float] = None
    degradation_type_accuracy: Optional[float] = None
    confidence_ece: Optional[float] = None
    confidence_brier: Optional[float] = None
    confidence_bucket_accuracy: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    seen_accuracy: Optional[float] = None
    unseen_accuracy: Optional[float] = None
    per_label_accuracy: dict[str, float] = Field(default_factory=dict)


class NodeMappingEvalSummary(BaseModel):
    name: str
    backend: str
    sample_count: int = 0
    metrics: NodeMappingEvalMetrics = Field(default_factory=NodeMappingEvalMetrics)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)


class DSLPrecheckIssue(BaseModel):
    code: str
    message: str
    path: str = ""
    severity: str = "error"


class DSLPrecheckReport(BaseModel):
    schema_valid: bool = True
    node_type_valid: bool = True
    action_ref_valid: bool = True
    structure_valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    issues: list[DSLPrecheckIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.schema_valid
            and self.node_type_valid
            and self.action_ref_valid
            and self.structure_valid
        )


class DSLInputContext(BaseModel):
    utr: UTR
    skeleton: SequentialBlock
    action_index: dict[str, Action] = Field(default_factory=dict)
    block_stats: dict[str, int] = Field(default_factory=dict)


class DSLPrecheckOutput(BaseModel):
    context: Optional[DSLInputContext] = None
    report: DSLPrecheckReport


class DSLControlDomain(BaseModel):
    domain_type: str
    block_id: str
    branch: str = ""


class DSLNormalizedActionSlot(BaseModel):
    slot_id: str
    action_id: str
    action_name: str = ""
    parent_block_id: str = ""
    path: str = ""
    control_domains: list[DSLControlDomain] = Field(default_factory=list)
    upstream_slot_ids: list[str] = Field(default_factory=list)
    downstream_slot_ids: list[str] = Field(default_factory=list)


class DSLNormalizedBlock(BaseModel):
    block_id: str
    block_type: str
    parent_block_id: str = ""
    path: str = ""
    child_block_ids: list[str] = Field(default_factory=list)
    entry_slot_ids: list[str] = Field(default_factory=list)
    exit_slot_ids: list[str] = Field(default_factory=list)
    needs_join: bool = False
    has_default_else: bool = False


class DSLNormalizationIssue(BaseModel):
    code: str
    message: str
    path: str = ""
    severity: str = "error"


class DSLNormalizationReport(BaseModel):
    success: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    issues: list[DSLNormalizationIssue] = Field(default_factory=list)


class DSLNormalizedContext(BaseModel):
    utr: UTR
    normalized_skeleton: SequentialBlock
    action_index: dict[str, Action] = Field(default_factory=dict)
    block_stats: dict[str, int] = Field(default_factory=dict)
    root_entry_slot_ids: list[str] = Field(default_factory=list)
    root_exit_slot_ids: list[str] = Field(default_factory=list)
    action_slots: list[DSLNormalizedActionSlot] = Field(default_factory=list)
    blocks: list[DSLNormalizedBlock] = Field(default_factory=list)


class DSLNormalizationOutput(BaseModel):
    precheck_report: DSLPrecheckReport
    normalization_report: DSLNormalizationReport
    context: Optional[DSLNormalizedContext] = None


class DSLCompiledVariable(BaseModel):
    name: str
    value_type: str
    required: bool = False
    value_selector: list[str] = Field(default_factory=list)


class DSLCompiledNode(BaseModel):
    id: str
    node_type: DifyNodeType
    title: str
    desc: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] = Field(default_factory=dict)


class DSLCompiledEdge(BaseModel):
    id: str
    source: str
    target: str
    source_handle: str = "source"
    target_handle: str = "target"
    data: dict[str, Any] = Field(default_factory=dict)


class DSLCompileIssue(BaseModel):
    code: str
    message: str
    path: str = ""
    severity: str = "error"


class DSLCompileReport(BaseModel):
    success: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    issues: list[DSLCompileIssue] = Field(default_factory=list)


class DSLCompiledGraph(BaseModel):
    nodes: list[DSLCompiledNode] = Field(default_factory=list)
    edges: list[DSLCompiledEdge] = Field(default_factory=list)


class DSLCompiledWorkflow(BaseModel):
    app: dict[str, Any]
    workflow: dict[str, Any]
    kind: str = "app"
    version: str = "0.2.0"


class DSLCompileOutput(BaseModel):
    normalization_output: DSLNormalizationOutput
    compile_report: DSLCompileReport
    node_mappings: list[NodeMappingResult] = Field(default_factory=list)
    compiled_graph: Optional[DSLCompiledGraph] = None
    workflow: Optional[DSLCompiledWorkflow] = None


class WorkflowBuildOutput(BaseModel):
    utr_output: PipelineOutput
    skeleton: Optional[SequentialBlock] = None
    dsl_output: Optional[DSLCompileOutput] = None
    success: bool = True
    stage: str = "dsl"
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
