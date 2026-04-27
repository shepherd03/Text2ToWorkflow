from src.core.schema import Action, ActionSlot, ConditionalBlock, LoopBlock, ParallelBlock, SequentialBlock, UTR, UTRMetadata, Variable, VariableType
from src.dsl_generation.pipeline import DSLGenerationPipeline
from src.dsl_generation.workflow_validator import DifyWorkflowValidator


def _build_sample_utr() -> UTR:
    return UTR(
        task_id="task_dsl_compile_1",
        task_desc="读取文章并生成摘要",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="生成文章摘要",
            core_actions=[
                Action(action_id="act_1", action_name="generate_summary", description="根据文章内容生成摘要文本", inputs=["article_text"], outputs=["summary_text"]),
                Action(action_id="act_2", action_name="transform_json", description="转换结构化结果", inputs=["summary_text"], outputs=["normalized_result"]),
            ],
            core_variables=[Variable(var_id="var_1", name="article_text", type=VariableType.string, source="user")],
        ),
    )


def _build_valid_skeleton() -> SequentialBlock:
    return SequentialBlock(children=[ActionSlot(action_id="start_node", action_name="start"), ActionSlot(action_id="act_1", action_name="generate_summary"), ActionSlot(action_id="end_node", action_name="end")])


def _build_two_action_skeleton() -> SequentialBlock:
    return SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ActionSlot(action_id="act_1", action_name="generate_summary"),
            ActionSlot(action_id="act_2", action_name="transform_json"),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )


def _build_same_name_upstream_utr() -> UTR:
    return UTR(
        task_id="task_dsl_same_name_upstream_1",
        task_desc="rewrite and consume query",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="ensure transformed query flows downstream",
            core_actions=[
                Action(action_id="act_rewrite", action_name="rewrite_query", description="rewrite query", inputs=["query"], outputs=["query"]),
                Action(action_id="act_search", action_name="search_with_query", description="search with rewritten query", inputs=["query"], outputs=["search_result"]),
            ],
            core_variables=[Variable(var_id="var_query", name="query", type=VariableType.string, source="user")],
        ),
    )


def _build_same_name_upstream_skeleton() -> SequentialBlock:
    return SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ActionSlot(action_id="act_rewrite", action_name="rewrite_query"),
            ActionSlot(action_id="act_search", action_name="search_with_query"),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )


def _build_unrelated_upstream_utr() -> UTR:
    return UTR(
        task_id="task_dsl_unrelated_upstream_1",
        task_desc="summarize then request by city",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="keep independent start inputs available",
            core_actions=[
                Action(action_id="act_summary", action_name="generate_summary", description="summarize text", inputs=["article_text"], outputs=["summary_text"]),
                Action(action_id="act_weather", action_name="request_weather_api", description="call weather api with get method and url", inputs=["city"], outputs=["weather_result"]),
            ],
            core_variables=[
                Variable(var_id="var_article_text", name="article_text", type=VariableType.string, source="user"),
                Variable(var_id="var_city", name="city", type=VariableType.string, source="user"),
            ],
        ),
    )


def _build_unrelated_upstream_skeleton() -> SequentialBlock:
    return SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ActionSlot(action_id="act_summary", action_name="generate_summary"),
            ActionSlot(action_id="act_weather", action_name="request_weather_api"),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )


def _build_conditional_skeleton() -> SequentialBlock:
    return SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ConditionalBlock(
                condition_description="根据输入内容决定走哪个分支",
                branches={
                    "true": SequentialBlock(children=[ActionSlot(action_id="act_1", action_name="generate_summary")]),
                    "else": SequentialBlock(children=[ActionSlot(action_id="act_2", action_name="transform_json")]),
                },
            ),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )


def _build_conditional_then_action_skeleton() -> SequentialBlock:
    return SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ConditionalBlock(
                condition_description="branch by status",
                branches={
                    "true": SequentialBlock(children=[ActionSlot(action_id="act_1", action_name="generate_summary")]),
                    "else": SequentialBlock(children=[ActionSlot(action_id="act_2", action_name="transform_json")]),
                },
            ),
            ActionSlot(action_id="act_3", action_name="format_final_result"),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )


def _build_parallel_skeleton() -> SequentialBlock:
    return SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ParallelBlock(
                branches=[
                    SequentialBlock(children=[ActionSlot(action_id="act_1", action_name="generate_summary")]),
                    SequentialBlock(children=[ActionSlot(action_id="act_2", action_name="transform_json")]),
                ]
            ),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )


def _build_loop_skeleton() -> SequentialBlock:
    return SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            LoopBlock(
                loop_condition="遍历输入列表并生成摘要",
                body=SequentialBlock(children=[ActionSlot(action_id="act_1", action_name="generate_summary")]),
            ),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )


def _build_http_request_utr() -> UTR:
    return UTR(
        task_id="task_dsl_http_1",
        task_desc="调用天气接口并返回结果",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="调用外部 API",
            core_actions=[
                Action(
                    action_id="act_http",
                    action_name="request_weather_api",
                    description="调用 weather api，使用 get method 和 url",
                    inputs=["url", "method", "city"],
                    outputs=["weather_result"],
                )
            ],
            core_variables=[
                Variable(var_id="var_url", name="url", type=VariableType.string, source="user"),
                Variable(var_id="var_method", name="method", type=VariableType.string, source="user"),
                Variable(var_id="var_city", name="city", type=VariableType.string, source="user"),
            ],
        ),
    )


def _build_template_transform_utr() -> UTR:
    return UTR(
        task_id="task_dsl_template_1",
        task_desc="格式化通知消息",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="根据模板组织输出",
            core_actions=[
                Action(
                    action_id="act_tpl",
                    action_name="format_notice_template",
                    description="使用模板格式化通知输出",
                    inputs=["title", "content"],
                    outputs=["formatted_text"],
                )
            ],
            core_variables=[
                Variable(var_id="var_title", name="title", type=VariableType.string, source="user"),
                Variable(var_id="var_content", name="content", type=VariableType.string, source="user"),
            ],
        ),
    )


def _build_conditional_then_action_utr() -> UTR:
    return UTR(
        task_id="task_dsl_conditional_then_action_1",
        task_desc="branch and format",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="format merged branch output",
            core_actions=[
                Action(action_id="act_1", action_name="generate_summary", description="summarize article", inputs=["article_text"], outputs=["summary_text"]),
                Action(action_id="act_2", action_name="transform_json", description="convert structured result", inputs=["article_text"], outputs=["normalized_result"]),
                Action(action_id="act_3", action_name="format_final_result", description="format branch result with template", inputs=["branch_result"], outputs=["final_text"]),
            ],
            core_variables=[
                Variable(var_id="var_article_text", name="article_text", type=VariableType.string, source="user"),
                Variable(var_id="var_status", name="status", type=VariableType.string, source="user"),
            ],
        ),
    )


def _build_tool_utr() -> UTR:
    return UTR(
        task_id="task_dsl_tool_1",
        task_desc="调用搜索工具查询信息",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="使用工具完成查询",
            core_actions=[
                Action(
                    action_id="act_tool",
                    action_name="invoke_search_tool",
                    description="执行搜索工具",
                    inputs=["query"],
                    outputs=["search_result"],
                )
            ],
            core_variables=[
                Variable(var_id="var_query", name="query", type=VariableType.string, source="user"),
            ],
        ),
    )


def _build_doc_extractor_utr() -> UTR:
    return UTR(
        task_id="task_dsl_doc_1",
        task_desc="读取 PDF 文档并抽取正文",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="读取文档正文",
            core_actions=[
                Action(
                    action_id="act_doc",
                    action_name="extract_contract_pdf",
                    description="extract contract pdf text",
                    inputs=["pdf"],
                    outputs=["document_text"],
                )
            ],
            core_variables=[Variable(var_id="var_pdf", name="pdf", type=VariableType.file, source="user")],
        ),
    )


def _build_parameter_extractor_utr() -> UTR:
    return UTR(
        task_id="task_dsl_param_1",
        task_desc="从用户消息中抽取意图和实体",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="抽取结构化参数",
            core_actions=[
                Action(
                    action_id="act_param",
                    action_name="extract_intents",
                    description="extract user intents and parameters",
                    inputs=["message"],
                    outputs=["parameters"],
                )
            ],
            core_variables=[Variable(var_id="var_message", name="message", type=VariableType.string, source="user")],
        ),
    )


def _build_aggregator_utr() -> UTR:
    return UTR(
        task_id="task_dsl_aggregator_1",
        task_desc="合并多个分支输出",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="聚合结果",
            core_actions=[
                Action(
                    action_id="act_agg",
                    action_name="merge_branch_outputs",
                    description="aggregate outputs from many branches",
                    inputs=["branch_a", "branch_b"],
                    outputs=["merged_result"],
                )
            ],
            core_variables=[
                Variable(var_id="var_branch_a", name="branch_a", type=VariableType.object, source="upstream"),
                Variable(var_id="var_branch_b", name="branch_b", type=VariableType.object, source="upstream"),
            ],
        ),
    )


def _build_if_else_action_utr() -> UTR:
    return UTR(
        task_id="task_dsl_if_action_1",
        task_desc="根据状态判断下一步",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="条件判断",
            core_actions=[
                Action(
                    action_id="act_if",
                    action_name="check_condition",
                    description="check status and branch when status is high",
                    inputs=["conditions", "status"],
                    outputs=["route_result"],
                )
            ],
            core_variables=[
                Variable(var_id="var_conditions", name="conditions", type=VariableType.object, source="user"),
                Variable(var_id="var_status", name="status", type=VariableType.string, source="user"),
            ],
        ),
    )


def _build_iteration_action_utr() -> UTR:
    return UTR(
        task_id="task_dsl_iteration_action_1",
        task_desc="遍历文件并处理",
        create_time="2026-04-07T00:00:00Z",
        metadata=UTRMetadata(
            task_goal="遍历文件",
            core_actions=[
                Action(
                    action_id="act_iter",
                    action_name="iterate_files",
                    description="iterate files one by one",
                    inputs=["files"],
                    outputs=["processed_files"],
                )
            ],
            core_variables=[Variable(var_id="var_files", name="files", type=VariableType.list, source="user")],
        ),
    )


def _build_single_action_skeleton(action_id: str, action_name: str) -> SequentialBlock:
    return SequentialBlock(
        children=[
            ActionSlot(action_id="start_node", action_name="start"),
            ActionSlot(action_id=action_id, action_name=action_name),
            ActionSlot(action_id="end_node", action_name="end"),
        ]
    )


def test_run_step3_minimal_success():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_valid_skeleton())
    assert output.compile_report.success is True
    assert output.workflow is not None
    assert output.compiled_graph is not None
    assert len(output.compiled_graph.nodes) == 3
    assert len(output.compiled_graph.edges) == 2
    assert [node.node_type.value for node in output.compiled_graph.nodes] == ["start", "llm", "end"]


def test_run_step3_minimal_start_variables_exposed():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_valid_skeleton())
    start_node = output.compiled_graph.nodes[0]
    assert start_node.node_type.value == "start"
    assert len(start_node.data["variables"]) == 1
    assert start_node.data["variables"][0]["variable"] == "article_text"


def test_run_step3_minimal_end_outputs_link_previous_node():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_valid_skeleton())
    end_node = output.compiled_graph.nodes[-1]
    assert end_node.node_type.value == "end"
    assert len(end_node.data["outputs"]) == 1
    assert end_node.data["outputs"][0]["variable"] == "summary_text"
    assert end_node.data["outputs"][0]["value_selector"] == ["node_act_1", "summary_text"]


def test_sequential_action_input_binds_to_previous_action_output():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_two_action_skeleton())
    assert output.compile_report.success is True

    second_action_node = output.compiled_graph.nodes[2]
    assert second_action_node.id == "node_act_2"
    assert second_action_node.data["variables"][0] == {
        "variable": "summary_text",
        "value_selector": ["node_act_1", "summary_text"],
    }


def test_upstream_output_takes_precedence_over_same_named_start_variable():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_same_name_upstream_utr(),
        _build_same_name_upstream_skeleton(),
    )
    assert output.compile_report.success is True

    search_node = next(node for node in output.compiled_graph.nodes if node.id == "node_act_search")
    assert search_node.data["variables"][0] == {
        "variable": "query",
        "value_selector": ["node_act_rewrite", "query"],
    }


def test_unrelated_start_input_does_not_bind_to_previous_single_output():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_unrelated_upstream_utr(),
        _build_unrelated_upstream_skeleton(),
    )
    assert output.compile_report.success is True

    weather_node = next(node for node in output.compiled_graph.nodes if node.id == "node_act_weather")
    assert weather_node.data["variables"][0] == {
        "variable": "city",
        "value_selector": ["start", "city"],
    }


def test_minimal_workflow_matches_basic_dify_constraints():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_valid_skeleton())
    validator = DifyWorkflowValidator()
    validated = validator.validate(output)
    assert validated.compile_report.success is True
    assert validated.workflow.version == "0.2.0"
    assert validated.workflow.kind == "app"
    assert validated.workflow.app["mode"] == "workflow"


def test_workflow_validator_checks_template_selector_references():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_tool_utr(), _build_single_action_skeleton("act_tool", "invoke_search_tool"))
    assert output.compile_report.success is True

    tool_node = output.workflow.workflow["graph"]["nodes"][1]
    tool_node["data"]["tool_parameters"]["query"]["value"] = "{{#missing_node.result#}}"
    output.compile_report.success = True
    output.compile_report.errors = []

    validated = DifyWorkflowValidator().validate(output)

    assert validated.compile_report.success is False
    assert "Selector source not found" in validated.compile_report.errors[0]


def test_conditional_block_compiles_to_if_else_graph():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_conditional_skeleton())
    assert output.compile_report.success is True
    assert output.compiled_graph is not None
    node_types = [node.node_type.value for node in output.compiled_graph.nodes]
    assert "if-else" in node_types
    if_else_node = next(node for node in output.compiled_graph.nodes if node.node_type.value == "if-else")
    handles = {edge.source_handle for edge in output.compiled_graph.edges if edge.source == if_else_node.id}
    case_ids = {case["case_id"] for case in if_else_node.data["cases"]}
    assert handles == case_ids | {"false"}
    assert node_types.count("start") == 1
    assert node_types.count("end") == 1
    assert len(output.compiled_graph.edges) >= 4


def test_conditional_end_binds_to_join_output():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_conditional_skeleton())
    end_node = output.compiled_graph.nodes[-1]
    assert end_node.node_type.value == "end"
    assert end_node.data["outputs"][0]["value_selector"][0].startswith("conditional_join_")
    assert end_node.data["outputs"][0]["value_selector"][1] == "result"


def test_action_after_conditional_binds_to_conditional_join_output():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_conditional_then_action_utr(),
        _build_conditional_then_action_skeleton(),
    )
    assert output.compile_report.success is True

    final_action_node = next(node for node in output.compiled_graph.nodes if node.id == "node_act_3")
    selector = final_action_node.data["variables"][0]["value_selector"]
    assert selector[0].startswith("conditional_join_")
    assert selector[1] == "result"


def test_parallel_block_compiles_to_join_graph():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_parallel_skeleton())
    assert output.compile_report.success is True
    assert output.compiled_graph is not None
    node_types = [node.node_type.value for node in output.compiled_graph.nodes]
    assert "variable-aggregator" in node_types
    join_nodes = [node for node in output.compiled_graph.nodes if node.node_type.value == "variable-aggregator"]
    assert len(join_nodes) == 1
    assert join_nodes[0].data["output_type"] == "object"
    assert node_types.count("start") == 1
    assert node_types.count("end") == 1
    assert len(output.compiled_graph.edges) >= 4


def test_loop_block_compiles_to_iteration_graph():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_loop_skeleton())
    assert output.compile_report.success is True
    assert output.compiled_graph is not None
    node_types = [node.node_type.value for node in output.compiled_graph.nodes]
    assert "iteration" in node_types
    assert "iteration-start" in node_types
    iteration_nodes = [node for node in output.compiled_graph.nodes if node.node_type.value == "iteration"]
    assert len(iteration_nodes) == 1
    iteration_node = iteration_nodes[0]
    assert iteration_node.data["iterator_selector"] == ["start", "article_text"]
    assert iteration_node.data["start_node_id"] == f"{iteration_node.id}_start"
    assert iteration_node.data["output_selector"] == ["node_act_1", "summary_text"]
    assert iteration_node.data["output_type"] == "array[object]"

    iteration_start_node = next(node for node in output.compiled_graph.nodes if node.node_type.value == "iteration-start")
    assert iteration_start_node.id == iteration_node.data["start_node_id"]
    assert iteration_start_node.data["isInIteration"] is True

    body_node = next(node for node in output.compiled_graph.nodes if node.id == "node_act_1")
    assert body_node.data["isInIteration"] is True
    assert body_node.data["iteration_id"] == iteration_node.id
    assert body_node.data["isIterationStart"] is True
    assert body_node.data["variables"][0]["value_selector"] == [iteration_node.id, "item"]

    loop_join = next(node for node in output.compiled_graph.nodes if node.id.startswith("loop_join_"))
    assert loop_join.data["variables"][0]["value_selector"] == [iteration_node.id, "output"]

    internal_edge = next(edge for edge in output.compiled_graph.edges if edge.source == iteration_start_node.id)
    assert internal_edge.target == body_node.id
    assert internal_edge.data["isInIteration"] is True
    assert internal_edge.data["iteration_id"] == iteration_node.id
    assert node_types.count("start") == 1
    assert node_types.count("end") == 1
    assert len(output.compiled_graph.edges) >= 4


def test_loop_workflow_matches_dify_iteration_constraints():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(_build_sample_utr(), _build_loop_skeleton())

    validated = DifyWorkflowValidator().validate(output)

    assert validated.compile_report.success is True


def test_http_request_action_compiles_to_http_request_node():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_http_request_utr(),
        _build_single_action_skeleton("act_http", "request_weather_api"),
    )
    assert output.compile_report.success is True
    node = output.compiled_graph.nodes[1]
    assert node.node_type.value == "http-request"
    assert node.data["method"] == "get"
    assert node.data["authorization"] == {"type": "no-auth", "config": None}
    assert node.data["timeout"]["max_read_timeout"] == 30
    assert node.data["url_selector"] == ["start", "url"]
    assert "weather_result" in node.data["outputs"]


def test_template_transform_action_compiles_to_template_node():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_template_transform_utr(),
        _build_single_action_skeleton("act_tpl", "format_notice_template"),
    )
    assert output.compile_report.success is True
    node = output.compiled_graph.nodes[1]
    assert node.node_type.value == "template-transform"
    assert "title: {{title}}" in node.data["template"]
    assert "formatted_text" in node.data["outputs"]


def test_tool_action_compiles_to_tool_node():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_tool_utr(),
        _build_single_action_skeleton("act_tool", "invoke_search_tool"),
    )
    assert output.compile_report.success is True
    node = output.compiled_graph.nodes[1]
    assert node.node_type.value == "tool"
    assert node.data["provider_id"] == "search"
    assert node.data["provider_type"] == "builtin"
    assert node.data["tool_name"] == "search"
    assert "query" in node.data["tool_parameters"]


def test_doc_extractor_action_compiles_to_doc_extractor_node():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_doc_extractor_utr(),
        _build_single_action_skeleton("act_doc", "extract_contract_pdf"),
    )
    assert output.compile_report.success is True
    node = output.compiled_graph.nodes[1]
    assert node.node_type.value == "doc-extractor"
    assert node.data["type"] == "doc-extractor"
    assert node.data["variable_selector"] == ["start", "pdf"]
    assert "document_text" in node.data["outputs"]


def test_parameter_extractor_action_compiles_to_parameter_extractor_node():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_parameter_extractor_utr(),
        _build_single_action_skeleton("act_param", "extract_intents"),
    )
    assert output.compile_report.success is True
    node = output.compiled_graph.nodes[1]
    assert node.node_type.value == "parameter-extractor"
    assert node.data["type"] == "parameter-extractor"
    assert node.data["instruction"] == "extract user intents and parameters"
    assert node.data["parameters"][0]["name"] == "parameters"


def test_aggregator_action_compiles_to_variable_aggregator_node():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_aggregator_utr(),
        _build_single_action_skeleton("act_agg", "merge_branch_outputs"),
    )
    assert output.compile_report.success is True
    node = output.compiled_graph.nodes[1]
    assert node.node_type.value == "variable-aggregator"
    assert node.data["type"] == "variable-aggregator"
    assert node.data["output_type"] == "object"
    assert "merged_result" in node.data["outputs"]


def test_if_else_action_compiles_to_if_else_data_not_code_payload():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_if_else_action_utr(),
        _build_single_action_skeleton("act_if", "check_condition"),
    )
    assert output.compile_report.success is True
    node = output.compiled_graph.nodes[1]
    assert node.node_type.value == "if-else"
    assert node.data["type"] == "if-else"
    assert node.data["cases"][0]["conditions"][0]["variable_selector"] == ["start", "status"]


def test_iteration_action_compiles_to_iteration_data_not_code_payload():
    pipeline = DSLGenerationPipeline()
    output = pipeline.run_step3_minimal(
        _build_iteration_action_utr(),
        _build_single_action_skeleton("act_iter", "iterate_files"),
    )
    assert output.compile_report.success is True
    node = output.compiled_graph.nodes[1]
    assert node.node_type.value == "iteration"
    assert node.data["type"] == "iteration"
    assert node.data["iterator_selector"] == ["start", "files"]
