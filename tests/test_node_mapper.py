import pytest

from src.core.schema import Action, DifyNodeType, MappingConfidence
from src.dsl_generation.node_mapper import NodeMapper
from src.dsl_generation.node_mapping_rules import MAPPING_RULES, MAPPING_RULES_BY_NODE_TYPE


def test_map_start_node():
    mapper = NodeMapper()
    action = Action(action_id="start_node", action_name="start")
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.start
    assert result.degraded is False
    assert result.confidence == MappingConfidence.high


def test_map_end_node():
    mapper = NodeMapper()
    action = Action(action_id="end_node", action_name="end")
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.end
    assert result.degraded is False
    assert result.confidence == MappingConfidence.high


def test_map_llm_action():
    mapper = NodeMapper()
    action = Action(
        action_id="act_1",
        action_name="generate_summary",
        description="generate a concise summary for the article",
        inputs=["article_text"],
        outputs=["summary_text"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.llm
    assert result.degraded is False
    assert result.scoring_weights.rule_weight == 0.40
    assert result.candidate_scores
    assert result.candidate_scores[0].node_type == DifyNodeType.llm
    assert result.candidate_scores[0].total_score == result.chosen_score


def test_map_http_request_action_with_params():
    mapper = NodeMapper()
    action = Action(
        action_id="act_2",
        action_name="request_weather_api",
        description="call weather api with get method and url",
        inputs=["url", "method", "city"],
        outputs=["weather_result"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.http_request
    assert result.degraded is False


def test_map_http_request_action_degrade_when_missing_params():
    mapper = NodeMapper()
    action = Action(
        action_id="act_3",
        action_name="request_external_service",
        description="call external api and return data",
        inputs=["payload"],
        outputs=["result"],
    )
    result = mapper.map_action(action)
    assert result.degraded is True
    assert result.chosen_node_type == DifyNodeType.code
    assert result.needs_human_fill is True
    top_score = result.candidate_scores[0]
    assert top_score.node_type == DifyNodeType.http_request
    assert top_score.coverage_score == 0.5
    assert result.chosen_score == top_score.total_score


def test_map_template_transform_action():
    mapper = NodeMapper()
    action = Action(
        action_id="act_4",
        action_name="format_report_template",
        description="format a report with a template",
        inputs=["title", "content"],
        outputs=["formatted_text"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.template_transform
    assert result.degraded is False
    assert result.candidate_scores[0].node_type == DifyNodeType.template_transform
    assert result.candidate_scores[0].coverage_score == 1.0


def test_candidate_scores_are_sorted_and_exposed():
    mapper = NodeMapper()
    action = Action(
        action_id="act_scores",
        action_name="invoke_search_tool",
        description="execute search tool with query",
        inputs=["query"],
        outputs=["result"],
    )
    result = mapper.map_action(action)
    assert result.candidate_scores
    totals = [item.total_score for item in result.candidate_scores]
    assert totals == sorted(totals, reverse=True)
    assert result.candidate_scores[0].node_type in result.candidate_node_types
    assert result.chosen_score >= result.runner_up_score


@pytest.mark.parametrize(
    "action_name,description,inputs,parent_block_type,available_resources,expected,degraded",
    [
        ("生成摘要", "请根据文章内容生成摘要", ["文章内容"], "Sequential", [], DifyNodeType.llm, False),
        ("调用天气接口", "调用天气API，使用url和get方法获取结果", ["url", "method", "城市"], "Sequential", [], DifyNodeType.http_request, False),
        ("套用模板生成通知", "根据模板生成通知文本", ["template", "标题", "内容"], "Sequential", [], DifyNodeType.template_transform, False),
        ("遍历文件列表", "遍历文件列表并逐个处理", ["files"], "Loop", [], DifyNodeType.iteration, False),
        ("条件判断分流", "如果状态异常则进入告警分支", ["状态"], "Conditional", [], DifyNodeType.code, True),
        ("调用搜索工具", "调用搜索工具查询资料", ["query"], "Sequential", ["search_service"], DifyNodeType.http_request, True),
    ],
)
def test_chinese_mapping_examples(
    action_name,
    description,
    inputs,
    parent_block_type,
    available_resources,
    expected,
    degraded,
):
    mapper = NodeMapper()
    action = Action(
        action_id="act_cn",
        action_name=action_name,
        description=description,
        inputs=inputs,
        outputs=["输出"],
    )
    result = mapper.map_action(
        action,
        parent_block_type=parent_block_type,
        available_resources=available_resources,
    )
    assert result.chosen_node_type == expected
    assert result.degraded is degraded


def test_map_aggregator_action():
    mapper = NodeMapper()
    action = Action(
        action_id="act_5",
        action_name="merge_results",
        description="aggregate outputs from multiple branches",
        inputs=["branch_a", "branch_b"],
        outputs=["merged_result"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.variable_aggregator
    assert result.degraded is False


def test_map_default_code_action():
    mapper = NodeMapper()
    action = Action(
        action_id="act_6",
        action_name="custom_internal_step",
        description="",
        inputs=["x"],
        outputs=["y"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.code


def test_mapping_rules_count_and_quality():
    assert len(MAPPING_RULES) >= 500
    assert len(MAPPING_RULES_BY_NODE_TYPE) >= 8
    assert len(MAPPING_RULES_BY_NODE_TYPE[DifyNodeType.llm.value]) >= 300
    assert len(MAPPING_RULES_BY_NODE_TYPE[DifyNodeType.code.value]) >= 300
    assert len(MAPPING_RULES_BY_NODE_TYPE[DifyNodeType.http_request.value]) >= 250
    assert len(MAPPING_RULES_BY_NODE_TYPE[DifyNodeType.iteration.value]) >= 200


@pytest.mark.parametrize(
    "action_name,description,inputs,expected",
    [
        ("generate_report", "generate report content", ["draft"], DifyNodeType.llm),
        ("summarize_document", "summarize the document", ["document_text"], DifyNodeType.llm),
        ("rewrite_email", "rewrite the email content", ["email_draft"], DifyNodeType.llm),
        ("transform_json", "transform json structure", ["raw_json"], DifyNodeType.code),
        ("clean_dataset", "clean the dataset and deduplicate rows", ["dataset"], DifyNodeType.code),
        ("normalize_records", "normalize record fields", ["records"], DifyNodeType.code),
        ("request_inventory_api", "request inventory api with url and get method", ["url", "method"], DifyNodeType.http_request),
        ("fetch_user_api", "call the user api endpoint with url and method", ["url", "method"], DifyNodeType.http_request),
        ("submit_payment_api", "submit a payment api request with url and method", ["url", "method"], DifyNodeType.http_request),
        ("format_notice_template", "format notice output with template", ["template", "notice_text"], DifyNodeType.template_transform),
        ("compose_email_template", "compose email using template", ["template", "email_body"], DifyNodeType.template_transform),
        ("render_report_template", "render report template", ["template", "report_data"], DifyNodeType.template_transform),
        ("use_calendar_tool", "invoke calendar tool", ["query"], DifyNodeType.tool),
        ("invoke_search_tool", "execute search tool", ["query"], DifyNodeType.tool),
        ("call_plugin", "use plugin to finish query", ["query"], DifyNodeType.tool),
        ("extract_contract_pdf", "extract contract pdf text", ["pdf"], DifyNodeType.doc_extractor),
        ("read_document", "read document and parse content", ["document"], DifyNodeType.doc_extractor),
        ("parse_invoice_file", "parse invoice file", ["file"], DifyNodeType.doc_extractor),
        ("extract_intents", "extract user intents and parameters", ["message"], DifyNodeType.parameter_extractor),
        ("classify_entities", "classify entities and fields", ["message"], DifyNodeType.parameter_extractor),
        ("tag_keywords", "tag keywords and categories", ["message"], DifyNodeType.parameter_extractor),
        ("merge_branch_outputs", "aggregate outputs from many branches", ["branch_a", "branch_b"], DifyNodeType.variable_aggregator),
        ("aggregate_results", "aggregate processing results", ["result_a", "result_b"], DifyNodeType.variable_aggregator),
        ("collect_metrics", "collect and merge metrics", ["metric_a", "metric_b"], DifyNodeType.variable_aggregator),
        ("iterate_files", "iterate files one by one", ["files"], DifyNodeType.iteration),
        ("for_each_records", "batch process a list of records", ["records"], DifyNodeType.iteration),
        ("loop_items", "loop through product items", ["items"], DifyNodeType.iteration),
        ("check_condition", "check a condition and branch", ["status"], DifyNodeType.if_else),
        ("route_by_risk_level", "route by risk level", ["risk_level"], DifyNodeType.if_else),
        ("verify_status", "verify status then branch", ["status"], DifyNodeType.if_else),
    ],
)
def test_keyword_mapping_matrix(action_name, description, inputs, expected):
    mapper = NodeMapper()
    action = Action(
        action_id="act_matrix",
        action_name=action_name,
        description=description,
        inputs=inputs,
        outputs=["output_a"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == expected


def test_conflict_rerank_prefers_template_when_intent_is_formatting():
    mapper = NodeMapper()
    action = Action(
        action_id="act_conflict_1",
        action_name="hybrid_mapping_step",
        description="generate report and format template for final output",
        inputs=["title", "body"],
        outputs=["message"],
    )
    result = mapper.map_action(action)
    assert DifyNodeType.llm in result.candidate_node_types
    assert DifyNodeType.template_transform in result.candidate_node_types
    assert result.chosen_node_type == DifyNodeType.template_transform
    assert any(item.startswith("stage2:ranking") for item in result.trace)


def test_conflict_rerank_prefers_llm_when_generation_intent_is_stronger():
    mapper = NodeMapper()
    action = Action(
        action_id="act_conflict_2",
        action_name="generate_polished_response",
        description="generate, rewrite, and explain the final response for the user",
        inputs=["draft_text"],
        outputs=["final_text"],
    )
    result = mapper.map_action(action)
    assert DifyNodeType.llm in result.candidate_node_types
    assert result.chosen_node_type == DifyNodeType.llm
    assert result.confidence in {MappingConfidence.low, MappingConfidence.medium, MappingConfidence.high}


def test_iteration_beats_llm_when_loop_semantics_are_present():
    mapper = NodeMapper()
    action = Action(
        action_id="act_conflict_3",
        action_name="iterate_and_generate_summary",
        description="iterate records and generate a summary for each item in the list",
        inputs=["records"],
        outputs=["summaries"],
    )
    result = mapper.map_action(action)
    assert DifyNodeType.iteration in result.candidate_node_types
    assert DifyNodeType.llm in result.candidate_node_types
    assert result.chosen_node_type == DifyNodeType.iteration


def test_trace_contains_two_stage_markers():
    mapper = NodeMapper()
    action = Action(
        action_id="act_trace",
        action_name="invoke_search_tool",
        description="execute search tool with query",
        inputs=["query"],
        outputs=["result"],
    )
    result = mapper.map_action(action)
    assert any(item.startswith("stage1:") for item in result.trace)
    assert any(item.startswith("stage2:") for item in result.trace)
    assert any("backend=" in item for item in result.trace if item.startswith("stage2:semantic-candidates"))


def test_semantic_retrieval_supports_long_tail_llm_expression():
    mapper = NodeMapper()
    action = Action(
        action_id="act_semantic_llm",
        action_name="craft_customer_facing_copy",
        description="polish the wording, explain the result clearly, and draft a customer-facing reply",
        inputs=["draft_text"],
        outputs=["final_reply"],
    )
    result = mapper.map_action(action)
    assert DifyNodeType.llm in result.candidate_node_types
    assert result.chosen_node_type == DifyNodeType.llm


def test_semantic_retrieval_supports_long_tail_http_expression():
    mapper = NodeMapper()
    action = Action(
        action_id="act_semantic_http",
        action_name="hit_partner_gateway",
        description="reach the external service endpoint with url and post method",
        inputs=["url", "method", "payload"],
        outputs=["gateway_result"],
    )
    result = mapper.map_action(action)
    assert DifyNodeType.http_request in result.candidate_node_types
    assert result.chosen_node_type == DifyNodeType.http_request


@pytest.mark.parametrize(
    "action_name,description,expected_fallback",
    [
        ("request_external_api", "request external api", DifyNodeType.code),
        ("compose_message", "compose formatted notice output", DifyNodeType.code),
        ("process_each_records", "process each record sequentially", DifyNodeType.code),
        ("extract_contract", "", DifyNodeType.code),
        ("extract_entities", "", DifyNodeType.code),
        ("generate_summary", "", DifyNodeType.code),
        ("collect_scores", "collect score results", DifyNodeType.code),
    ],
)
def test_degradation_when_required_params_missing(action_name, description, expected_fallback):
    mapper = NodeMapper()
    action = Action(
        action_id="act_degrade",
        action_name=action_name,
        description=description,
        inputs=["x"],
        outputs=["y"],
    )
    result = mapper.map_action(action)
    assert result.degraded is True
    assert result.needs_human_fill is True
    assert result.chosen_node_type == expected_fallback
    assert result.confidence == MappingConfidence.low


def test_external_style_code_node_beats_iteration_when_js_code_is_present():
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_code_1",
        action_name="Add param1 to output5",
        description=(
            "N8N for Beginners: Looping Over Items. "
            "Most nodes iterate over each item, but this step runs custom code. "
            "$json.param1 = 'x'; return $json"
        ),
        inputs=["jsCode", "items"],
        outputs=["result"],
    )
    result = mapper.map_action(action)
    assert DifyNodeType.iteration in result.candidate_node_types
    assert result.chosen_node_type == DifyNodeType.code
    assert result.degraded is False


def test_external_style_code_node_beats_parameter_extractor_when_regex_code_is_present():
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_code_2",
        action_name="Extracts defined values in better format",
        description=(
            "const allData = $input.all(); "
            "Use regex to extract dividends and parseFloat values, then return result."
        ),
        inputs=["jsCode", "data", "output"],
        outputs=["result"],
    )
    result = mapper.map_action(action)
    assert DifyNodeType.parameter_extractor in result.candidate_node_types
    assert result.chosen_node_type == DifyNodeType.code
    assert result.degraded is False


@pytest.mark.parametrize(
    "action_name,description,inputs,outputs",
    [
        (
            "OpenAI Chat Model",
            "AI-powered WhatsApp chatbot with OpenAI and GPT-4o for answering user questions",
            [],
            ["text"],
        ),
        (
            "AI Agent",
            (
                "You are an intelligent assistant specialized in answering user questions. "
                "Use available tools when needed."
            ),
            ["text", "message", "file", "files"],
            ["text"],
        ),
        (
            "Knowledge Base Agent",
            "RAG chatbot for conversation and contextual answers to user questions",
            ["text", "message"],
            ["text"],
        ),
    ],
)
def test_external_style_llm_platform_expressions_map_to_llm(
    action_name,
    description,
    inputs,
    outputs,
):
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_llm",
        action_name=action_name,
        description=description,
        inputs=inputs,
        outputs=outputs,
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.llm
    assert result.degraded is False


@pytest.mark.parametrize(
    "action_name,inputs,outputs,available_resources",
    [
        (
            "Download File From Google Drive",
            [],
            ["result"],
            ["provider_id", "tool_name", "google_drive", "Download File From Google Drive"],
        ),
        (
            "Send Declined Message",
            ["text"],
            ["result"],
            ["provider_id", "tool_name", "telegram", "Send Declined Message"],
        ),
        (
            "Google Drive",
            ["output", "result", "text"],
            ["result"],
            ["provider_id", "tool_name", "google_drive", "Google Drive"],
        ),
    ],
)
def test_external_style_tool_provider_signals_map_to_tool(
    action_name,
    inputs,
    outputs,
    available_resources,
):
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_tool",
        action_name=action_name,
        description="External app integration step",
        inputs=inputs,
        outputs=outputs,
    )
    result = mapper.map_action(action, available_resources=available_resources)
    assert result.chosen_node_type == DifyNodeType.tool
    assert result.degraded is False


@pytest.mark.parametrize(
    "action_name,description,inputs",
    [
        (
            "Download Voicemail",
            "Service call with url, method, and response payload from a remote endpoint",
            ["url", "method"],
        ),
        (
            "Download Document",
            "Download document from https://example.com/api with response body and headers",
            ["url", "response", "method"],
        ),
        (
            "Get status",
            "Query request_id status from https://queue.fal.run/fal-ai/veo3/requests/{id}/status",
            ["url", "video", "method"],
        ),
    ],
)
def test_external_style_http_service_calls_map_to_http_request(
    action_name,
    description,
    inputs,
):
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_http",
        action_name=action_name,
        description=description,
        inputs=inputs,
        outputs=["response"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.http_request
    assert result.degraded is False


@pytest.mark.parametrize(
    "action_name,description,inputs,outputs",
    [
        (
            "Edit Fields",
            'Edit fields with {"assignments":[{"name":"score","value":"={{ $json.score }}"}]}',
            ["score", "template"],
            ["output", "score"],
        ),
        (
            "Map JSON",
            'Map JSON with {"assignments":[{"name":"data","value":"={{ $json }}"}]}',
            ["data", "template"],
            ["output", "data"],
        ),
    ],
)
def test_external_style_template_assignment_nodes_map_to_template_transform(
    action_name,
    description,
    inputs,
    outputs,
):
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_template",
        action_name=action_name,
        description=description,
        inputs=inputs,
        outputs=outputs,
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.template_transform
    assert result.degraded is False


def test_information_extractor_prefers_parameter_extractor_over_if_else():
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_param",
        action_name="Information Extractor",
        description=(
            "You are a real estate assistant. "
            "Classify the lead quality, extract structured info, and return parameters."
        ),
        inputs=["text", "instruction", "score"],
        outputs=["parameters"],
    )
    result = mapper.map_action(action)
    assert DifyNodeType.if_else in result.candidate_node_types
    assert result.chosen_node_type == DifyNodeType.parameter_extractor
    assert result.degraded is False


def test_external_dify_long_system_prompt_maps_to_llm():
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_dify_llm_1",
        action_name="首次深度初始化",
        description=(
            "基于完整知识库进行深度初始化。你是史蒂夫·乔布斯。不是模拟，不是角色扮演。"
            "核心身份系统、核心价值观、对话原则都已经定义好，请使用第一人称与用户对话。"
            "{{#query#}}"
        ),
        inputs=[],
        outputs=["text"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.llm
    assert result.degraded is False


def test_external_dify_tool_with_provider_fields_maps_to_tool():
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_dify_tool_1",
        action_name="Fetch Single Page",
        description='{"summary": 0, "gather_all_images_at_the_end": 0, "max_retries": 3}',
        inputs=["tool_name", "provider_id", "url"],
        outputs=["result"],
    )
    result = mapper.map_action(
        action,
        available_resources=["jina", "jina_reader", "Fetch Single Page"],
    )
    assert result.chosen_node_type == DifyNodeType.tool
    assert result.degraded is False


def test_external_dify_python_code_beats_parameter_extractor_keywords():
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_dify_code_1",
        action_name="代码执行",
        description=(
            "import requests\nimport json\n"
            "def main(prompt: str, app_id: str, apikey: str) -> dict:\n"
            "    url = f'https://dashscope.aliyuncs.com/api/v1/apps/{app_id}/completion'\n"
            "    headers = {'Authorization': f'Bearer {apikey}'}\n"
            "    return {'result': prompt}\n"
        ),
        inputs=["prompt", "query", "app_id", "apikey"],
        outputs=["result"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.code
    assert result.degraded is False


def test_external_dify_python_code_beats_doc_extractor_semantics():
    mapper = NodeMapper()
    action = Action(
        action_id="act_ext_dify_code_2",
        action_name="代码处理生成html调用",
        description=(
            "import json\nimport re\nimport time\nimport requests\n"
            "def main(json_html: str, apikey: str, apiurl: str, strtype: str) -> dict:\n"
            "    html_content = re.sub(r'^```html\\\\s*|\\\\s*```$', '', json_html)\n"
            "    return {'filename': 'demo.html', 'html_url': apiurl}\n"
        ),
        inputs=["json_html", "html", "apikey", "apiurl", "strtype"],
        outputs=["filename", "html_url", "markdown_result"],
    )
    result = mapper.map_action(action)
    assert result.chosen_node_type == DifyNodeType.code
    assert result.degraded is False
