import importlib.util

from src.core.schema import DifyNodeType


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_module = _load_module(
    "prepare_external_node_mapping_eval_data",
    "scripts/09_prepare_external_node_mapping_eval_data.py",
)
eval_module = _load_module(
    "evaluate_external_node_mapping",
    "scripts/10_evaluate_external_node_mapping.py",
)


def test_infer_inputs_from_http_request_parameters():
    inputs = prepare_module.infer_inputs_from_parameters(
        "n8n-nodes-base.httpRequest",
        {
            "url": "https://api.example.com",
            "method": "POST",
            "jsonBody": "={{ { \"items\": $json.items } }}",
        },
    )

    assert "url" in inputs
    assert "method" in inputs
    assert "items" in inputs


def test_infer_outputs_from_information_extractor():
    outputs = prepare_module.infer_outputs(
        "@n8n/n8n-nodes-langchain.informationExtractor",
        "Information Extractor",
        {"attributes": {"attributes": [{"name": "Invoice Number"}]}},
    )

    assert "parameters" in outputs


def test_infer_expected_degraded_for_tool():
    degraded = prepare_module.infer_expected_degraded(
        DifyNodeType.tool,
        inputs=["query"],
        resources=["google_sheets"],
    )

    assert degraded is True


def test_build_confusion_matrix_for_external_eval():
    prediction = eval_module.NodeMappingEvalPrediction(
        sample_id="sample",
        expected_node_type=DifyNodeType.llm,
        predicted_node_type=DifyNodeType.code,
        correct=False,
    )
    matrix = eval_module.build_confusion_matrix([prediction])

    assert matrix["llm"]["code"] == 1
