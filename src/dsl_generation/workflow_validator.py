from __future__ import annotations

import re

from src.core.schema import DSLCompileOutput


class DifyWorkflowValidator:
    _selector_pattern = re.compile(r"\{\{#([^#]+)#\}\}")
    _allowed_node_types = {
        "start",
        "end",
        "llm",
        "code",
        "if-else",
        "iteration",
        "iteration-start",
        "template-transform",
        "http-request",
        "variable-aggregator",
        "tool",
        "doc-extractor",
        "parameter-extractor",
    }

    def validate(self, compile_output: DSLCompileOutput) -> DSLCompileOutput:
        workflow = compile_output.workflow
        report = compile_output.compile_report
        if workflow is None or compile_output.compiled_graph is None:
            report.success = False
            report.errors.append("Missing workflow or compiled_graph for validation")
            return compile_output

        graph = workflow.workflow.get("graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        if workflow.version != "0.2.0":
            report.success = False
            report.errors.append("Dify workflow version must be 0.2.0")
        if workflow.kind != "app":
            report.success = False
            report.errors.append("Dify kind must be app")
        if workflow.app.get("mode") != "workflow":
            report.success = False
            report.errors.append("app.mode must be workflow")

        start_count = 0
        end_count = 0
        node_ids: set[str] = set()

        for node in nodes:
            node_id = node.get("id")
            data = node.get("data", {})
            node_type = data.get("type")
            if not node_id:
                report.success = False
                report.errors.append("Node is missing id")
                continue
            if node_id in node_ids:
                report.success = False
                report.errors.append(f"Duplicate node id: {node_id}")
            node_ids.add(node_id)

            if node_type not in self._allowed_node_types:
                report.success = False
                report.errors.append(f"Unsupported node type: {node_type}")
                continue

            if node_type == "start":
                start_count += 1
                self._validate_start_node(data, report)
            elif node_type == "end":
                end_count += 1
                self._validate_end_node(data, report)
            elif node_type == "llm":
                self._validate_llm_node(data, report)
            elif node_type == "code":
                self._validate_code_node(data, report)
            elif node_type == "if-else":
                self._validate_if_else_node(data, report)
            elif node_type == "iteration":
                self._validate_iteration_node(data, report)
            elif node_type == "iteration-start":
                self._validate_iteration_start_node(data, report)
            elif node_type == "variable-aggregator":
                self._validate_variable_aggregator_node(data, report)
            elif node_type == "http-request":
                self._validate_http_request_node(data, report)
            elif node_type == "template-transform":
                self._validate_template_transform_node(data, report)
            elif node_type == "tool":
                self._validate_tool_node(data, report)
            elif node_type == "doc-extractor":
                self._validate_doc_extractor_node(data, report)
            elif node_type == "parameter-extractor":
                self._validate_parameter_extractor_node(data, report)

        if start_count != 1:
            report.success = False
            report.errors.append("Workflow must contain exactly one start node")
        if end_count != 1:
            report.success = False
            report.errors.append("Workflow must contain exactly one end node")

        output_index = self._build_output_index(nodes)
        for node in nodes:
            node_id = node.get("id", "")
            self._validate_node_selectors(node_id, node.get("data", {}), output_index, report)

        if_else_handles = self._build_if_else_handle_index(nodes)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source not in node_ids:
                report.success = False
                report.errors.append(f"Edge source not found: {source}")
            if target not in node_ids:
                report.success = False
                report.errors.append(f"Edge target not found: {target}")
            edge_data = edge.get("data", {})
            if "sourceType" not in edge_data or "targetType" not in edge_data:
                report.success = False
                report.errors.append("Edge data must contain sourceType and targetType")
            if source in if_else_handles and edge.get("sourceHandle") != "source":
                source_handle = edge.get("sourceHandle")
                if source_handle not in if_else_handles[source]:
                    report.success = False
                    report.errors.append(f"if-else edge uses unknown sourceHandle: {source_handle}")

        return compile_output

    def _build_output_index(self, nodes: list[dict]) -> dict[str, set[str]]:
        output_index: dict[str, set[str]] = {}
        for node in nodes:
            node_id = node.get("id")
            data = node.get("data", {})
            if not node_id:
                continue
            node_type = data.get("type")
            fields: set[str] = set()
            if node_type == "start":
                fields.update(
                    item.get("variable")
                    for item in data.get("variables", [])
                    if isinstance(item, dict) and item.get("variable")
                )
            if node_type == "iteration":
                fields.update({"item", "output"})
                output_selector = data.get("output_selector")
                if isinstance(output_selector, list) and len(output_selector) == 1:
                    fields.add(str(output_selector[0]))
            outputs = data.get("outputs")
            if isinstance(outputs, dict):
                fields.update(str(output_name) for output_name in outputs.keys())
            elif isinstance(outputs, list):
                fields.update(
                    item.get("variable")
                    for item in outputs
                    if isinstance(item, dict) and item.get("variable")
                )
            output_index[str(node_id)] = fields
        return output_index

    def _build_if_else_handle_index(self, nodes: list[dict]) -> dict[str, set[str]]:
        handle_index: dict[str, set[str]] = {}
        for node in nodes:
            node_id = node.get("id")
            data = node.get("data", {})
            if not node_id or data.get("type") != "if-else":
                continue
            handles = {
                str(case.get("case_id"))
                for case in data.get("cases", [])
                if isinstance(case, dict) and case.get("case_id")
            }
            handles.add("false")
            handle_index[str(node_id)] = handles
        return handle_index

    def _validate_node_selectors(
        self,
        node_id: str,
        data: dict,
        output_index: dict[str, set[str]],
        report,
    ) -> None:
        seen_selectors: set[tuple[str, ...]] = set()
        for selector in self._collect_value_selectors(data):
            selector_key = tuple(selector)
            if selector_key in seen_selectors:
                continue
            seen_selectors.add(selector_key)
            if len(selector) < 2:
                report.success = False
                report.errors.append(f"Invalid value selector on node {node_id}: {selector}")
                continue
            source_id = str(selector[0])
            field_name = str(selector[1])
            if source_id in {"sys", "conversation", "env"}:
                continue
            if source_id not in output_index:
                report.success = False
                report.errors.append(f"Selector source not found on node {node_id}: {selector}")
                continue
            source_fields = output_index[source_id]
            if source_fields and field_name not in source_fields:
                report.success = False
                report.errors.append(f"Selector field not found on node {node_id}: {selector}")

    def _collect_value_selectors(self, value) -> list[list[str]]:
        selectors: list[list[str]] = []
        if isinstance(value, dict):
            selector = value.get("value_selector")
            if isinstance(selector, list):
                selectors.append([str(item) for item in selector])
            for nested in value.values():
                selectors.extend(self._collect_value_selectors(nested))
            return selectors
        if isinstance(value, list):
            if len(value) >= 2 and all(isinstance(item, str) for item in value):
                selectors.append([str(item) for item in value])
            else:
                for nested in value:
                    selectors.extend(self._collect_value_selectors(nested))
        if isinstance(value, str):
            for match in self._selector_pattern.finditer(value):
                selector = [part for part in match.group(1).split(".") if part]
                if selector:
                    selectors.append(selector)
        return selectors

    def _validate_start_node(self, data: dict, report) -> None:
        variables = data.get("variables")
        if not isinstance(variables, list):
            report.success = False
            report.errors.append("start node variables must be a list")
            return
        for item in variables:
            for key in ["label", "variable", "type", "required"]:
                if key not in item:
                    report.success = False
                    report.errors.append(f"start node variable missing field: {key}")

    def _validate_end_node(self, data: dict, report) -> None:
        outputs = data.get("outputs")
        if not isinstance(outputs, list):
            report.success = False
            report.errors.append("end node outputs must be a list")
            return
        for item in outputs:
            if "variable" not in item or "value_selector" not in item:
                report.success = False
                report.errors.append("end node output must include variable and value_selector")

    def _validate_llm_node(self, data: dict, report) -> None:
        model = data.get("model")
        if not isinstance(model, dict):
            report.success = False
            report.errors.append("llm node missing model")
        else:
            for key in ["mode", "name", "provider", "completion_params"]:
                if key not in model:
                    report.success = False
                    report.errors.append(f"llm.model missing field: {key}")
        if not isinstance(data.get("prompt_template"), list):
            report.success = False
            report.errors.append("llm node missing prompt_template list")
        if not isinstance(data.get("variables"), list):
            report.success = False
            report.errors.append("llm node missing variables list")
        if not isinstance(data.get("context"), dict):
            report.success = False
            report.errors.append("llm node missing context object")
        if not isinstance(data.get("vision"), dict):
            report.success = False
            report.errors.append("llm node missing vision object")

    def _validate_code_node(self, data: dict, report) -> None:
        if "code_language" not in data:
            report.success = False
            report.errors.append("code node missing code_language")
        if "code" not in data:
            report.success = False
            report.errors.append("code node missing code")
        outputs = data.get("outputs")
        if not isinstance(outputs, dict):
            report.success = False
            report.errors.append("code node missing outputs dict")
            return
        for output_name, output_schema in outputs.items():
            if not isinstance(output_schema, dict) or "type" not in output_schema:
                report.success = False
                report.errors.append(f"invalid code output schema: {output_name}")

    def _validate_if_else_node(self, data: dict, report) -> None:
        cases = data.get("cases")
        if not isinstance(cases, list):
            report.success = False
            report.errors.append("if-else node cases must be a list")
            return
        for case in cases:
            if "case_id" not in case or "conditions" not in case:
                report.success = False
                report.errors.append("if-else case must include case_id and conditions")
                continue
            conditions = case.get("conditions")
            if not isinstance(conditions, list) or not conditions:
                report.success = False
                report.errors.append("if-else case conditions must be a non-empty list")
                continue
            for condition in conditions:
                for key in ["variable_selector", "comparison_operator", "value", "varType"]:
                    if key not in condition:
                        report.success = False
                        report.errors.append(f"if-else condition missing field: {key}")

    def _validate_iteration_node(self, data: dict, report) -> None:
        if not isinstance(data.get("iterator_selector"), list) or not data.get("iterator_selector"):
            report.success = False
            report.errors.append("iteration node missing iterator_selector")
        if not isinstance(data.get("output_selector"), list) or not data.get("output_selector"):
            report.success = False
            report.errors.append("iteration node missing output_selector")
        if "start_node_id" in data and not isinstance(data.get("start_node_id"), str):
            report.success = False
            report.errors.append("iteration node start_node_id must be a string")

    def _validate_iteration_start_node(self, data: dict, report) -> None:
        if data.get("isInIteration") is not True:
            report.success = False
            report.errors.append("iteration-start node must be marked isInIteration")

    def _validate_variable_aggregator_node(self, data: dict, report) -> None:
        if "output_type" not in data:
            report.success = False
            report.errors.append("variable-aggregator node missing output_type")
        if not isinstance(data.get("variables"), list):
            report.success = False
            report.errors.append("variable-aggregator node variables must be a list")

    def _validate_http_request_node(self, data: dict, report) -> None:
        for key in ["method", "url", "headers", "params", "body", "outputs"]:
            if key not in data:
                report.success = False
                report.errors.append(f"http-request node missing {key}")
        if data.get("method") not in {"get", "post", "put", "delete", "patch", "head", "options"}:
            report.success = False
            report.errors.append("http-request node method must be lowercase HTTP method")
        if "body" in data and not isinstance(data.get("body"), dict):
            report.success = False
            report.errors.append("http-request node body must be an object")
        if "authorization" in data and not isinstance(data.get("authorization"), dict):
            report.success = False
            report.errors.append("http-request node authorization must be an object")
        if "timeout" in data and not isinstance(data.get("timeout"), dict):
            report.success = False
            report.errors.append("http-request node timeout must be an object")
        if "outputs" in data and not isinstance(data.get("outputs"), dict):
            report.success = False
            report.errors.append("http-request node outputs must be a dict")

    def _validate_template_transform_node(self, data: dict, report) -> None:
        if not isinstance(data.get("template"), str):
            report.success = False
            report.errors.append("template-transform node missing template text")
        if not isinstance(data.get("outputs"), dict):
            report.success = False
            report.errors.append("template-transform node outputs must be a dict")

    def _validate_tool_node(self, data: dict, report) -> None:
        for key in ["provider_id", "tool_name", "tool_parameters", "outputs"]:
            if key not in data:
                report.success = False
                report.errors.append(f"tool node missing {key}")
        for key in ["provider_name", "provider_type", "tool_label", "tool_configurations"]:
            if key in data and data[key] is None:
                report.success = False
                report.errors.append(f"tool node invalid {key}")
        if "tool_parameters" in data and not isinstance(data.get("tool_parameters"), dict):
            report.success = False
            report.errors.append("tool node tool_parameters must be a dict")
        if "outputs" in data and not isinstance(data.get("outputs"), dict):
            report.success = False
            report.errors.append("tool node outputs must be a dict")

    def _validate_doc_extractor_node(self, data: dict, report) -> None:
        for key in ["variable_selector", "outputs"]:
            if key not in data:
                report.success = False
                report.errors.append(f"doc-extractor node missing {key}")
        if "variable_selector" in data and not isinstance(data.get("variable_selector"), list):
            report.success = False
            report.errors.append("doc-extractor node variable_selector must be a list")
        if "outputs" in data and not isinstance(data.get("outputs"), dict):
            report.success = False
            report.errors.append("doc-extractor node outputs must be a dict")

    def _validate_parameter_extractor_node(self, data: dict, report) -> None:
        for key in ["instruction", "parameters", "outputs"]:
            if key not in data:
                report.success = False
                report.errors.append(f"parameter-extractor node missing {key}")
        if "instruction" in data and not isinstance(data.get("instruction"), str):
            report.success = False
            report.errors.append("parameter-extractor node instruction must be a string")
        if "parameters" in data and not isinstance(data.get("parameters"), list):
            report.success = False
            report.errors.append("parameter-extractor node parameters must be a list")
        if "outputs" in data and not isinstance(data.get("outputs"), dict):
            report.success = False
            report.errors.append("parameter-extractor node outputs must be a dict")
