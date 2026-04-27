from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.schema import ActionSlot, ConditionalBlock, DSLCompileOutput, DSLCompileReport, DSLCompiledEdge, DSLCompiledGraph, DSLCompiledNode, DSLCompiledWorkflow, DSLNormalizationOutput, DSLNormalizedContext, DifyNodeType, LoopBlock, ParallelBlock, SequentialBlock
from src.dsl_generation.node_mapper import NodeMapper


@dataclass
class _CompiledFlow:
    entry_node_ids: list[str]
    exit_node_ids: list[str]


class MinimalDifyWorkflowCompiler:
    def __init__(self) -> None:
        self.node_mapper = NodeMapper()
        self._compiled_nodes: list[DSLCompiledNode] = []
        self._compiled_edges: list[DSLCompiledEdge] = []
        self._node_mappings = []
        self._join_counter = 0
        self._control_join_nodes: dict[str, str] = {}
        self._loop_iterator_selectors: dict[str, list[str]] = {}

    def compile(self, normalization_output: DSLNormalizationOutput) -> DSLCompileOutput:
        report = DSLCompileReport(success=True)
        if normalization_output.context is None or not normalization_output.normalization_report.success:
            report.success = False
            report.errors.append("归一化未成功，无法进入 DSL 编译阶段")
            return DSLCompileOutput(normalization_output=normalization_output, compile_report=report)

        context = normalization_output.context
        self._compiled_nodes = []
        self._compiled_edges = []
        self._node_mappings = []
        self._join_counter = 0
        self._control_join_nodes = {}
        self._loop_iterator_selectors = {}
        flow = self._compile_block(context.normalized_skeleton, context, 0.0, 120.0)
        if not flow.entry_node_ids or not flow.exit_node_ids:
            report.success = False
            report.errors.append("无法编译根骨架块")
            return DSLCompileOutput(normalization_output=normalization_output, compile_report=report)

        graph = DSLCompiledGraph(nodes=self._compiled_nodes, edges=self._compiled_edges)
        workflow = DSLCompiledWorkflow(
            app={"name": context.utr.task_id, "description": context.utr.task_desc, "icon": "🤖", "icon_background": "#EFF1F5", "mode": "workflow", "use_icon_as_answer_icon": False},
            workflow={"conversation_variables": [], "environment_variables": [], "features": {}, "graph": {"nodes": [self._dump_node(n) for n in self._compiled_nodes], "edges": [self._dump_edge(e) for e in self._compiled_edges]}},
        )
        return DSLCompileOutput(normalization_output=normalization_output, compile_report=report, node_mappings=self._node_mappings, compiled_graph=graph, workflow=workflow)

    def _compile_block(
        self,
        block: SequentialBlock | ActionSlot | ConditionalBlock | ParallelBlock | LoopBlock,
        context: DSLNormalizedContext,
        x: float,
        y: float,
    ) -> _CompiledFlow:
        if isinstance(block, ActionSlot):
            return self._compile_action_slot(block, context, x, y)
        if isinstance(block, SequentialBlock):
            return self._compile_sequential(block, context, x, y)
        if isinstance(block, ConditionalBlock):
            return self._compile_conditional(block, context, x, y)
        if isinstance(block, ParallelBlock):
            return self._compile_parallel(block, context, x, y)
        if isinstance(block, LoopBlock):
            return self._compile_loop(block, context, x, y)
        return _CompiledFlow([], [])

    def _compile_sequential(self, block: SequentialBlock, context: DSLNormalizedContext, x: float, y: float) -> _CompiledFlow:
        flows: list[_CompiledFlow] = []
        cursor = x
        for child in block.children:
            flows.append(self._compile_block(child, context, cursor, y))
            cursor += 320.0
        for i in range(1, len(flows)):
            for s in flows[i - 1].exit_node_ids:
                for t in flows[i].entry_node_ids:
                    self._append_edge(s, t)
        if not flows:
            return _CompiledFlow([], [])
        return _CompiledFlow(flows[0].entry_node_ids, flows[-1].exit_node_ids)

    def _compile_conditional(self, block: ConditionalBlock, context: DSLNormalizedContext, x: float, y: float) -> _CompiledFlow:
        node_id = f"ifelse_{block.id}"
        cases = []
        branch_handles: dict[str, str] = {}
        branch_items = list(block.branches.items())
        selector, operator, value, var_type = self._build_condition_from_description(block.condition_description)
        selector = self._normalize_selector(selector, context)
        for idx, (name, _) in enumerate(branch_items):
            if name == "else":
                branch_handles[name] = "false"
                continue
            branch_value = name if value in {"", None, "branch"} else value
            case_id = f"{node_id}_case_{idx}"
            branch_handles[name] = case_id
            cases.append({"case_id": case_id, "logical_operator": "and", "conditions": [{"variable_selector": selector, "comparison_operator": operator, "value": branch_value, "varType": var_type}]})
        if_node = DSLCompiledNode(id=node_id, node_type=DifyNodeType.if_else, title="Condition", desc=block.condition_description, data={"type": DifyNodeType.if_else.value, "title": "Condition", "desc": block.condition_description, "selected": False, "cases": cases}, position={"x": x, "y": y})
        self._compiled_nodes.append(if_node)

        exits: list[str] = []
        for idx, (name, branch) in enumerate(branch_items):
            flow = self._compile_sequential(branch, context, x + 320.0, y + idx * 180.0)
            for entry in flow.entry_node_ids:
                self._append_edge(node_id, entry, source_handle=branch_handles.get(name, "false"))
            exits.extend(flow.exit_node_ids)

        join_id = f"conditional_join_{self._join_counter}_{block.id}"
        self._join_counter += 1
        join_node = DSLCompiledNode(id=join_id, node_type=DifyNodeType.code, title="Conditional Join", desc="Merge conditional branch outputs", data={"type": DifyNodeType.code.value, "title": "Conditional Join", "desc": "Merge conditional branch outputs", "selected": False, "variables": self._build_flow_variables(exits, context), "code_language": "python3", "code": "def main(**kwargs):\n    return {'result': kwargs}", "outputs": {"result": {"type": "object", "children": None}}}, position={"x": x + 640.0, "y": y + 90.0})
        self._compiled_nodes.append(join_node)
        self._control_join_nodes[block.id] = join_id
        for exit_id in exits:
            self._append_edge(exit_id, join_id)
        return _CompiledFlow([node_id], [join_id])

    def _compile_parallel(self, block: ParallelBlock, context: DSLNormalizedContext, x: float, y: float) -> _CompiledFlow:
        branch_flows: list[_CompiledFlow] = []
        entry_node_ids: list[str] = []

        for idx, branch in enumerate(block.branches):
            branch_flow = self._compile_sequential(branch, context, x + 320.0, y + idx * 220.0)
            branch_flows.append(branch_flow)
            entry_node_ids.extend(branch_flow.entry_node_ids)

        join_id = f"parallel_join_{self._join_counter}_{block.id}"
        self._join_counter += 1
        join_node = DSLCompiledNode(
            id=join_id,
            node_type=DifyNodeType.variable_aggregator,
            title="Parallel Join",
            desc="Merge parallel branch outputs",
            data={
                "type": DifyNodeType.variable_aggregator.value,
                "title": "Parallel Join",
                "desc": "Merge parallel branch outputs",
                "selected": False,
                "variables": self._build_flow_variable_selectors(
                    [exit_id for branch_flow in branch_flows for exit_id in branch_flow.exit_node_ids],
                    context,
                ),
                "output_type": "object",
                "outputs": {"result": {"type": "object", "children": None}},
            },
            position={"x": x + 640.0, "y": y + 100.0},
        )
        self._compiled_nodes.append(join_node)
        self._control_join_nodes[block.id] = join_id

        for branch_flow in branch_flows:
            for exit_id in branch_flow.exit_node_ids:
                self._append_edge(exit_id, join_id)

        return _CompiledFlow(entry_node_ids=list(dict.fromkeys(entry_node_ids)), exit_node_ids=[join_id])

    def _compile_loop(self, block: LoopBlock, context: DSLNormalizedContext, x: float, y: float) -> _CompiledFlow:
        loop_id = f"iteration_{block.id}"
        iterator_selector = self._resolve_loop_iterator_selector(block, context)
        self._loop_iterator_selectors[block.id] = iterator_selector
        loop_node = DSLCompiledNode(
            id=loop_id,
            node_type=DifyNodeType.iteration,
            title="Iteration",
            desc=block.loop_condition,
            data={
                "type": DifyNodeType.iteration.value,
                "title": "Iteration",
                "desc": block.loop_condition,
                "selected": False,
                "iterator_selector": iterator_selector,
                "output_selector": ["output"],
                "output_type": "array[object]",
                "is_parallel": False,
                "parallel_nums": 10,
                "start_node_id": f"{loop_id}_start",
                "error_handle_mode": "remove-abnormal-output",
            },
            position={"x": x, "y": y},
        )
        self._compiled_nodes.append(loop_node)

        body_flow = self._compile_sequential(block.body, context, x + 320.0, y)
        if body_flow.exit_node_ids:
            output_node_id = body_flow.exit_node_ids[-1]
            loop_node.data["output_selector"] = [
                output_node_id,
                self._default_output_field_for_node(output_node_id, context),
            ]
        start_node = DSLCompiledNode(
            id=f"{loop_id}_start",
            node_type=DifyNodeType.iteration_start,
            title="Iteration Start",
            desc="Iteration body start node",
            data={
                "type": DifyNodeType.iteration_start.value,
                "title": "Iteration Start",
                "desc": "Iteration body start node",
                "selected": False,
                "isInIteration": True,
            },
            position={"x": x + 320.0, "y": y - 120.0},
        )
        self._compiled_nodes.append(start_node)
        self._mark_iteration_body(body_flow, loop_id)
        for entry in body_flow.entry_node_ids:
            self._append_edge(f"{loop_id}_start", entry, iteration_id=loop_id)

        join_id = f"loop_join_{self._join_counter}_{block.id}"
        self._join_counter += 1
        join_node = DSLCompiledNode(
            id=join_id,
            node_type=DifyNodeType.code,
            title="Loop Join",
            desc="Merge iteration outputs",
            data={
                "type": DifyNodeType.code.value,
                "title": "Loop Join",
                "desc": "Merge iteration outputs",
                "selected": False,
                "variables": [
                    {
                        "variable": "iteration_output",
                        "value_selector": [loop_id, "output"],
                    }
                ],
                "code_language": "python3",
                "code": "def main(**kwargs):\n    return {'result': kwargs}",
                "outputs": {"result": {"type": "object", "children": None}},
            },
            position={"x": x + 640.0, "y": y},
        )
        self._compiled_nodes.append(join_node)
        self._control_join_nodes[block.id] = join_id
        self._append_edge(loop_id, join_id)

        return _CompiledFlow([loop_id], [join_id])

    def _mark_iteration_body(self, body_flow: _CompiledFlow, iteration_id: str) -> None:
        body_node_ids = self._collect_reachable_node_ids(body_flow.entry_node_ids)
        for node in self._compiled_nodes:
            if node.id not in body_node_ids:
                continue
            node.data["isInIteration"] = True
            node.data["iteration_id"] = iteration_id
            if node.id in body_flow.entry_node_ids:
                node.data["isIterationStart"] = True
        for edge in self._compiled_edges:
            if edge.source in body_node_ids and edge.target in body_node_ids:
                edge.data["isInIteration"] = True
                edge.data["iteration_id"] = iteration_id

    def _collect_reachable_node_ids(self, entry_node_ids: list[str]) -> set[str]:
        reachable: set[str] = set()
        queue = list(entry_node_ids)
        while queue:
            node_id = queue.pop(0)
            if node_id in reachable:
                continue
            reachable.add(node_id)
            for edge in self._compiled_edges:
                if edge.source == node_id and edge.target not in reachable:
                    queue.append(edge.target)
        return reachable

    def _compile_action_slot(self, slot: ActionSlot, context: DSLNormalizedContext, x: float, y: float) -> _CompiledFlow:
        slot_index = {s.slot_id: s for s in context.action_slots}
        view = slot_index.get(slot.id)
        if slot.action_id == "start_node":
            node = self._build_start_node(context, slot.id, x, y)
        elif slot.action_id == "end_node":
            previous_node_id, previous_output_field = self._resolve_end_output_binding(view, context, slot_index)
            node = self._build_end_node(slot.id, previous_node_id, previous_output_field, x, y)
        else:
            action = context.action_index[slot.action_id]
            mapping = self.node_mapper.map_action(action=action, parent_block_type=self._infer_parent_block_type(context, view.parent_block_id if view else ""), upstream_actions=self._resolve_action_ids(view.upstream_slot_ids if view else [], slot_index), downstream_actions=self._resolve_action_ids(view.downstream_slot_ids if view else [], slot_index), available_variables=[v.name for v in context.utr.metadata.core_variables], available_resources=[r.name for r in context.utr.metadata.core_resources])
            self._node_mappings.append(mapping)
            inputs = self._build_input_bindings(action, view, context, slot_index)
            node = self._build_business_node(action, mapping, inputs, x, y)
        self._compiled_nodes.append(node)
        return _CompiledFlow([node.id], [node.id])

    def _build_start_node(self, context: DSLNormalizedContext, slot_id: str, x: float, y: float) -> DSLCompiledNode:
        variables = [{"label": v.name, "variable": v.name, "type": "paragraph" if v.type.value == "string" else "text-input", "required": False, "max_length": 48000} for v in context.utr.metadata.core_variables]
        return DSLCompiledNode(id="start", node_type=DifyNodeType.start, title="Start", desc="Workflow start node", data={"type": DifyNodeType.start.value, "title": "Start", "desc": "Workflow start node", "selected": False, "variables": variables}, position={"x": x, "y": y})

    def _build_end_node(self, slot_id: str, previous_node_id: str | None, previous_output_field: str | None, x: float, y: float) -> DSLCompiledNode:
        outputs = []
        if previous_node_id and previous_output_field:
            outputs.append({"variable": previous_output_field, "value_selector": [previous_node_id, previous_output_field]})
        return DSLCompiledNode(id=f"end_{slot_id}", node_type=DifyNodeType.end, title="End", desc="Workflow end node", data={"type": DifyNodeType.end.value, "title": "End", "desc": "Workflow end node", "outputs": outputs}, position={"x": x, "y": y})

    def _build_business_node(self, action: Any, mapping: Any, inputs: list[dict[str, Any]], x: float, y: float) -> DSLCompiledNode:
        if mapping.chosen_node_type == DifyNodeType.llm:
            data = self._build_llm_node_data(action, inputs)
        elif mapping.chosen_node_type == DifyNodeType.http_request:
            data = self._build_http_request_node_data(action, inputs)
        elif mapping.chosen_node_type == DifyNodeType.template_transform:
            data = self._build_template_transform_node_data(action, inputs)
        elif mapping.chosen_node_type == DifyNodeType.tool:
            data = self._build_tool_node_data(action, inputs)
        elif mapping.chosen_node_type == DifyNodeType.doc_extractor:
            data = self._build_doc_extractor_node_data(action, inputs)
        elif mapping.chosen_node_type == DifyNodeType.parameter_extractor:
            data = self._build_parameter_extractor_node_data(action, inputs)
        elif mapping.chosen_node_type == DifyNodeType.variable_aggregator:
            data = self._build_variable_aggregator_node_data(action, inputs)
        elif mapping.chosen_node_type == DifyNodeType.if_else:
            data = self._build_action_if_else_node_data(action, inputs)
        elif mapping.chosen_node_type == DifyNodeType.iteration:
            data = self._build_action_iteration_node_data(action, inputs)
        else:
            data = self._build_code_node_data(action, inputs)
        return DSLCompiledNode(id=f"node_{action.action_id}", node_type=mapping.chosen_node_type, title=action.action_name, desc=action.description, data=data, position={"x": x, "y": y})

    def _build_llm_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        user_prompt = self._build_primary_input_reference(inputs)
        return {
            "type": DifyNodeType.llm.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": inputs,
            "model": {
                "mode": "chat",
                "name": "gpt-4o-mini",
                "provider": "openai",
                "completion_params": {"temperature": 0.7},
            },
            "context": {"enabled": False, "variable_selector": []},
            "vision": {"enabled": False},
            "prompt_template": [
                {"role": "system", "text": action.description or action.action_name},
                {"role": "user", "text": user_prompt},
            ],
        }

    def _build_http_request_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        field = self._default_output_field(action)
        method = self._infer_http_method(action)
        url_selector = self._select_input_selector(inputs, ["url", "endpoint", "api_url"])
        body_selector = self._select_input_selector(inputs, ["payload", "body", "request_body", "data", "json"])
        return {
            "type": DifyNodeType.http_request.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": inputs,
            "method": method,
            "url": self._selector_reference(url_selector) if url_selector else "https://example.com/api",
            "url_selector": url_selector,
            "authorization": {"type": "no-auth", "config": None},
            "headers": "",
            "params": "",
            "body": {
                "type": "json",
                "data": self._selector_reference(body_selector) if body_selector else "",
                "data_selector": body_selector,
            },
            "timeout": {
                "max_connect_timeout": 30,
                "max_read_timeout": 30,
                "max_write_timeout": 30,
            },
            "retry_config": {
                "retry_enabled": False,
                "max_retries": 3,
                "retry_interval": 100,
            },
            "outputs": {
                field: {"type": "object", "children": None},
            },
        }

    def _build_template_transform_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        field = self._default_output_field(action)
        return {
            "type": DifyNodeType.template_transform.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": inputs,
            "template": self._build_template_text(action),
            "outputs": {
                field: {"type": "string", "children": None},
            },
        }

    def _build_tool_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        field = self._default_output_field(action)
        provider_id, tool_name = self._infer_tool_identity(action)
        return {
            "type": DifyNodeType.tool.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": inputs,
            "provider_id": provider_id,
            "provider_name": provider_id,
            "provider_type": "builtin",
            "tool_label": action.action_name,
            "tool_name": tool_name,
            "tool_configurations": {},
            "tool_parameters": {
                item["variable"]: {"type": "mixed", "value": self._selector_reference(item["value_selector"])}
                for item in inputs
            },
            "outputs": {
                field: {"type": "object", "children": None},
            },
        }

    def _build_doc_extractor_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        field = self._default_output_field(action)
        return {
            "type": DifyNodeType.doc_extractor.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": inputs,
            "variable_selector": self._resolve_document_selector(inputs),
            "outputs": {
                field: {"type": "string", "children": None},
            },
        }

    def _build_parameter_extractor_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        field = self._default_output_field(action)
        return {
            "type": DifyNodeType.parameter_extractor.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": inputs,
            "instruction": action.description or action.action_name,
            "query": self._build_primary_input_reference(inputs),
            "model": {
                "mode": "chat",
                "name": "gpt-4o-mini",
                "provider": "openai",
                "completion_params": {"temperature": 0.0},
            },
            "reasoning_mode": "prompt",
            "vision": {"enabled": False},
            "parameters": [
                {
                    "name": output_name,
                    "type": "string",
                    "description": f"Extracted value for {output_name}",
                    "required": False,
                }
                for output_name in (action.outputs or [field])
            ],
            "outputs": {
                field: {"type": "object", "children": None},
            },
        }

    def _build_variable_aggregator_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        field = self._default_output_field(action)
        return {
            "type": DifyNodeType.variable_aggregator.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": [item["value_selector"] for item in inputs],
            "output_type": "object",
            "outputs": {
                field: {"type": "object", "children": None},
            },
        }

    def _build_action_if_else_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        selector, operator, value, var_type = self._build_condition_from_description(
            action.description or action.action_name
        )
        selector = self._select_input_selector(
            inputs,
            [selector[-1] if selector else "", "status", "category", "risk", "score", "flag", "conditions"],
        ) or selector
        return {
            "type": DifyNodeType.if_else.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": inputs,
            "cases": [
                {
                    "case_id": f"node_{action.action_id}_case_0",
                    "logical_operator": "and",
                    "conditions": [
                        {
                            "variable_selector": selector,
                            "comparison_operator": operator,
                            "value": value,
                            "varType": var_type,
                        }
                    ],
                }
            ],
        }

    def _build_action_iteration_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        field = self._default_output_field(action)
        return {
            "type": DifyNodeType.iteration.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": inputs,
            "iterator_selector": self._resolve_iterator_selector(inputs),
            "output_selector": [field],
            "outputs": {
                field: {"type": "array[object]", "children": None},
            },
        }

    def _build_code_node_data(self, action: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        field = self._default_output_field(action)
        return {
            "type": DifyNodeType.code.value,
            "title": action.action_name,
            "desc": action.description,
            "selected": False,
            "variables": inputs,
            "code_language": "python3",
            "code": f"def main(**kwargs):\n    return {{'{field}': 'TODO'}}",
            "outputs": {field: {"type": "string", "children": None}},
        }

    def _default_output_field(self, action: Any) -> str:
        if action.outputs:
            return action.outputs[0]
        if "summary" in action.action_name.lower() or "summary" in action.description.lower():
            return "text"
        return "result"

    def _build_primary_input_reference(self, inputs: list[dict[str, Any]]) -> str:
        if not inputs:
            return "{{#sys.query#}}"
        return self._selector_reference(inputs[0]["value_selector"])

    def _resolve_document_selector(self, inputs: list[dict[str, Any]]) -> list[str]:
        for candidate in ["file", "document", "pdf", "docx", "attachment"]:
            selector = self._select_input_selector(inputs, [candidate])
            if selector:
                return selector
        if inputs:
            return inputs[0]["value_selector"]
        return ["sys", "files"]

    def _resolve_iterator_selector(self, inputs: list[dict[str, Any]]) -> list[str]:
        for item in inputs:
            normalized = item["variable"].lower()
            if any(keyword in normalized for keyword in ["items", "files", "records", "list", "rows", "documents"]):
                return item["value_selector"]
        if inputs:
            return inputs[0]["value_selector"]
        return ["start", "items"]

    def _infer_http_method(self, action: Any) -> str:
        text = f"{action.action_name} {action.description}".lower()
        for method in ["post", "put", "delete", "patch", "get"]:
            if method in text:
                return method
        return "get"

    def _build_template_text(self, action: Any) -> str:
        if not action.inputs:
            return action.description or action.action_name
        lines = [action.description or action.action_name]
        lines.extend(f"{item}: {{{{{item}}}}}" for item in action.inputs)
        return "\n".join(lines)

    def _infer_tool_identity(self, action: Any) -> tuple[str, str]:
        normalized_name = action.action_name.lower()
        if "search" in normalized_name:
            return "search", "search"
        if "calendar" in normalized_name:
            return "calendar", "calendar"
        if "mail" in normalized_name or "email" in normalized_name:
            return "mail", "mail"
        return "tool", normalized_name or "tool"

    def _build_condition_from_description(self, description: str) -> tuple[list[str], str, str, str]:
        text = (description or "").lower()
        selector = ["sys", "query"]
        var_type = "string"
        if any(keyword in text for keyword in ["状态", "status"]):
            selector = ["start", "status"]
        elif any(keyword in text for keyword in ["类型", "category", "分类"]):
            selector = ["start", "category"]
        elif any(keyword in text for keyword in ["风险", "risk"]):
            selector = ["start", "risk_level"]
            var_type = "number"
        elif any(keyword in text for keyword in ["分数", "score", "得分"]):
            selector = ["start", "score"]
            var_type = "number"
        elif any(keyword in text for keyword in ["真假", "true", "false", "是否"]):
            selector = ["start", "flag"]
            var_type = "boolean"

        number_chars = []
        collecting = False
        for ch in text:
            if ch.isdigit() or (collecting and ch == "."):
                number_chars.append(ch)
                collecting = True
            elif collecting:
                break
        extracted_number = "".join(number_chars)

        if any(keyword in text for keyword in ["不为空", "not empty"]):
            return selector, "not empty", "", var_type
        if any(keyword in text for keyword in ["为空", "empty"]):
            return selector, "empty", "", var_type
        if any(keyword in text for keyword in ["大于等于", ">=", "at least"]):
            return selector, ">=", extracted_number or "0", var_type
        if any(keyword in text for keyword in ["小于等于", "<=", "at most"]):
            return selector, "<=", extracted_number or "0", var_type
        if any(keyword in text for keyword in ["大于", ">", "greater"]):
            return selector, ">", extracted_number or "0", var_type
        if any(keyword in text for keyword in ["小于", "<", "less"]):
            return selector, "<", extracted_number or "0", var_type
        if any(keyword in text for keyword in ["等于", "=", "equal"]):
            return selector, "=", "target", var_type
        if any(keyword in text for keyword in ["高风险", "high"]):
            return selector, "contains", "high", var_type
        if any(keyword in text for keyword in ["低风险", "low"]):
            return selector, "contains", "low", var_type
        if any(keyword in text for keyword in ["true", "真", "是"]):
            return selector, "=", "true", var_type
        if any(keyword in text for keyword in ["false", "假", "否"]):
            return selector, "=", "false", var_type
        return selector, "contains", "branch", var_type

    def _infer_parent_block_type(self, context: DSLNormalizedContext, parent_block_id: str) -> str:
        for block in context.blocks:
            if block.block_id == parent_block_id:
                return block.block_type
        return "Sequential"

    def _resolve_action_ids(self, slot_ids: list[str], slot_index: dict[str, Any]) -> list[str]:
        return [slot_index[slot_id].action_id for slot_id in slot_ids if slot_id in slot_index]

    def _build_input_bindings(
        self,
        action: Any,
        view: Any,
        context: DSLNormalizedContext,
        slot_index: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "variable": input_name,
                "value_selector": self._resolve_input_selector(
                    input_name=input_name,
                    view=view,
                    context=context,
                    slot_index=slot_index,
                ),
            }
            for input_name in action.inputs
        ]

    def _resolve_input_selector(
        self,
        input_name: str,
        view: Any,
        context: DSLNormalizedContext,
        slot_index: dict[str, Any],
    ) -> list[str]:
        if view:
            loop_selector = self._selector_for_loop_item(input_name, view, context)
            for upstream_slot_id in reversed(view.upstream_slot_ids):
                if loop_selector and not self._slot_shares_innermost_loop_domain(
                    view,
                    slot_index.get(upstream_slot_id),
                ):
                    continue
                selector = self._selector_for_upstream_slot(
                    input_name=input_name,
                    upstream_slot_id=upstream_slot_id,
                    context=context,
                    slot_index=slot_index,
                )
                if selector:
                    return selector
            if loop_selector:
                return loop_selector
        start_variable_names = {v.name for v in context.utr.metadata.core_variables}
        if input_name in start_variable_names:
            return ["start", input_name]
        start_selector = self._first_start_selector(context)
        if start_selector:
            return start_selector
        return ["sys", "query"]

    def _selector_for_upstream_slot(
        self,
        input_name: str,
        upstream_slot_id: str,
        context: DSLNormalizedContext,
        slot_index: dict[str, Any],
    ) -> list[str] | None:
        upstream_slot = slot_index.get(upstream_slot_id)
        if upstream_slot is None:
            return None

        join_node_id = self._join_node_for_slot_control_domain(upstream_slot)
        if join_node_id:
            return [join_node_id, "result"]

        upstream_action_id = upstream_slot.action_id
        if upstream_action_id == "start_node":
            if input_name in {v.name for v in context.utr.metadata.core_variables}:
                return ["start", input_name]
            return self._first_start_selector(context) or ["sys", "query"]
        upstream_action = context.action_index.get(upstream_action_id)
        if upstream_action is None:
            return None

        start_variable_names = {v.name for v in context.utr.metadata.core_variables}
        output_field = self._match_output_field(
            input_name,
            upstream_action.outputs,
            allow_single_output_fallback=input_name not in start_variable_names,
        )
        if output_field is None:
            return None
        return [f"node_{upstream_action.action_id}", output_field]

    def _selector_for_loop_item(
        self,
        input_name: str,
        view: Any,
        context: DSLNormalizedContext,
    ) -> list[str] | None:
        for domain in reversed(view.control_domains):
            if domain.domain_type != "loop":
                continue
            iterator_selector = self._loop_iterator_selectors.get(domain.block_id)
            if not iterator_selector:
                continue
            if self._loop_item_matches_input(input_name, iterator_selector, context):
                return [f"iteration_{domain.block_id}", "item"]
        return None

    def _loop_item_matches_input(
        self,
        input_name: str,
        iterator_selector: list[str],
        context: DSLNormalizedContext,
    ) -> bool:
        if len(iterator_selector) < 2 or iterator_selector[0] != "start":
            return False
        iterator_name = iterator_selector[1]
        if input_name == iterator_name:
            return True
        iterator_variable = next(
            (
                variable
                for variable in context.utr.metadata.core_variables
                if variable.name == iterator_name
            ),
            None,
        )
        return iterator_variable is not None and iterator_variable.type.value in {"list", "file", "data"}

    def _slot_shares_innermost_loop_domain(self, view: Any, upstream_view: Any) -> bool:
        if upstream_view is None:
            return False
        view_loop = self._innermost_loop_block_id(view)
        upstream_loop = self._innermost_loop_block_id(upstream_view)
        return bool(view_loop and view_loop == upstream_loop)

    def _innermost_loop_block_id(self, view: Any) -> str:
        for domain in reversed(view.control_domains):
            if domain.domain_type == "loop":
                return domain.block_id
        return ""

    def _join_node_for_slot_control_domain(self, slot: Any) -> str | None:
        for domain in reversed(slot.control_domains):
            join_node_id = self._control_join_nodes.get(domain.block_id)
            if join_node_id:
                return join_node_id
        return None

    def _match_output_field(
        self,
        input_name: str,
        outputs: list[str],
        allow_single_output_fallback: bool = True,
    ) -> str | None:
        if not outputs:
            return None
        normalized_input = input_name.lower()
        for output in outputs:
            if output == input_name:
                return output
        for output in outputs:
            normalized_output = output.lower()
            if normalized_output == normalized_input:
                return output
        for output in outputs:
            normalized_output = output.lower()
            if normalized_input in normalized_output or normalized_output in normalized_input:
                return output
        return outputs[0] if allow_single_output_fallback and len(outputs) == 1 else None

    def _select_input_selector(
        self,
        inputs: list[dict[str, Any]],
        candidates: list[str],
    ) -> list[str]:
        normalized_candidates = [candidate.lower() for candidate in candidates if candidate]
        for candidate in normalized_candidates:
            for item in inputs:
                if item["variable"].lower() == candidate:
                    return item["value_selector"]
        for candidate in normalized_candidates:
            for item in inputs:
                variable = item["variable"].lower()
                if candidate in variable or variable in candidate:
                    return item["value_selector"]
        return []

    def _selector_reference(self, selector: list[str]) -> str:
        if not selector:
            return ""
        return "{{#" + ".".join(selector) + "#}}"

    def _normalize_selector(self, selector: list[str], context: DSLNormalizedContext) -> list[str]:
        if len(selector) < 2 or selector[0] != "start":
            return selector
        start_variable_names = {v.name for v in context.utr.metadata.core_variables}
        if selector[1] in start_variable_names:
            return selector
        start_selector = self._first_start_selector(context)
        return start_selector or ["sys", "query"]

    def _first_start_selector(self, context: DSLNormalizedContext) -> list[str]:
        if context.utr.metadata.core_variables:
            return ["start", context.utr.metadata.core_variables[0].name]
        return []

    def _resolve_loop_iterator_selector(self, block: LoopBlock, context: DSLNormalizedContext) -> list[str]:
        candidates = [
            variable.name
            for variable in context.utr.metadata.core_variables
            if variable.type.value in {"list", "file", "data"}
        ]
        candidates.extend(variable.name for variable in context.utr.metadata.core_variables)
        for candidate in candidates:
            normalized = candidate.lower()
            if any(keyword in normalized for keyword in ["items", "files", "records", "list", "rows", "documents"]):
                return ["start", candidate]
        if candidates:
            return ["start", candidates[0]]
        return ["sys", "query"]

    def _resolve_end_output_binding(
        self,
        view: Any,
        context: DSLNormalizedContext,
        slot_index: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        if not view or not view.upstream_slot_ids:
            return None, None
        upstream_slot_ids = [slot_id for slot_id in view.upstream_slot_ids if slot_id in slot_index]
        if not upstream_slot_ids:
            return None, None
        if len(upstream_slot_ids) > 1 or any(self._slot_is_inside_control_domain(slot_id, slot_index) for slot_id in upstream_slot_ids):
            candidate_join_nodes = [
                node.id
                for node in self._compiled_nodes
                if node.id.startswith(("conditional_join_", "parallel_join_", "loop_join_"))
            ]
            if candidate_join_nodes:
                return candidate_join_nodes[-1], "result"

        prev_slot_id = upstream_slot_ids[0]
        prev_action_id = slot_index[prev_slot_id].action_id
        if prev_action_id == "end_node":
            return None, None
        if prev_action_id == "start_node":
            start_selector = self._first_start_selector(context)
            if start_selector:
                return start_selector[0], start_selector[1]
            return None, None
        if prev_action_id in context.action_index:
            prev_action = context.action_index[prev_action_id]
            return f"node_{prev_action.action_id}", self._default_output_field(prev_action)
        return None, None

    def _slot_is_inside_control_domain(self, slot_id: str, slot_index: dict[str, Any]) -> bool:
        slot = slot_index.get(slot_id)
        if not slot:
            return False
        return any(domain.domain_type in {"conditional", "parallel", "loop"} for domain in slot.control_domains)

    def _build_flow_variables(self, node_ids: list[str], context: DSLNormalizedContext) -> list[dict[str, Any]]:
        variables: list[dict[str, Any]] = []
        seen: set[str] = set()
        for node_id in node_ids:
            if node_id in seen:
                continue
            seen.add(node_id)
            variables.append(
                {
                    "variable": f"input_{len(variables) + 1}",
                    "value_selector": [node_id, self._default_output_field_for_node(node_id, context)],
                }
            )
        return variables

    def _build_flow_variable_selectors(self, node_ids: list[str], context: DSLNormalizedContext) -> list[list[str]]:
        selectors: list[list[str]] = []
        seen: set[str] = set()
        for node_id in node_ids:
            if node_id in seen:
                continue
            seen.add(node_id)
            selectors.append([node_id, self._default_output_field_for_node(node_id, context)])
        return selectors

    def _default_output_field_for_node(self, node_id: str, context: DSLNormalizedContext) -> str:
        if node_id.startswith(("conditional_join_", "parallel_join_", "loop_join_")):
            return "result"
        if node_id.startswith("node_"):
            action_id = node_id.removeprefix("node_")
            action = context.action_index.get(action_id)
            if action:
                return self._default_output_field(action)
        return "result"

    def _node_type_of(self, node_id: str) -> str:
        for node in self._compiled_nodes:
            if node.id == node_id:
                return node.node_type.value
        return "unknown"

    def _append_edge(
        self,
        source_id: str,
        target_id: str,
        source_handle: str = "source",
        iteration_id: str | None = None,
    ) -> None:
        data = {
            "sourceType": self._node_type_of(source_id),
            "targetType": self._node_type_of(target_id),
            "isInIteration": iteration_id is not None,
            "isInLoop": False,
        }
        if iteration_id:
            data["iteration_id"] = iteration_id
        self._compiled_edges.append(DSLCompiledEdge(id=f"edge_{len(self._compiled_edges)}_{source_id}_{target_id}", source=source_id, target=target_id, source_handle=source_handle, target_handle="target", data=data))

    def _dump_node(self, node: DSLCompiledNode) -> dict[str, Any]:
        dumped = {"id": node.id, "type": "custom", "position": node.position, "positionAbsolute": node.position, "width": 244, "height": 96, "selected": False, "sourcePosition": "right", "targetPosition": "left", "data": node.data}
        iteration_id = node.data.get("iteration_id")
        if isinstance(iteration_id, str) and iteration_id:
            dumped["parentId"] = iteration_id
            dumped["extent"] = "parent"
            dumped["zIndex"] = 1001 if node.data.get("isIterationStart") else 1002
        elif node.node_type == DifyNodeType.iteration_start:
            parent_id = node.id.removesuffix("_start")
            dumped["parentId"] = parent_id
            dumped["zIndex"] = 1002
        return dumped

    def _dump_edge(self, edge: DSLCompiledEdge) -> dict[str, Any]:
        z_index = 1002 if edge.data.get("isInIteration") else 0
        return {"id": edge.id, "type": "custom", "source": edge.source, "target": edge.target, "sourceHandle": edge.source_handle, "targetHandle": edge.target_handle, "selected": False, "data": edge.data, "zIndex": z_index}
