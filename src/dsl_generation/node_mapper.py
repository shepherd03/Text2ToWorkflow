from __future__ import annotations

import re
from collections import Counter

from src.core.config import load_settings
from src.core.schema import (
    Action,
    DifyNodeType,
    MappingConfidence,
    NodeCandidateScore,
    NodeMappingContext,
    NodeMappingResult,
    NodeScoringWeights,
)
from src.dsl_generation.node_mapping_rules import MAPPING_RULES, NODE_SELECTION_PRIORITY
from src.dsl_generation.semantic_retriever import build_semantic_backend


class NodeMapper:
    RULE_WEIGHT = 0.40
    SEMANTIC_WEIGHT = 0.32
    COVERAGE_WEIGHT = 0.20
    CONTEXT_WEIGHT = 0.05
    PRIORITY_WEIGHT = 0.03

    def __init__(self) -> None:
        self._fallback_map: dict[DifyNodeType, DifyNodeType] = {
            DifyNodeType.tool: DifyNodeType.http_request,
            DifyNodeType.http_request: DifyNodeType.code,
            DifyNodeType.template_transform: DifyNodeType.code,
            DifyNodeType.llm: DifyNodeType.code,
            DifyNodeType.doc_extractor: DifyNodeType.code,
            DifyNodeType.parameter_extractor: DifyNodeType.code,
            DifyNodeType.if_else: DifyNodeType.code,
            DifyNodeType.iteration: DifyNodeType.code,
            DifyNodeType.variable_aggregator: DifyNodeType.code,
        }
        self._required_params_map: dict[DifyNodeType, list[str]] = {
            DifyNodeType.start: [],
            DifyNodeType.end: [],
            DifyNodeType.code: [],
            DifyNodeType.llm: ["prompt"],
            DifyNodeType.http_request: ["url", "method"],
            DifyNodeType.template_transform: ["template"],
            DifyNodeType.variable_aggregator: ["variables"],
            DifyNodeType.tool: ["provider_id", "tool_name"],
            DifyNodeType.if_else: ["conditions"],
            DifyNodeType.iteration: ["iterator_selector"],
            DifyNodeType.doc_extractor: ["variable_selector"],
            DifyNodeType.parameter_extractor: ["instruction"],
        }
        self._priority_index = {
            node_type: index for index, node_type in enumerate(NODE_SELECTION_PRIORITY)
        }
        self._settings = load_settings()
        self._semantic_retriever = build_semantic_backend(self._settings)

    def map_action(
        self,
        action: Action,
        parent_block_type: str = "Sequential",
        upstream_actions: list[str] | None = None,
        downstream_actions: list[str] | None = None,
        available_variables: list[str] | None = None,
        available_resources: list[str] | None = None,
    ) -> NodeMappingResult:
        context = NodeMappingContext(
            action=action,
            parent_block_type=parent_block_type,
            upstream_actions=upstream_actions or [],
            downstream_actions=downstream_actions or [],
            available_variables=available_variables or [],
            available_resources=available_resources or [],
        )
        return self.map_context(context)

    def map_context(self, context: NodeMappingContext) -> NodeMappingResult:
        action = context.action
        text = self._merge_text(action)

        if action.action_id == "start_node" or action.action_name.lower() == "start":
            return NodeMappingResult(
                source_action_id=action.action_id,
                chosen_node_type=DifyNodeType.start,
                candidate_node_types=[DifyNodeType.start],
                confidence=MappingConfidence.high,
                decision_reason="Matched start node",
                required_params=[],
                available_params={},
                fallback_node_type=DifyNodeType.start,
                trace=["direct-match:start_node"],
            )

        if action.action_id == "end_node" or action.action_name.lower() == "end":
            return NodeMappingResult(
                source_action_id=action.action_id,
                chosen_node_type=DifyNodeType.end,
                candidate_node_types=[DifyNodeType.end],
                confidence=MappingConfidence.high,
                decision_reason="Matched end node",
                required_params=[],
                available_params={},
                fallback_node_type=DifyNodeType.end,
                trace=["direct-match:end_node"],
            )

        trace: list[str] = []
        available_params = self._infer_available_params(context, text)
        structural_candidates, structural_markers = self._collect_structural_candidates(
            context, available_params, text
        )
        rule_candidates, matched_keywords = self._collect_rule_candidates(text)
        stage1_candidates = structural_candidates + rule_candidates
        semantic_candidates = self._collect_semantic_candidates(text)
        candidate_node_types = self._prepare_candidates(stage1_candidates, semantic_candidates)

        if structural_markers:
            preview = ", ".join(structural_markers[:8])
            trace.append(
                f"stage1:structural-hit count={len(structural_markers)} preview={preview}"
            )
        if matched_keywords:
            preview = ", ".join(matched_keywords[:8])
            trace.append(f"stage1:rule-hit count={len(matched_keywords)} preview={preview}")
        else:
            trace.append("stage1:no-rule-hit")

        if semantic_candidates:
            preview = ", ".join(
                f"{node.value}={score:.3f}" for node, score in semantic_candidates[:4]
            )
            trace.append(
                f"stage2:semantic-candidates backend={self._semantic_retriever.backend_name} {preview}"
            )

        scored_candidates = self._score_candidates(
            candidate_node_types=candidate_node_types,
            stage1_candidates=stage1_candidates,
            semantic_candidates=dict(semantic_candidates),
            available_params=available_params,
            context=context,
            text=text,
        )

        chosen = scored_candidates[0].node_type
        chosen_score = scored_candidates[0].total_score
        runner_up_score = scored_candidates[1].total_score if len(scored_candidates) > 1 else 0.0
        trace.append(
            "stage2:ranking "
            + ", ".join(
                f"{item.node_type.value}={item.total_score:.3f}" for item in scored_candidates[:5]
            )
        )

        required_params = self._required_params_map.get(chosen, [])
        missing = [param for param in required_params if not available_params.get(param, False)]
        degraded = False
        needs_human_fill = False
        degrade_reason = ""
        fallback = self._fallback_map.get(chosen, DifyNodeType.code)
        decision_reason = f"two-stage rank selected {chosen.value}"

        if missing:
            degraded = True
            needs_human_fill = True
            degrade_reason = f"missing required params: {', '.join(missing)}"
            trace.append(f"landing-check:failed {degrade_reason}")
            if chosen in self._fallback_map:
                chosen = fallback
                required_params = self._required_params_map.get(chosen, [])
                decision_reason = f"reranked type missing params, fallback to {chosen.value}"
            trace.append(f"fallback:{chosen.value}")
        else:
            trace.append("landing-check:passed")

        confidence = self._resolve_confidence(
            chosen=chosen,
            score=chosen_score,
            margin=chosen_score - runner_up_score,
            degraded=degraded,
        )

        return NodeMappingResult(
            source_action_id=action.action_id,
            chosen_node_type=chosen,
            candidate_node_types=candidate_node_types,
            confidence=confidence,
            decision_reason=decision_reason,
            required_params=required_params,
            available_params=available_params,
            fallback_node_type=fallback,
            needs_human_fill=needs_human_fill,
            degraded=degraded,
            degrade_reason=degrade_reason,
            trace=trace,
            scoring_weights=NodeScoringWeights(
                rule_weight=self.RULE_WEIGHT,
                semantic_weight=self.SEMANTIC_WEIGHT,
                coverage_weight=self.COVERAGE_WEIGHT,
                context_weight=self.CONTEXT_WEIGHT,
                priority_weight=self.PRIORITY_WEIGHT,
            ),
            chosen_score=chosen_score,
            runner_up_score=runner_up_score,
            candidate_scores=scored_candidates,
        )

    def confidence_probability(self, result: NodeMappingResult) -> float:
        margin = max(0.0, result.chosen_score - result.runner_up_score)
        base = {
            MappingConfidence.high: 0.86,
            MappingConfidence.medium: 0.66,
            MappingConfidence.low: 0.42,
        }[result.confidence]
        score_adjustment = min(max(result.chosen_score, 0.0), 1.0) * 0.10
        margin_adjustment = min(margin, 0.35) * 0.35
        degraded_penalty = 0.18 if result.degraded else 0.0
        candidate_penalty = 0.04 if len(result.candidate_node_types) <= 1 else 0.0
        probability = base + score_adjustment + margin_adjustment - degraded_penalty - candidate_penalty
        return round(min(max(probability, 0.05), 0.99), 4)

    def _collect_rule_candidates(self, text: str) -> tuple[list[DifyNodeType], list[str]]:
        candidates: list[DifyNodeType] = []
        matched_keywords: list[str] = []
        for rule in MAPPING_RULES:
            keyword = rule["keyword"]
            if not self._rule_match(text, keyword):
                continue
            candidates.append(DifyNodeType(rule["node_type"]))
            matched_keywords.append(keyword)
        return candidates, matched_keywords

    def _collect_structural_candidates(
        self,
        context: NodeMappingContext,
        available_params: dict[str, bool],
        text: str,
    ) -> tuple[list[DifyNodeType], list[str]]:
        candidates: list[DifyNodeType] = []
        markers: list[str] = []
        input_names = [item.lower() for item in context.action.inputs]
        output_names = [item.lower() for item in context.action.outputs]
        action_name = context.action.action_name.lower()
        resource_names = [item.lower() for item in context.available_resources]
        tool_resource_pool = resource_names + input_names
        parent = context.parent_block_type.lower()
        code_signal = self._code_signal_strength(input_names, text)
        llm_signal = self._llm_signal_strength(action_name, text, input_names, output_names)
        template_signal = self._template_signal_strength(
            action_name, text, input_names, output_names
        )
        parameter_extractor_signal = self._parameter_extractor_signal_strength(
            action_name, text, input_names, output_names
        )
        tool_signal = self._tool_signal_strength(tool_resource_pool, action_name, text)
        http_signal = self._http_signal_strength(
            input_names=input_names,
            output_names=output_names,
            action_name=action_name,
            text=text,
            available_params=available_params,
        )

        if code_signal >= 3:
            candidates.extend(
                [
                    DifyNodeType.code,
                    DifyNodeType.code,
                    DifyNodeType.code,
                    DifyNodeType.code,
                ]
            )
            markers.append("pattern:code-execution")

        if llm_signal >= 3 and code_signal < 3:
            candidates.extend([DifyNodeType.llm, DifyNodeType.llm, DifyNodeType.llm])
            markers.append("pattern:llm-platform")

        if template_signal >= 3:
            candidates.extend(
                [
                    DifyNodeType.template_transform,
                    DifyNodeType.template_transform,
                    DifyNodeType.template_transform,
                    DifyNodeType.template_transform,
                ]
            )
            markers.append("pattern:template-assignment")

        if parameter_extractor_signal >= 3 and code_signal < 3:
            candidates.extend(
                [
                    DifyNodeType.parameter_extractor,
                    DifyNodeType.parameter_extractor,
                    DifyNodeType.parameter_extractor,
                ]
            )
            markers.append("pattern:parameter-structure")

        if tool_signal >= 4:
            candidates.extend(
                [
                    DifyNodeType.tool,
                    DifyNodeType.tool,
                    DifyNodeType.tool,
                    DifyNodeType.tool,
                ]
            )
            markers.append("context:tool-resource")

        if http_signal >= 4:
            candidates.extend(
                [
                    DifyNodeType.http_request,
                    DifyNodeType.http_request,
                    DifyNodeType.http_request,
                    DifyNodeType.http_request,
                ]
            )
            markers.append("pattern:http-call")
        elif http_signal >= 3:
            candidates.extend([DifyNodeType.http_request, DifyNodeType.http_request])
            markers.append("pattern:http-call")

        if available_params.get("conditions") and self._condition_signal_strength(text) >= 1:
            candidates.extend(
                [
                    DifyNodeType.if_else,
                    DifyNodeType.if_else,
                    DifyNodeType.if_else,
                    DifyNodeType.if_else,
                ]
            )
            markers.append("param:conditions")
        elif parent == "conditional" or self._looks_like_condition_expression(text):
            candidates.append(DifyNodeType.if_else)
            markers.append("pattern:condition-gate")

        if available_params.get("iterator_selector") and code_signal < 3:
            candidates.extend([DifyNodeType.iteration, DifyNodeType.iteration])
            markers.append("param:iterator-selector")
        elif self._looks_like_iteration_action(input_names, text) and code_signal < 3:
            candidates.append(DifyNodeType.iteration)
            markers.append("pattern:collection-loop")

        if "parameters" in output_names:
            candidates.extend([DifyNodeType.parameter_extractor, DifyNodeType.parameter_extractor])
            markers.append("output:parameters")
        if "instruction" in input_names or "query" in input_names:
            candidates.append(DifyNodeType.parameter_extractor)
            markers.append("inputs:instruction-query")

        if self._looks_like_aggregation_action(input_names, output_names, text):
            candidates.extend(
                [DifyNodeType.variable_aggregator, DifyNodeType.variable_aggregator]
            )
            markers.append("pattern:variables-aggregate")

        return candidates, markers

    def _collect_semantic_candidates(self, text: str) -> list[tuple[DifyNodeType, float]]:
        return [
            (candidate.node_type, candidate.score)
            for candidate in self._semantic_retriever.search(text, top_k=5)
            if candidate.score >= 0.08
        ]

    def _prepare_candidates(
        self,
        stage1_candidates: list[DifyNodeType],
        semantic_candidates: list[tuple[DifyNodeType, float]],
    ) -> list[DifyNodeType]:
        combined = self._dedupe(stage1_candidates + [node for node, _ in semantic_candidates[:3]])
        if not combined:
            combined = [DifyNodeType.code]
        if DifyNodeType.code not in combined:
            combined.append(DifyNodeType.code)
        return combined

    def _score_candidates(
        self,
        candidate_node_types: list[DifyNodeType],
        stage1_candidates: list[DifyNodeType],
        semantic_candidates: dict[DifyNodeType, float],
        available_params: dict[str, bool],
        context: NodeMappingContext,
        text: str,
    ) -> list[NodeCandidateScore]:
        counter = Counter(stage1_candidates)
        total_hits = max(len(stage1_candidates), 1)
        scored: list[NodeCandidateScore] = []

        for node_type in candidate_node_types:
            rule_score = counter[node_type] / total_hits
            semantic_score = self._adjust_semantic_score(
                node_type=node_type,
                semantic_score=semantic_candidates.get(node_type, 0.0),
                available_params=available_params,
                context=context,
                text=text,
            )
            coverage_score = self._param_coverage(node_type, available_params)
            context_score = self._context_score(node_type, context)
            priority_bonus = self._priority_bonus(node_type)

            score = (
                self.RULE_WEIGHT * rule_score
                + self.SEMANTIC_WEIGHT * semantic_score
                + self.COVERAGE_WEIGHT * coverage_score
                + self.CONTEXT_WEIGHT * context_score
                + self.PRIORITY_WEIGHT * priority_bonus
            )
            scored.append(
                NodeCandidateScore(
                    node_type=node_type,
                    rule_score=rule_score,
                    semantic_score=semantic_score,
                    coverage_score=coverage_score,
                    context_score=context_score,
                    priority_bonus=priority_bonus,
                    total_score=score,
                )
            )

        scored.sort(
            key=lambda item: (-item.total_score, self._priority_index.get(item.node_type, 999))
        )
        return scored

    def _resolve_confidence(
        self,
        chosen: DifyNodeType,
        score: float,
        margin: float,
        degraded: bool,
    ) -> MappingConfidence:
        if degraded:
            return MappingConfidence.low
        if chosen in {DifyNodeType.start, DifyNodeType.end}:
            return MappingConfidence.high
        if score >= 0.60 and margin >= 0.12:
            return MappingConfidence.high
        if score >= 0.38 and margin >= 0.05:
            return MappingConfidence.medium
        return MappingConfidence.low

    def _param_coverage(
        self, node_type: DifyNodeType, available_params: dict[str, bool]
    ) -> float:
        required_params = self._required_params_map.get(node_type, [])
        if not required_params:
            return 1.0
        covered = sum(1 for param in required_params if available_params.get(param, False))
        return covered / len(required_params)

    def _context_score(self, node_type: DifyNodeType, context: NodeMappingContext) -> float:
        score = 0.0
        parent = context.parent_block_type.lower()
        text = self._merge_text(context.action)
        action_name = context.action.action_name.lower()
        output_names = {item.lower() for item in context.action.outputs}
        input_names = {item.lower() for item in context.action.inputs}
        resource_names = [item.lower() for item in context.available_resources]
        tool_resource_pool = resource_names + list(input_names)
        description = context.action.description.strip()
        code_signal = self._code_signal_strength(list(input_names), text)
        llm_signal = self._llm_signal_strength(action_name, text, list(input_names), list(output_names))
        template_signal = self._template_signal_strength(
            action_name, text, list(input_names), list(output_names)
        )
        parameter_extractor_signal = self._parameter_extractor_signal_strength(
            action_name, text, list(input_names), list(output_names)
        )
        tool_signal = self._tool_signal_strength(tool_resource_pool, action_name, text)
        http_signal = self._http_signal_strength(
            input_names=list(input_names),
            output_names=list(output_names),
            action_name=action_name,
            text=text,
            available_params=self._infer_available_params(context, text),
        )

        if parent == "loop" and node_type == DifyNodeType.iteration:
            score += 0.35
        if parent == "conditional" and node_type == DifyNodeType.if_else:
            score += 0.25
        if context.available_resources and node_type in {
            DifyNodeType.tool,
            DifyNodeType.http_request,
            DifyNodeType.doc_extractor,
        }:
            score += 0.15
        if context.upstream_actions and node_type == DifyNodeType.variable_aggregator:
            score += 0.10
        if context.downstream_actions and node_type == DifyNodeType.if_else:
            score += 0.05
        if ("branch" in text or "route" in text) and node_type == DifyNodeType.if_else:
            score += 0.10
        if not input_names and output_names == {"text"} and node_type == DifyNodeType.llm:
            score += 0.22
        if self._looks_like_chat_prompt(description) and node_type == DifyNodeType.llm:
            score += 0.15
        if "parameters" in output_names and node_type == DifyNodeType.parameter_extractor:
            score += 0.20
        if {"instruction", "query"} & input_names and node_type == DifyNodeType.parameter_extractor:
            score += 0.08
        if "template" in input_names and node_type == DifyNodeType.template_transform:
            score += 0.25
        if {"template", "output"} <= input_names | output_names and node_type == DifyNodeType.template_transform:
            score += 0.15
        if "variables" in input_names and node_type == DifyNodeType.variable_aggregator:
            score += 0.18
        if "output" in output_names and node_type == DifyNodeType.variable_aggregator:
            score += 0.12
        if parent == "loop" and node_type == DifyNodeType.code and "iterator_selector" not in input_names:
            score += 0.08
        if code_signal >= 3:
            if node_type == DifyNodeType.code:
                score += 1.00
            if node_type == DifyNodeType.http_request:
                score -= 0.10
            if node_type in {
                DifyNodeType.iteration,
                DifyNodeType.parameter_extractor,
                DifyNodeType.doc_extractor,
            }:
                score -= 0.45
        if llm_signal >= 3:
            if node_type == DifyNodeType.llm:
                score += 0.70
            if node_type == DifyNodeType.doc_extractor:
                score -= 0.40
            if node_type == DifyNodeType.code:
                score -= 0.20
            if node_type == DifyNodeType.tool and not context.available_resources:
                score -= 0.15
        if tool_signal >= 4:
            if node_type == DifyNodeType.tool:
                score += 0.90
            if node_type in {
                DifyNodeType.doc_extractor,
                DifyNodeType.parameter_extractor,
                DifyNodeType.variable_aggregator,
                DifyNodeType.code,
            }:
                score -= 0.30
        if http_signal >= 4:
            if node_type == DifyNodeType.http_request:
                score += 0.90
            if node_type in {
                DifyNodeType.code,
                DifyNodeType.llm,
                DifyNodeType.template_transform,
            }:
                score -= 0.20
        if template_signal >= 3:
            if node_type == DifyNodeType.template_transform:
                score += 0.90
            if node_type in {
                DifyNodeType.variable_aggregator,
                DifyNodeType.code,
                DifyNodeType.parameter_extractor,
            }:
                score -= 0.20
        if parameter_extractor_signal >= 3:
            if node_type == DifyNodeType.parameter_extractor:
                score += 0.85
            if node_type in {
                DifyNodeType.if_else,
                DifyNodeType.variable_aggregator,
                DifyNodeType.code,
            }:
                score -= 0.18

        return max(min(score, 1.0), -1.0)

    def _adjust_semantic_score(
        self,
        node_type: DifyNodeType,
        semantic_score: float,
        available_params: dict[str, bool],
        context: NodeMappingContext,
        text: str,
    ) -> float:
        if semantic_score <= 0:
            return semantic_score

        input_names = [item.lower() for item in context.action.inputs]
        output_names = [item.lower() for item in context.action.outputs]
        action_name = context.action.action_name.lower()
        resource_names = [item.lower() for item in context.available_resources]
        tool_resource_pool = resource_names + input_names
        parent = context.parent_block_type.lower()

        code_signal = self._code_signal_strength(input_names, text)
        llm_signal = self._llm_signal_strength(action_name, text, input_names, output_names)
        template_signal = self._template_signal_strength(
            action_name, text, input_names, output_names
        )
        parameter_extractor_signal = self._parameter_extractor_signal_strength(
            action_name, text, input_names, output_names
        )
        tool_signal = self._tool_signal_strength(tool_resource_pool, action_name, text)
        http_signal = self._http_signal_strength(
            input_names=input_names,
            output_names=output_names,
            action_name=action_name,
            text=text,
            available_params=available_params,
        )
        strong_condition_signal = (
            "conditions" in input_names
            or parent == "conditional"
            or (
                available_params.get("conditions")
                and self._condition_signal_strength(text) >= 1
            )
        )

        adjusted = semantic_score

        if code_signal >= 3:
            if node_type in {
                DifyNodeType.iteration,
                DifyNodeType.parameter_extractor,
                DifyNodeType.doc_extractor,
            }:
                adjusted *= 0.35
            elif node_type == DifyNodeType.http_request:
                adjusted *= 0.75
            elif node_type == DifyNodeType.code:
                adjusted = max(adjusted, 0.35)

        if llm_signal >= 3:
            if node_type in {DifyNodeType.doc_extractor, DifyNodeType.code}:
                adjusted *= 0.35
            elif node_type == DifyNodeType.tool and not context.available_resources:
                adjusted *= 0.50
            elif node_type == DifyNodeType.llm:
                adjusted = max(adjusted, 0.35)

        if template_signal >= 3:
            if node_type in {
                DifyNodeType.code,
                DifyNodeType.variable_aggregator,
                DifyNodeType.parameter_extractor,
                DifyNodeType.iteration,
            }:
                adjusted *= 0.45
            elif node_type == DifyNodeType.template_transform:
                adjusted = max(adjusted, 0.35)

        if parameter_extractor_signal >= 3:
            if node_type in {
                DifyNodeType.if_else,
                DifyNodeType.variable_aggregator,
                DifyNodeType.code,
                DifyNodeType.template_transform,
            }:
                adjusted *= 0.45
            elif node_type == DifyNodeType.parameter_extractor:
                adjusted = max(adjusted, 0.35)

        if tool_signal >= 4:
            if node_type in {
                DifyNodeType.doc_extractor,
                DifyNodeType.parameter_extractor,
                DifyNodeType.variable_aggregator,
                DifyNodeType.code,
            }:
                adjusted *= 0.30
            elif node_type == DifyNodeType.tool:
                adjusted = max(adjusted, 0.30)

        if http_signal >= 4:
            if node_type in {
                DifyNodeType.llm,
                DifyNodeType.code,
                DifyNodeType.template_transform,
            }:
                adjusted *= 0.45
            elif node_type == DifyNodeType.http_request:
                adjusted = max(adjusted, 0.30)

        if strong_condition_signal:
            if node_type in {DifyNodeType.http_request, DifyNodeType.llm}:
                adjusted *= 0.25
            elif node_type == DifyNodeType.if_else:
                adjusted = max(adjusted, 0.30)

        return adjusted

    def _priority_bonus(self, node_type: DifyNodeType) -> float:
        rank = self._priority_index.get(node_type, len(self._priority_index))
        return max(0.0, 1.0 - rank / max(len(self._priority_index), 1))

    def _infer_available_params(self, context: NodeMappingContext, text: str) -> dict[str, bool]:
        inputs = [item.lower() for item in context.action.inputs]
        outputs = [item.lower() for item in context.action.outputs]
        variables = [item.lower() for item in context.available_variables]
        resources = [item.lower() for item in context.available_resources]
        pool = set(inputs + outputs + variables + resources)
        method_tokens = {"get", "post", "put", "delete", "patch"}
        has_method_token = bool(set(self._tokenize(text)) & method_tokens)
        has_collection_input = any(self._has_collection_token(item) for item in inputs)
        parent = context.parent_block_type.lower()

        return {
            "prompt": bool(context.action.description.strip()) or "prompt" in pool,
            "url": (
                self._match_any(text, ["url", "endpoint", "api", "http"])
                or "url" in pool
                or any("url" in item or "endpoint" in item for item in inputs)
            ),
            "method": has_method_token or "method" in pool,
            "template": "template" in pool or self._match_any(text, ["template", "render"]),
            "variables": self._looks_like_aggregation_action(inputs, outputs, text),
            "provider_id": "provider_id" in pool
            or self._match_any(text, ["provider", "plugin", "tool", "calendar", "search", "mail"]),
            "tool_name": "tool_name" in pool
            or self._match_any(text, ["tool", "plugin", "calendar", "search", "mail"]),
            "conditions": "conditions" in pool
            or self._condition_signal_strength(text) >= 2
            or self._has_condition_route_phrase(text),
            "iterator_selector": "iterator_selector" in pool
            or (has_collection_input and (parent == "loop" or "items" in pool))
            or (
                has_collection_input
                and self._match_any(
                    text,
                    [
                        "iterate",
                        "for_each",
                        "loop",
                        "batch",
                        "each item",
                        "process each",
                        "walk through",
                        "one by one",
                    ],
                )
            ),
            "variable_selector": (
                any(item in {"file", "document", "doc", "pdf", "attachment"} for item in inputs)
                or self._match_any(text, ["file", "document", "doc", "pdf", "attachment"])
            ),
            "instruction": bool(context.action.description.strip()) or "instruction" in pool,
        }

    def _merge_text(self, action: Action) -> str:
        text = " ".join(
            [action.action_name, action.description, " ".join(action.inputs), " ".join(action.outputs)]
        )
        return text.lower()

    def _match_any(self, text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _tokenize(self, text: str) -> list[str]:
        normalized = text.lower().replace("-", " ").replace("_", " ")
        return re.findall(r"[a-z0-9]+", normalized)

    def _has_collection_token(self, value: str) -> bool:
        tokens = self._tokenize(value)
        collection_tokens = {
            "item",
            "items",
            "record",
            "records",
            "file",
            "files",
            "document",
            "documents",
            "row",
            "rows",
            "page",
            "pages",
            "entry",
            "entries",
            "chunk",
            "chunks",
            "section",
            "sections",
            "message",
            "messages",
            "task",
            "tasks",
            "url",
            "urls",
            "list",
        }
        return any(token in collection_tokens for token in tokens)

    def _code_signal_strength(self, inputs: list[str], text: str) -> int:
        input_names = {item.lower() for item in inputs}
        score = 0
        if {"jscode", "javascript", "pythoncode"} & input_names:
            score += 3
        if self._match_any(
            text,
            [
                "def main(",
                "import requests",
                "import json",
                "try:",
                "except ",
                "return {",
                "return {\"",
                "requests.get(",
                "requests.post(",
                "json.loads(",
                "re.sub(",
            ],
        ):
            score += 3
        if self._match_any(
            text,
            [
                "return $json",
                "$input.all",
                "const ",
                "let ",
                "function(",
                "=>",
                "return items",
                "item.json",
                "regex",
            ],
        ):
            score += 2
        if self._match_any(
            text,
            [
                "parsefloat(",
                ".map(",
                ".filter(",
                ".reduce(",
                "replace(",
                "$json.",
                "json[",
            ],
        ):
            score += 1
        return score

    def _llm_signal_strength(
        self,
        action_name: str,
        text: str,
        inputs: list[str],
        outputs: list[str],
    ) -> int:
        score = 0
        if self._match_any(
            action_name,
            [
                "openai chat model",
                "chat model",
                "ai agent",
                "knowledge base agent",
                "assistant",
                "chatbot",
                "language model",
                "llm",
            ],
        ):
            score += 2
        if self._match_any(
            text,
            [
                "openai",
                "gpt",
                "gpt-4",
                "gpt4",
                "gpt-4o",
                "gemini",
                "claude",
                "anthropic",
                "openrouter",
                "lmchatopenai",
            ],
        ):
            score += 1
        if self._match_any(
            text,
            [
                "answer user questions",
                "answering user questions",
                "you are an intelligent assistant",
                "chatinput",
                "conversation",
                "conversational",
                "rag chatbot",
            ],
        ):
            score += 1
        if set(outputs) == {"text"} and not inputs and "{{#" in text:
            score += 1
        if set(outputs) == {"text"}:
            score += 1
        if {"message", "prompt", "chatinput"} & set(inputs):
            score += 1
        if self._looks_like_chat_prompt(text):
            score += 2
        if self._match_any(
            text,
            [
                "第一人称",
                "角色",
                "回答规则",
                "对话原则",
                "tone",
                "回答的规则",
                "用户的问题",
                "assistant",
                "回答用户",
                "根据用户输入",
                "根据用户问题",
                "あなたは",
                "ユーザー",
                "回答のルール",
                "トーン",
                "役割",
                "相談",
                "案内してください",
            ],
        ):
            score += 1
        return score

    def _template_signal_strength(
        self,
        action_name: str,
        text: str,
        inputs: list[str],
        outputs: list[str],
    ) -> int:
        input_names = set(inputs)
        output_names = set(outputs)
        score = 0
        if "template" in input_names:
            score += 1
        if "output" in output_names:
            score += 1
        if self._match_any(
            action_name,
            [
                "edit fields",
                "map json",
                "set fields",
                "fill fields",
                "assign fields",
                "respond to user",
            ],
        ):
            score += 2
        if self._match_any(
            text,
            [
                "\"assignments\"",
                "assignments",
                "assignment",
                "\"value\"",
                "={{",
                "{{ $json",
            ],
        ):
            score += 1
        if "parameters" in output_names:
            score = max(0, score - 2)
        return score

    def _parameter_extractor_signal_strength(
        self,
        action_name: str,
        text: str,
        inputs: list[str],
        outputs: list[str],
    ) -> int:
        input_names = set(inputs)
        output_names = set(outputs)
        score = 0
        if "parameters" in output_names:
            score += 3
        if {"instruction", "query"} & input_names:
            score += 1
        if self._code_signal_strength(inputs, text) >= 3:
            score = max(0, score - 2)
        if self._match_any(
            action_name,
            [
                "information extractor",
                "extract meta data",
                "metadata extractor",
                "extractor",
                "qualifications",
                "personal data",
            ],
        ):
            score += 1
        if self._match_any(
            text,
            [
                "extract structured info",
                "structured info",
                "extract metadata",
                "classify the lead quality",
                "extract essential information",
                "extract key information",
            ],
        ):
            score += 1
        return score

    def _tool_signal_strength(
        self,
        resources: list[str],
        action_name: str,
        text: str,
    ) -> int:
        score = 0
        resource_text = " ".join(resources)
        if "provider_id" in resources:
            score += 2
        if "tool_name" in resources:
            score += 2
        if "file" in resources and self._match_any(action_name, ["parse file", "extract file", "mineru"]):
            score += 1
        if self._match_any(
            f"{resource_text} {action_name} {text}",
            [
                "google_drive",
                "google drive",
                "telegram",
                "gmail",
                "google sheets",
                "google_sheet",
                "airtable",
                "notion",
                "slack",
                "discord",
                "youtube",
                "calendar",
                "drive",
                "qdrant",
                "jina",
                "jina_reader",
                "firecrawl",
                "crawl",
                "parse file",
                "mineru",
                "current_time",
                "time",
            ],
        ):
            score += 1
        if self._match_any(text, ["integration", "connector", "plugin", "provider_id", "tool_name"]):
            score += 1
        return score

    def _http_signal_strength(
        self,
        input_names: list[str],
        output_names: list[str],
        action_name: str,
        text: str,
        available_params: dict[str, bool],
    ) -> int:
        score = 0
        explicit_io_signal = 0
        if any("url" in item or "endpoint" in item for item in input_names):
            score += 2
            explicit_io_signal += 2
        if "method" in input_names:
            score += 1
            explicit_io_signal += 1
        if "response" in output_names or "response" in input_names:
            score += 1
            explicit_io_signal += 1
        action_name_signal = self._match_any(
            action_name,
            [
                "download",
                "upload",
                "get status",
                "request",
                "fetch",
                "call",
                "invoke",
                "webhook",
            ],
        )
        if action_name_signal:
            score += 1
        if self._match_any(
            text,
            [
                "endpoint",
                "api",
                "webhook",
                "request_id",
                "request id",
                "headers",
                "status",
                "response",
                "http://",
                "https://",
                "queue.",
                "queue/",
            ],
        ) and (explicit_io_signal > 0 or action_name_signal):
            score += 1
        return score

    def _looks_like_iteration_action(self, inputs: list[str], text: str) -> bool:
        return any(self._has_collection_token(item) for item in inputs) and self._match_any(
            text,
            [
                "iterate",
                "for_each",
                "loop",
                "batch",
                "each item",
                "process each",
                "walk through",
                "one by one",
            ],
        )

    def _looks_like_condition_expression(self, text: str) -> bool:
        return self._condition_signal_strength(text) >= 2 or self._has_condition_route_phrase(text)

    def _looks_like_aggregation_action(
        self,
        inputs: list[str],
        outputs: list[str],
        text: str,
    ) -> bool:
        input_tokens = [token for item in inputs for token in self._tokenize(item)]
        output_tokens = [token for item in outputs for token in self._tokenize(item)]
        input_names = {item.lower() for item in inputs}
        output_names = {item.lower() for item in outputs}
        branch_like = {"branch", "result", "results", "reason", "reasons"}
        merge_terms_present = self._match_any(
            text,
            [
                "merge",
                "aggregate",
                "collect",
                "combine",
                "consolidate",
                "join",
                "fuse",
                "branch",
            ],
        )
        branch_hits = sum(1 for token in input_tokens if token in branch_like)

        if "variables" in input_names:
            return True
        if "parameters" in output_names:
            return False
        if (
            "template" in input_names
            and not merge_terms_present
            and branch_hits < 2
        ):
            return False
        if len(inputs) < 2:
            return False
        if merge_terms_present:
            return True
        if branch_hits >= 2:
            return True
        return (
            any(token in {"output", "packet", "payload"} for token in output_tokens)
            and branch_hits >= 1
        )

    def _condition_signal_strength(self, text: str) -> int:
        score = 0
        normalized_text = text.lower().replace("_", " ")
        tokens = set(self._tokenize(normalized_text))
        single_word_rules = {
            "if",
            "condition",
            "when",
            "check",
            "judge",
            "verify",
            "exists",
        }
        multi_word_rules = [
            "route by",
            "branch by",
            "is null",
            "is empty",
            "not exists",
            "greater than",
            "less than",
            "equal to",
        ]
        for phrase in single_word_rules:
            if phrase in tokens:
                score += 1
        for phrase in multi_word_rules:
            if phrase in normalized_text:
                score += 1
        return score

    def _looks_like_chat_prompt(self, description: str) -> bool:
        lowered = description.lower()
        prompt_markers = [
            "you are",
            "when answer to user",
            "role",
            "rules",
            "profile",
            "goals",
            "responsibilities",
            "tone",
            "persona",
            "assistant",
            "回答规则",
            "对话原则",
            "角色",
            "请根据用户输入",
            "根据用户输入",
            "用户的问题",
            "你是一位",
            "你是",
            "示例输出",
            "输出",
            "请",
            "markdown",
            "<context>",
        ]
        return any(marker in lowered for marker in prompt_markers)

    def _has_condition_route_phrase(self, text: str) -> bool:
        return self._match_any(
            text,
            [
                "route by",
                "branch by",
                "judge status",
                "check condition",
                "verify status",
            ],
        )

    def _rule_match(self, text: str, keyword: str) -> bool:
        if any(ord(ch) > 127 for ch in keyword):
            return keyword in text
        normalized_text = text.replace("_", " ")
        normalized_keyword = keyword.replace("_", " ")
        escaped = re.escape(keyword).replace(r"\ ", r"\s+")
        normalized_escaped = re.escape(normalized_keyword).replace(r"\ ", r"\s+")
        pattern = rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])"
        normalized_pattern = rf"(?<![a-z0-9_]){normalized_escaped}(?![a-z0-9_])"
        return bool(re.search(pattern, text) or re.search(normalized_pattern, normalized_text))

    def _dedupe(self, items: list[DifyNodeType]) -> list[DifyNodeType]:
        result: list[DifyNodeType] = []
        seen: set[DifyNodeType] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result
