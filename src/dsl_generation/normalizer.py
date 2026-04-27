from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from src.core.schema import (
    ActionSlot,
    BlockType,
    ConditionalBlock,
    DSLControlDomain,
    DSLNormalizationIssue,
    DSLNormalizationOutput,
    DSLNormalizationReport,
    DSLNormalizedActionSlot,
    DSLNormalizedBlock,
    DSLNormalizedContext,
    DSLPrecheckOutput,
    LoopBlock,
    ParallelBlock,
    SequentialBlock,
)
from src.core.utils import unique_keep_order


@dataclass
class _FlowResult:
    entry_slot_ids: list[str]
    exit_slot_ids: list[str]


class DSLStructureNormalizer:
    def normalize(self, precheck_output: DSLPrecheckOutput) -> DSLNormalizationOutput:
        precheck_report = precheck_output.report
        report = DSLNormalizationReport(success=True)

        if precheck_output.context is None or not precheck_report.passed:
            report.success = False
            self._add_error(report, "PRECHECK_NOT_PASSED", "输入前置校验未通过，无法进入归一化阶段", "precheck")
            return DSLNormalizationOutput(
                precheck_report=precheck_report,
                normalization_report=report,
                context=None,
            )

        source_context = precheck_output.context
        normalized_skeleton = source_context.skeleton.model_copy(deep=True)
        auto_else_blocks = self._ensure_default_else_branches(normalized_skeleton, report, path="root")

        action_slot_views: dict[str, DSLNormalizedActionSlot] = {}
        normalized_blocks: list[DSLNormalizedBlock] = []
        edges: list[tuple[str, str]] = []

        flow = self._analyze_block(
            block=normalized_skeleton,
            parent_block_id="",
            path="root",
            control_domains=[],
            action_slot_views=action_slot_views,
            normalized_blocks=normalized_blocks,
            edges=edges,
            auto_else_blocks=auto_else_blocks,
        )

        upstream_map: dict[str, list[str]] = defaultdict(list)
        downstream_map: dict[str, list[str]] = defaultdict(list)
        for source_slot_id, target_slot_id in edges:
            downstream_map[source_slot_id].append(target_slot_id)
            upstream_map[target_slot_id].append(source_slot_id)

        for slot_id, view in action_slot_views.items():
            view.upstream_slot_ids = unique_keep_order(upstream_map.get(slot_id, []))
            view.downstream_slot_ids = unique_keep_order(downstream_map.get(slot_id, []))

        block_stats = self._collect_block_stats(normalized_skeleton)
        context = DSLNormalizedContext(
            utr=source_context.utr,
            normalized_skeleton=normalized_skeleton,
            action_index=source_context.action_index,
            block_stats=block_stats,
            root_entry_slot_ids=flow.entry_slot_ids,
            root_exit_slot_ids=flow.exit_slot_ids,
            action_slots=list(action_slot_views.values()),
            blocks=normalized_blocks,
        )
        return DSLNormalizationOutput(
            precheck_report=precheck_report,
            normalization_report=report,
            context=context,
        )

    def _analyze_block(
        self,
        block: BlockType,
        parent_block_id: str,
        path: str,
        control_domains: list[DSLControlDomain],
        action_slot_views: dict[str, DSLNormalizedActionSlot],
        normalized_blocks: list[DSLNormalizedBlock],
        edges: list[tuple[str, str]],
        auto_else_blocks: set[str],
    ) -> _FlowResult:
        if isinstance(block, ActionSlot):
            slot_view = DSLNormalizedActionSlot(
                slot_id=block.id,
                action_id=block.action_id,
                action_name=block.action_name,
                parent_block_id=parent_block_id,
                path=path,
                control_domains=[domain.model_copy(deep=True) for domain in control_domains],
            )
            action_slot_views[block.id] = slot_view
            normalized_blocks.append(
                DSLNormalizedBlock(
                    block_id=block.id,
                    block_type=block.type,
                    parent_block_id=parent_block_id,
                    path=path,
                    entry_slot_ids=[block.id],
                    exit_slot_ids=[block.id],
                )
            )
            return _FlowResult(entry_slot_ids=[block.id], exit_slot_ids=[block.id])

        if isinstance(block, SequentialBlock):
            child_flow_results: list[_FlowResult] = []
            child_block_ids: list[str] = []
            for index, child in enumerate(block.children):
                child_result = self._analyze_block(
                    block=child,
                    parent_block_id=block.id,
                    path=f"{path}.children[{index}]",
                    control_domains=control_domains,
                    action_slot_views=action_slot_views,
                    normalized_blocks=normalized_blocks,
                    edges=edges,
                    auto_else_blocks=auto_else_blocks,
                )
                child_flow_results.append(child_result)
                child_block_ids.append(child.id)
            for index in range(1, len(child_flow_results)):
                for source_slot_id in child_flow_results[index - 1].exit_slot_ids:
                    for target_slot_id in child_flow_results[index].entry_slot_ids:
                        edges.append((source_slot_id, target_slot_id))
            entry_slot_ids = child_flow_results[0].entry_slot_ids if child_flow_results else []
            exit_slot_ids = child_flow_results[-1].exit_slot_ids if child_flow_results else []
            normalized_blocks.append(
                DSLNormalizedBlock(
                    block_id=block.id,
                    block_type=block.type,
                    parent_block_id=parent_block_id,
                    path=path,
                    child_block_ids=child_block_ids,
                    entry_slot_ids=entry_slot_ids,
                    exit_slot_ids=exit_slot_ids,
                )
            )
            return _FlowResult(entry_slot_ids=entry_slot_ids, exit_slot_ids=exit_slot_ids)

        if isinstance(block, ParallelBlock):
            branch_entry_slot_ids: list[str] = []
            branch_exit_slot_ids: list[str] = []
            child_block_ids: list[str] = []
            for index, branch in enumerate(block.branches):
                branch_domains = control_domains + [
                    DSLControlDomain(domain_type="parallel", block_id=block.id, branch=f"branch_{index}")
                ]
                branch_result = self._analyze_block(
                    block=branch,
                    parent_block_id=block.id,
                    path=f"{path}.branches[{index}]",
                    control_domains=branch_domains,
                    action_slot_views=action_slot_views,
                    normalized_blocks=normalized_blocks,
                    edges=edges,
                    auto_else_blocks=auto_else_blocks,
                )
                child_block_ids.append(branch.id)
                branch_entry_slot_ids.extend(branch_result.entry_slot_ids)
                branch_exit_slot_ids.extend(branch_result.exit_slot_ids)
            entry_slot_ids = unique_keep_order(branch_entry_slot_ids)
            exit_slot_ids = unique_keep_order(branch_exit_slot_ids)
            normalized_blocks.append(
                DSLNormalizedBlock(
                    block_id=block.id,
                    block_type=block.type,
                    parent_block_id=parent_block_id,
                    path=path,
                    child_block_ids=child_block_ids,
                    entry_slot_ids=entry_slot_ids,
                    exit_slot_ids=exit_slot_ids,
                    needs_join=len(block.branches) > 1,
                )
            )
            return _FlowResult(entry_slot_ids=entry_slot_ids, exit_slot_ids=exit_slot_ids)

        if isinstance(block, ConditionalBlock):
            branch_entry_slot_ids: list[str] = []
            branch_exit_slot_ids: list[str] = []
            child_block_ids: list[str] = []
            for branch_name, branch in block.branches.items():
                branch_domains = control_domains + [
                    DSLControlDomain(domain_type="conditional", block_id=block.id, branch=branch_name)
                ]
                branch_result = self._analyze_block(
                    block=branch,
                    parent_block_id=block.id,
                    path=f"{path}.branches.{branch_name}",
                    control_domains=branch_domains,
                    action_slot_views=action_slot_views,
                    normalized_blocks=normalized_blocks,
                    edges=edges,
                    auto_else_blocks=auto_else_blocks,
                )
                child_block_ids.append(branch.id)
                branch_entry_slot_ids.extend(branch_result.entry_slot_ids)
                branch_exit_slot_ids.extend(branch_result.exit_slot_ids)
            entry_slot_ids = unique_keep_order(branch_entry_slot_ids)
            exit_slot_ids = unique_keep_order(branch_exit_slot_ids)
            normalized_blocks.append(
                DSLNormalizedBlock(
                    block_id=block.id,
                    block_type=block.type,
                    parent_block_id=parent_block_id,
                    path=path,
                    child_block_ids=child_block_ids,
                    entry_slot_ids=entry_slot_ids,
                    exit_slot_ids=exit_slot_ids,
                    needs_join=len(block.branches) > 1,
                    has_default_else=block.id in auto_else_blocks,
                )
            )
            return _FlowResult(entry_slot_ids=entry_slot_ids, exit_slot_ids=exit_slot_ids)

        if isinstance(block, LoopBlock):
            loop_domains = control_domains + [
                DSLControlDomain(domain_type="loop", block_id=block.id, branch="body")
            ]
            body_result = self._analyze_block(
                block=block.body,
                parent_block_id=block.id,
                path=f"{path}.body",
                control_domains=loop_domains,
                action_slot_views=action_slot_views,
                normalized_blocks=normalized_blocks,
                edges=edges,
                auto_else_blocks=auto_else_blocks,
            )
            normalized_blocks.append(
                DSLNormalizedBlock(
                    block_id=block.id,
                    block_type=block.type,
                    parent_block_id=parent_block_id,
                    path=path,
                    child_block_ids=[block.body.id],
                    entry_slot_ids=body_result.entry_slot_ids,
                    exit_slot_ids=body_result.exit_slot_ids,
                )
            )
            return _FlowResult(
                entry_slot_ids=body_result.entry_slot_ids,
                exit_slot_ids=body_result.exit_slot_ids,
            )

        return _FlowResult(entry_slot_ids=[], exit_slot_ids=[])

    def _ensure_default_else_branches(
        self,
        block: BlockType,
        report: DSLNormalizationReport,
        path: str,
    ) -> set[str]:
        auto_else_blocks: set[str] = set()
        self._walk_else_completion(block, report, path, auto_else_blocks)
        return auto_else_blocks

    def _walk_else_completion(
        self,
        block: BlockType,
        report: DSLNormalizationReport,
        path: str,
        auto_else_blocks: set[str],
    ) -> None:
        if isinstance(block, SequentialBlock):
            for index, child in enumerate(block.children):
                self._walk_else_completion(
                    child,
                    report,
                    f"{path}.children[{index}]",
                    auto_else_blocks,
                )
            return

        if isinstance(block, ParallelBlock):
            for index, branch in enumerate(block.branches):
                self._walk_else_completion(
                    branch,
                    report,
                    f"{path}.branches[{index}]",
                    auto_else_blocks,
                )
            return

        if isinstance(block, ConditionalBlock):
            has_false_branch = "false" in block.branches
            has_else_branch = "else" in block.branches
            if not has_false_branch and not has_else_branch:
                block.branches["else"] = SequentialBlock(children=[])
                auto_else_blocks.add(block.id)
                self._add_warning(
                    report,
                    "CONDITIONAL_DEFAULT_ELSE_ADDED",
                    "ConditionalBlock 缺少 false/else 分支，已自动补全空 else 分支",
                    path,
                )
            for branch_name, branch in block.branches.items():
                self._walk_else_completion(
                    branch,
                    report,
                    f"{path}.branches.{branch_name}",
                    auto_else_blocks,
                )
            return

        if isinstance(block, LoopBlock):
            self._walk_else_completion(block.body, report, f"{path}.body", auto_else_blocks)

    def _collect_block_stats(self, skeleton: SequentialBlock) -> dict[str, int]:
        counter: Counter[str] = Counter()
        self._walk_for_stats(skeleton, counter)
        return dict(counter)

    def _walk_for_stats(self, block: BlockType, counter: Counter[str]) -> None:
        counter[block.type] += 1
        if isinstance(block, SequentialBlock):
            for child in block.children:
                self._walk_for_stats(child, counter)
            return
        if isinstance(block, ParallelBlock):
            for branch in block.branches:
                self._walk_for_stats(branch, counter)
            return
        if isinstance(block, ConditionalBlock):
            for branch in block.branches.values():
                self._walk_for_stats(branch, counter)
            return
        if isinstance(block, LoopBlock):
            self._walk_for_stats(block.body, counter)

    def _add_error(self, report: DSLNormalizationReport, code: str, message: str, path: str = "") -> None:
        report.errors.append(message)
        report.issues.append(
            DSLNormalizationIssue(
                code=code,
                message=message,
                path=path,
                severity="error",
            )
        )

    def _add_warning(self, report: DSLNormalizationReport, code: str, message: str, path: str = "") -> None:
        report.warnings.append(message)
        report.issues.append(
            DSLNormalizationIssue(
                code=code,
                message=message,
                path=path,
                severity="warning",
            )
        )
