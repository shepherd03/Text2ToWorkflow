from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import ValidationError

from src.core.schema import (
    Action,
    ActionSlot,
    BlockType,
    ConditionalBlock,
    DSLInputContext,
    DSLPrecheckIssue,
    DSLPrecheckOutput,
    DSLPrecheckReport,
    LoopBlock,
    ParallelBlock,
    SequentialBlock,
    UTR,
)


class DSLInputValidator:
    _allowed_types = {"ActionSlot", "Sequential", "Parallel", "Conditional", "Loop"}

    def validate(
        self,
        utr_input: UTR | dict[str, Any],
        skeleton_input: SequentialBlock | dict[str, Any],
    ) -> DSLPrecheckOutput:
        report = DSLPrecheckReport()
        utr = self._parse_utr(utr_input, report)
        skeleton = self._parse_skeleton(skeleton_input, report)
        if utr is None or skeleton is None:
            report.schema_valid = False
            return DSLPrecheckOutput(context=None, report=report)

        action_index = self._build_action_index(utr, report)
        self._validate_block_types(skeleton, report, path="root")
        self._validate_action_refs(skeleton, action_index, report, path="root")
        self._validate_structure(skeleton, report, path="root")

        if not report.node_type_valid or not report.action_ref_valid or not report.structure_valid:
            return DSLPrecheckOutput(context=None, report=report)

        stats = self._collect_block_stats(skeleton)
        context = DSLInputContext(
            utr=utr,
            skeleton=skeleton,
            action_index=action_index,
            block_stats=stats,
        )
        return DSLPrecheckOutput(context=context, report=report)

    def _parse_utr(
        self,
        utr_input: UTR | dict[str, Any],
        report: DSLPrecheckReport,
    ) -> UTR | None:
        if isinstance(utr_input, UTR):
            return utr_input
        try:
            return UTR.model_validate(utr_input)
        except ValidationError as exc:
            self._add_error(report, "UTR_SCHEMA_INVALID", f"UTR Schema 校验失败: {exc}", "utr")
            return None

    def _parse_skeleton(
        self,
        skeleton_input: SequentialBlock | dict[str, Any],
        report: DSLPrecheckReport,
    ) -> SequentialBlock | None:
        if isinstance(skeleton_input, SequentialBlock):
            return skeleton_input
        try:
            skeleton = SequentialBlock.model_validate(skeleton_input)
        except ValidationError as exc:
            self._add_error(
                report,
                "SKELETON_SCHEMA_INVALID",
                f"Skeleton Schema 校验失败: {exc}",
                "skeleton",
            )
            return None
        if skeleton.type != "Sequential":
            self._add_error(
                report,
                "SKELETON_ROOT_INVALID",
                "Skeleton 根节点必须为 Sequential 类型",
                "skeleton.type",
            )
            report.structure_valid = False
            return None
        return skeleton

    def _build_action_index(self, utr: UTR, report: DSLPrecheckReport) -> dict[str, Action]:
        index: dict[str, Action] = {}
        for action in utr.metadata.core_actions:
            action_id = action.action_id.strip()
            if not action_id:
                self._add_error(
                    report,
                    "ACTION_ID_EMPTY",
                    "UTR metadata.core_actions 中存在空 action_id",
                    "utr.metadata.core_actions",
                )
                report.action_ref_valid = False
                continue
            if action_id in index:
                self._add_error(
                    report,
                    "ACTION_ID_DUPLICATED",
                    f"UTR metadata.core_actions 中存在重复 action_id: {action_id}",
                    "utr.metadata.core_actions",
                )
                report.action_ref_valid = False
                continue
            index[action_id] = action
        return index

    def _validate_block_types(
        self,
        block: BlockType,
        report: DSLPrecheckReport,
        path: str,
    ) -> None:
        if block.type not in self._allowed_types:
            self._add_error(
                report,
                "BLOCK_TYPE_NOT_ALLOWED",
                f"发现不支持的骨架节点类型: {block.type}",
                f"{path}.type",
            )
            report.node_type_valid = False

        if isinstance(block, SequentialBlock):
            for index, child in enumerate(block.children):
                self._validate_block_types(child, report, f"{path}.children[{index}]")
            return

        if isinstance(block, ParallelBlock):
            for index, branch in enumerate(block.branches):
                self._validate_block_types(branch, report, f"{path}.branches[{index}]")
            return

        if isinstance(block, ConditionalBlock):
            for branch_name, branch in block.branches.items():
                self._validate_block_types(branch, report, f"{path}.branches.{branch_name}")
            return

        if isinstance(block, LoopBlock):
            self._validate_block_types(block.body, report, f"{path}.body")

    def _validate_action_refs(
        self,
        block: BlockType,
        action_index: dict[str, Action],
        report: DSLPrecheckReport,
        path: str,
    ) -> None:
        if isinstance(block, ActionSlot):
            if block.action_id in {"start_node", "end_node"}:
                return
            if block.action_id not in action_index:
                self._add_error(
                    report,
                    "ACTION_REF_NOT_FOUND",
                    f"ActionSlot 引用了不存在的 action_id: {block.action_id}",
                    f"{path}.action_id",
                )
                report.action_ref_valid = False
            return

        if isinstance(block, SequentialBlock):
            for index, child in enumerate(block.children):
                self._validate_action_refs(child, action_index, report, f"{path}.children[{index}]")
            return

        if isinstance(block, ParallelBlock):
            for index, branch in enumerate(block.branches):
                self._validate_action_refs(branch, action_index, report, f"{path}.branches[{index}]")
            return

        if isinstance(block, ConditionalBlock):
            for branch_name, branch in block.branches.items():
                self._validate_action_refs(branch, action_index, report, f"{path}.branches.{branch_name}")
            return

        if isinstance(block, LoopBlock):
            self._validate_action_refs(block.body, action_index, report, f"{path}.body")

    def _validate_structure(
        self,
        skeleton: SequentialBlock,
        report: DSLPrecheckReport,
        path: str,
    ) -> None:
        if not skeleton.children:
            self._add_error(
                report,
                "SKELETON_EMPTY",
                "Skeleton 根节点 children 不能为空",
                f"{path}.children",
            )
            report.structure_valid = False
            return

        action_slots = self._collect_action_slots(skeleton)
        if len(action_slots) < 2:
            self._add_error(
                report,
                "START_END_MISSING",
                "Skeleton 至少应包含 start_node 与 end_node",
                path,
            )
            report.structure_valid = False
        else:
            first_slot = action_slots[0]
            last_slot = action_slots[-1]
            if first_slot.action_id != "start_node":
                self._add_error(
                    report,
                    "START_POSITION_INVALID",
                    "start_node 必须是首个 ActionSlot",
                    path,
                )
                report.structure_valid = False
            if last_slot.action_id != "end_node":
                self._add_error(
                    report,
                    "END_POSITION_INVALID",
                    "end_node 必须是最后一个 ActionSlot",
                    path,
                )
                report.structure_valid = False

        self._validate_nested_structure(skeleton, report, path)

    def _validate_nested_structure(
        self,
        block: BlockType,
        report: DSLPrecheckReport,
        path: str,
    ) -> None:
        if isinstance(block, ParallelBlock):
            if not block.branches:
                self._add_error(
                    report,
                    "PARALLEL_EMPTY_BRANCHES",
                    "ParallelBlock 不能为空分支",
                    f"{path}.branches",
                )
                report.structure_valid = False
            for index, branch in enumerate(block.branches):
                if not branch.children:
                    self._add_error(
                        report,
                        "PARALLEL_BRANCH_EMPTY",
                        "ParallelBlock 的分支 SequentialBlock children 不能为空",
                        f"{path}.branches[{index}].children",
                    )
                    report.structure_valid = False
                self._validate_nested_structure(branch, report, f"{path}.branches[{index}]")
            return

        if isinstance(block, ConditionalBlock):
            if not block.branches:
                self._add_error(
                    report,
                    "CONDITIONAL_EMPTY_BRANCHES",
                    "ConditionalBlock 至少包含一个分支",
                    f"{path}.branches",
                )
                report.structure_valid = False
            for branch_name, branch in block.branches.items():
                if not branch_name.strip():
                    self._add_error(
                        report,
                        "CONDITIONAL_BRANCH_NAME_EMPTY",
                        "ConditionalBlock 存在空分支名",
                        f"{path}.branches",
                    )
                    report.structure_valid = False
                if not branch.children:
                    self._add_error(
                        report,
                        "CONDITIONAL_BRANCH_EMPTY",
                        "ConditionalBlock 分支 SequentialBlock children 不能为空",
                        f"{path}.branches.{branch_name}.children",
                    )
                    report.structure_valid = False
                self._validate_nested_structure(branch, report, f"{path}.branches.{branch_name}")
            return

        if isinstance(block, LoopBlock):
            if not block.body.children:
                self._add_error(
                    report,
                    "LOOP_BODY_EMPTY",
                    "LoopBlock 的 body.children 不能为空",
                    f"{path}.body.children",
                )
                report.structure_valid = False
            self._validate_nested_structure(block.body, report, f"{path}.body")
            return

        if isinstance(block, SequentialBlock):
            for index, child in enumerate(block.children):
                self._validate_nested_structure(child, report, f"{path}.children[{index}]")

    def _collect_action_slots(self, block: BlockType) -> list[ActionSlot]:
        result: list[ActionSlot] = []
        if isinstance(block, ActionSlot):
            result.append(block)
            return result
        if isinstance(block, SequentialBlock):
            for child in block.children:
                result.extend(self._collect_action_slots(child))
            return result
        if isinstance(block, ParallelBlock):
            for branch in block.branches:
                result.extend(self._collect_action_slots(branch))
            return result
        if isinstance(block, ConditionalBlock):
            for branch in block.branches.values():
                result.extend(self._collect_action_slots(branch))
            return result
        if isinstance(block, LoopBlock):
            result.extend(self._collect_action_slots(block.body))
            return result
        return result

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

    def _add_error(self, report: DSLPrecheckReport, code: str, message: str, path: str = "") -> None:
        report.errors.append(message)
        report.issues.append(
            DSLPrecheckIssue(
                code=code,
                message=message,
                path=path,
                severity="error",
            )
        )
