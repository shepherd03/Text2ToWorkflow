from __future__ import annotations

from typing import Any

from src.core.schema import DSLCompileOutput, DSLNormalizationOutput, DSLPrecheckOutput, SequentialBlock, UTR
from src.dsl_generation.compiler import MinimalDifyWorkflowCompiler
from src.dsl_generation.normalizer import DSLStructureNormalizer
from src.dsl_generation.validators import DSLInputValidator
from src.dsl_generation.workflow_validator import DifyWorkflowValidator


class DSLGenerationPipeline:
    def __init__(self) -> None:
        self.input_validator = DSLInputValidator()
        self.structure_normalizer = DSLStructureNormalizer()
        self.workflow_compiler = MinimalDifyWorkflowCompiler()
        self.workflow_validator = DifyWorkflowValidator()

    def validate_inputs(
        self,
        utr: UTR | dict[str, Any],
        skeleton: SequentialBlock | dict[str, Any],
    ) -> DSLPrecheckOutput:
        return self.input_validator.validate(utr, skeleton)

    def normalize_structure(self, precheck_output: DSLPrecheckOutput) -> DSLNormalizationOutput:
        return self.structure_normalizer.normalize(precheck_output)

    def run_step2(
        self,
        utr: UTR | dict[str, Any],
        skeleton: SequentialBlock | dict[str, Any],
    ) -> DSLNormalizationOutput:
        precheck_output = self.validate_inputs(utr, skeleton)
        return self.normalize_structure(precheck_output)

    def compile_minimal_workflow(
        self,
        normalization_output: DSLNormalizationOutput,
    ) -> DSLCompileOutput:
        compiled = self.workflow_compiler.compile(normalization_output)
        return self.workflow_validator.validate(compiled)

    def run_step3_minimal(
        self,
        utr: UTR | dict[str, Any],
        skeleton: SequentialBlock | dict[str, Any],
    ) -> DSLCompileOutput:
        normalization_output = self.run_step2(utr, skeleton)
        return self.compile_minimal_workflow(normalization_output)
