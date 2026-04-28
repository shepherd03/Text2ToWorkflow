from __future__ import annotations

from src.core.config import Settings, load_settings
from src.core.schema import WorkflowBuildOutput
from src.dsl_generation.pipeline import DSLGenerationPipeline
from src.skeleton_planning.skeleton_planner import SkeletonPlanner
from src.utr_generation.pipeline import UTRGenerationPipeline


class WorkflowBuildPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.utr_pipeline = UTRGenerationPipeline(self.settings)
        self.skeleton_planner = SkeletonPlanner(self.settings)
        self.dsl_pipeline = DSLGenerationPipeline()

    def run(self, task_text: str, stage: str = "dsl") -> WorkflowBuildOutput:
        normalized_stage = stage.lower()
        if normalized_stage not in {"utr", "skeleton", "dsl"}:
            raise ValueError("stage must be one of: utr, skeleton, dsl")

        utr_output = self.utr_pipeline.run(task_text)
        output = WorkflowBuildOutput(
            utr_output=utr_output,
            stage=normalized_stage,
            success=utr_output.report.passed,
            warnings=list(utr_output.report.warnings),
            meta={
                "llm_enabled": self.settings.llm_enabled,
                "stage": normalized_stage,
            },
        )
        if not utr_output.report.passed or normalized_stage == "utr":
            output.errors.extend(utr_output.report.errors)
            return output

        try:
            output.skeleton = self.skeleton_planner.plan(utr_output.utr)
        except Exception as exc:
            output.success = False
            output.errors.append(f"Skeleton planning failed: {exc}")
            return output

        if normalized_stage == "skeleton":
            return output

        output.dsl_output = self.dsl_pipeline.run_step3_minimal(utr_output.utr, output.skeleton)
        if not output.dsl_output.compile_report.success:
            output.success = False
            output.errors.extend(output.dsl_output.compile_report.errors)
        return output
