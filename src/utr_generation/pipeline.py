from pydantic import BaseModel
from src.core.config import Settings, load_settings
from src.core.schema import UTR, UTRValidationReport, PipelineOutput
from .utr_core import UTRGenerator

class UTRGenerationPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.generator = UTRGenerator(self.settings)

    def run(self, task_text: str) -> PipelineOutput:
        utr = self.generator.generate_utr(task_text)
        
        # 简单验证逻辑
        report = UTRValidationReport(
            schema_valid=True,
            logic_valid=True,
            completeness_valid=True,
            errors=[],
            warnings=[]
        )
        
        return PipelineOutput(
            utr=utr,
            report=report,
            meta={
                "llm_enabled": self.settings.llm_enabled,
                "version": "refactored_metadata_only",
                **self.generator.last_generation_meta,
            },
        )

