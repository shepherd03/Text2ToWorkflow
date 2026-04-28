from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from src.utr_generation.pipeline import UTRGenerationPipeline
from src.workflow_pipeline import WorkflowBuildPipeline
from src.core.schema import PipelineOutput, WorkflowBuildOutput

app = FastAPI(title="UTR Workflow Compiler", version="1.1.0")
utr_pipeline = UTRGenerationPipeline()
workflow_pipeline = WorkflowBuildPipeline()


class GenerateRequest(BaseModel):
    text: str


class WorkflowBuildRequest(BaseModel):
    text: str
    stage: Literal["utr", "skeleton", "dsl"] = "dsl"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/utr/generate", response_model=PipelineOutput)
def generate_utr(req: GenerateRequest) -> PipelineOutput:
    return utr_pipeline.run(req.text)


@app.post("/workflow/build", response_model=WorkflowBuildOutput)
def build_workflow(req: WorkflowBuildRequest) -> WorkflowBuildOutput:
    return workflow_pipeline.run(req.text, stage=req.stage)
