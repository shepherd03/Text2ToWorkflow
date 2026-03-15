from fastapi import FastAPI
from pydantic import BaseModel

from utr_generator.pipeline import PipelineOutput, UTRGenerationPipeline

app = FastAPI(title="UTR Generator", version="1.0.0")
pipeline = UTRGenerationPipeline()


class GenerateRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/utr/generate", response_model=PipelineOutput)
def generate_utr(req: GenerateRequest) -> PipelineOutput:
    return pipeline.run(req.text)
