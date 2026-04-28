import pytest

from src.core.config import Settings
from src.workflow_pipeline import WorkflowBuildPipeline


def _build_settings() -> Settings:
    return Settings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        semantic_backend="tfidf",
        semantic_embedding_provider="local-hash",
        semantic_embedding_model="embedding-default",
        semantic_embedding_api_key="",
        semantic_embedding_base_url="https://api.openai.com/v1",
        semantic_embedding_cache_path="generated_data/semantic_cache/test_embeddings.json",
        strict_completeness=False,
    )


def test_workflow_build_pipeline_runs_to_dsl_without_llm_key():
    pipeline = WorkflowBuildPipeline(_build_settings())

    output = pipeline.run("读取文章并生成摘要", stage="dsl")

    assert output.success is True
    assert output.stage == "dsl"
    assert output.utr_output.report.passed is True
    assert output.skeleton is not None
    assert output.dsl_output is not None
    assert output.dsl_output.compile_report.success is True
    assert output.dsl_output.workflow is not None
    assert output.dsl_output.workflow.app["mode"] == "workflow"


def test_workflow_build_pipeline_can_stop_at_skeleton():
    pipeline = WorkflowBuildPipeline(_build_settings())

    output = pipeline.run("读取文章并生成摘要", stage="skeleton")

    assert output.success is True
    assert output.skeleton is not None
    assert output.dsl_output is None


def test_workflow_build_pipeline_rejects_unknown_stage():
    pipeline = WorkflowBuildPipeline(_build_settings())

    with pytest.raises(ValueError, match="stage must be one of"):
        pipeline.run("读取文章并生成摘要", stage="unknown")
