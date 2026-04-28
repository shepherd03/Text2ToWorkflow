from src.core.config import Settings
from src.core.schema import Action
from src.utr_generation.utr_core import UTRGenerator


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


def test_normalize_dependencies_filters_self_loops_duplicates_and_unknown_refs():
    generator = UTRGenerator(_build_settings())
    actions = [
        Action(action_id="act_1", action_name="step_one"),
        Action(action_id="act_2", action_name="step_two"),
    ]
    deps = [
        {"from": "act_1", "to": "act_1", "reason": "self loop"},
        {"from": "act_1", "to": "act_2", "reason": "valid edge"},
        {"from": "act_1", "to": "act_2", "reason": "duplicate edge"},
        {"from": "act_404", "to": "act_2", "reason": "unknown source"},
        {"from": "act_1", "to": "act_999", "reason": "unknown target"},
        {"from": "", "to": "act_2", "reason": "missing source"},
    ]

    normalized = generator._normalize_dependencies(actions, deps)

    assert normalized == [
        {"from": "act_1", "to": "act_2", "reason": "valid edge"},
    ]


def test_fallback_generation_meta_records_no_llm_call():
    generator = UTRGenerator(_build_settings())

    utr = generator.generate_utr("读取文章并生成摘要")

    assert utr.metadata.task_goal == "mock task"
    assert generator.last_generation_meta["generation_source"] == "fallback"
    assert generator.last_generation_meta["llm_call_count"] == 0
