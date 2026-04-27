from pathlib import Path

from src.core.config import Settings
from src.dsl_generation.semantic_retriever import (
    HybridSemanticBackend,
    JsonEmbeddingCache,
    LocalHashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    RemoteEmbeddingBackend,
    TfidfSemanticBackend,
    build_semantic_backend,
)


def _settings(semantic_backend: str) -> Settings:
    return Settings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        semantic_backend=semantic_backend,
        semantic_embedding_provider="local-hash",
        semantic_embedding_model="embedding-default",
        semantic_embedding_api_key="",
        semantic_embedding_base_url="https://api.openai.com/v1",
        semantic_embedding_cache_path="generated_data/semantic_cache/test_embeddings.json",
        strict_completeness=False,
    )


def test_build_semantic_backend_defaults_to_tfidf():
    backend = build_semantic_backend(_settings("tfidf"))
    assert isinstance(backend, TfidfSemanticBackend)
    assert backend.backend_name == "tfidf"


def test_build_semantic_backend_supports_remote_embedding_mode():
    backend = build_semantic_backend(_settings("embedding"))
    assert isinstance(backend, RemoteEmbeddingBackend)
    assert backend.backend_name == "remote-embedding"


def test_unknown_backend_falls_back_to_tfidf():
    backend = build_semantic_backend(_settings("unknown-backend"))
    assert isinstance(backend, TfidfSemanticBackend)


def test_build_semantic_backend_supports_hybrid_mode():
    backend = build_semantic_backend(_settings("hybrid"))
    assert isinstance(backend, HybridSemanticBackend)
    assert backend.backend_name == "hybrid"


def test_json_embedding_cache_roundtrip(tmp_path: Path):
    cache = JsonEmbeddingCache(str(tmp_path / "embeddings.json"))
    cache.set("abc", [0.1, 0.2, 0.3])
    assert cache.get("abc") == [0.1, 0.2, 0.3]


def test_remote_embedding_backend_falls_back_without_api_key(tmp_path: Path):
    settings = Settings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        semantic_backend="embedding",
        semantic_embedding_provider="local-hash",
        semantic_embedding_model="embedding-default",
        semantic_embedding_api_key="",
        semantic_embedding_base_url="https://api.openai.com/v1",
        semantic_embedding_cache_path=str(tmp_path / "embeddings.json"),
        strict_completeness=False,
    )
    backend = RemoteEmbeddingBackend(
        settings=settings,
        provider=LocalHashEmbeddingProvider(),
        fallback_backend=TfidfSemanticBackend(),
        cache=JsonEmbeddingCache(settings.semantic_embedding_cache_path),
    )
    results = backend.search("draft a customer-facing response", top_k=3)
    assert results
    assert backend.backend_name == "remote-embedding"


class _FailingProvider:
    provider_name = "failing"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("boom")


def test_remote_embedding_backend_falls_back_on_provider_error(tmp_path: Path):
    settings = Settings(
        deepseek_api_key="non-empty-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        semantic_backend="embedding",
        semantic_embedding_provider="local-hash",
        semantic_embedding_model="embedding-default",
        semantic_embedding_api_key="fake-key",
        semantic_embedding_base_url="https://api.openai.com/v1",
        semantic_embedding_cache_path=str(tmp_path / "embeddings.json"),
        strict_completeness=False,
    )
    backend = RemoteEmbeddingBackend(
        settings=settings,
        provider=_FailingProvider(),
        fallback_backend=TfidfSemanticBackend(),
        cache=JsonEmbeddingCache(settings.semantic_embedding_cache_path),
    )
    results = backend.search("call external endpoint with url and post method", top_k=3)
    assert results


def test_openai_compatible_provider_requires_api_key():
    settings = Settings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        semantic_backend="embedding",
        semantic_embedding_provider="openai-compatible",
        semantic_embedding_model="text-embedding-3-small",
        semantic_embedding_api_key="",
        semantic_embedding_base_url="https://api.openai.com/v1",
        semantic_embedding_cache_path="generated_data/semantic_cache/test_embeddings.json",
        strict_completeness=False,
    )
    provider = OpenAICompatibleEmbeddingProvider(settings)
    try:
        provider.embed(["hello"])
    except RuntimeError as exc:
        assert "SEMANTIC_EMBEDDING_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing api key error")
