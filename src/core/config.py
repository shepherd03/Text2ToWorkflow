import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    semantic_backend: str
    semantic_embedding_provider: str
    semantic_embedding_model: str
    semantic_embedding_api_key: str
    semantic_embedding_base_url: str
    semantic_embedding_cache_path: str
    strict_completeness: bool

    @property
    def llm_enabled(self) -> bool:
        return bool(self.deepseek_api_key.strip())


def load_settings() -> Settings:
    load_dotenv()
    strict_raw = os.getenv("UTR_STRICT_COMPLETENESS", "false").strip().lower()
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        semantic_backend=os.getenv("SEMANTIC_BACKEND", "tfidf"),
        semantic_embedding_provider=os.getenv("SEMANTIC_EMBEDDING_PROVIDER", "openai-compatible"),
        semantic_embedding_model=os.getenv("SEMANTIC_EMBEDDING_MODEL", "embedding-default"),
        semantic_embedding_api_key=os.getenv(
            "SEMANTIC_EMBEDDING_API_KEY",
            os.getenv("DEEPSEEK_API_KEY", ""),
        ),
        semantic_embedding_base_url=os.getenv(
            "SEMANTIC_EMBEDDING_BASE_URL",
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ),
        semantic_embedding_cache_path=os.getenv(
            "SEMANTIC_EMBEDDING_CACHE_PATH",
            "generated_data/semantic_cache/embeddings.json",
        ),
        strict_completeness=strict_raw in {"1", "true", "yes", "on"},
    )
