import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
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
        strict_completeness=strict_raw in {"1", "true", "yes", "on"},
    )
