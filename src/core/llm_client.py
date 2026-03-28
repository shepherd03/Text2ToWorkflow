from typing import Any

import httpx

from .config import Settings
from .utils import safe_json_extract


class DeepSeekClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if not self.settings.llm_enabled:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置")
        url = f"{self.settings.deepseek_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.deepseek_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        return safe_json_extract(content)
