import os
import requests


class PromptRunner:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def run(self, prompt: str, temperature: float = 0.0, max_tokens: int = 16) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()
