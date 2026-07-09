import os
import json
import time
from pathlib import Path
import requests


class OllamaPromptRunner:
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


class HFSeq2SeqPromptRunner:
    def __init__(self, model: str | None = None, device: str | None = None):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "PROMPT_BACKEND=hf requires transformers and torch. "
                "Install them before running Flan-T5 experiments."
            ) from exc

        self.model = model or os.getenv("HF_MODEL", "google/flan-t5-large")
        self.device = int(device or os.getenv("HF_DEVICE", "-1"))
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model_obj = AutoModelForSeq2SeqLM.from_pretrained(self.model)
        self.generator = pipeline(
            "text2text-generation",
            model=model_obj,
            tokenizer=tokenizer,
            device=self.device,
        )

    def run(self, prompt: str, temperature: float = 0.0, max_tokens: int = 16) -> str:
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "num_return_sequences": 1,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        outputs = self.generator(prompt, **generation_kwargs)
        return outputs[0]["generated_text"].strip()


class OpenAICompatiblePromptRunner:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        session=None,
    ):
        self.model = model or os.getenv("OPENAI_COMPAT_MODEL", "Gemini-2.5-Flash")
        self.base_url = (base_url or os.getenv("OPENAI_COMPAT_BASE_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("OPENAI_COMPAT_BASE_URL is required for PROMPT_BACKEND=openai_compatible.")

        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_COMPAT_API_KEY", "")
        self.timeout = int(os.getenv("OPENAI_COMPAT_TIMEOUT_SECONDS", "300"))
        self.min_max_tokens = int(os.getenv("OPENAI_COMPAT_MIN_MAX_TOKENS", "0"))
        self.max_retries = int(os.getenv("OPENAI_COMPAT_MAX_RETRIES", "4"))
        self.retry_sleep_seconds = float(os.getenv("OPENAI_COMPAT_RETRY_SLEEP_SECONDS", "5"))
        self.empty_length_retries = int(os.getenv("OPENAI_COMPAT_EMPTY_LENGTH_RETRIES", "2"))
        self.empty_content_retries = int(os.getenv("OPENAI_COMPAT_EMPTY_CONTENT_RETRIES", "2"))
        self.session = session or requests.Session()

    @staticmethod
    def is_retryable_error(exc: requests.HTTPError) -> bool:
        status_code = getattr(exc.response, "status_code", None)
        return status_code == 429 or (status_code is not None and 500 <= status_code < 600)

    @staticmethod
    def is_retryable_request_error(exc: requests.RequestException) -> bool:
        return isinstance(
            exc,
            (
                requests.ConnectionError,
                requests.Timeout,
            ),
        )

    def run(self, prompt: str, temperature: float = 0.0, max_tokens: int = 16) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        effective_max_tokens = max(max_tokens, self.min_max_tokens)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": False,
        }

        max_empty_retries = max(self.empty_length_retries, self.empty_content_retries)
        for empty_attempt in range(max_empty_retries + 1):
            payload["max_tokens"] = effective_max_tokens
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    break
                except requests.HTTPError as exc:
                    if attempt >= self.max_retries or not self.is_retryable_error(exc):
                        raise
                    time.sleep(self.retry_sleep_seconds * (attempt + 1))
                except requests.RequestException as exc:
                    if attempt >= self.max_retries or not self.is_retryable_request_error(exc):
                        raise
                    time.sleep(self.retry_sleep_seconds * (attempt + 1))

            data = response.json()
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content", "").strip()
            if content:
                return content

            finish_reason = choice.get("finish_reason", "unknown")
            if finish_reason == "length" and empty_attempt < self.empty_length_retries:
                effective_max_tokens *= 2
                continue
            if empty_attempt < self.empty_content_retries:
                continue

            choice_summary = json.dumps(choice, ensure_ascii=True)[:1000]
            diagnostic_path = os.getenv(
                "OPENAI_COMPAT_EMPTY_RESPONSE_PATH",
                str(Path("results") / "openai_compatible_empty_response.json"),
            )
            diagnostic_file = Path(diagnostic_path)
            diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_file.write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "finish_reason": finish_reason,
                        "choice": choice,
                        "response": data,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise ValueError(
                "OpenAI-compatible API returned empty content; "
                f"finish_reason={finish_reason}; diagnostic_path={diagnostic_file}; choice={choice_summary}"
            )

        raise RuntimeError("unreachable")


class PromptRunner:
    def __init__(self, model: str | None = None, base_url: str | None = None, backend: str | None = None):
        selected_backend = (backend or os.getenv("PROMPT_BACKEND", "ollama")).strip().lower()

        if selected_backend == "ollama":
            self.runner = OllamaPromptRunner(model=model, base_url=base_url)
        elif selected_backend == "hf":
            self.runner = HFSeq2SeqPromptRunner(model=model)
        elif selected_backend in {"openai_compatible", "openai-compatible", "openai"}:
            self.runner = OpenAICompatiblePromptRunner(model=model, base_url=base_url)
        else:
            raise ValueError(f"Unsupported PROMPT_BACKEND: {selected_backend}")

    def run(self, prompt: str, temperature: float = 0.0, max_tokens: int = 16) -> str:
        return self.runner.run(prompt, temperature=temperature, max_tokens=max_tokens)
