import os
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


class PromptRunner:
    def __init__(self, model: str | None = None, base_url: str | None = None, backend: str | None = None):
        selected_backend = (backend or os.getenv("PROMPT_BACKEND", "ollama")).strip().lower()

        if selected_backend == "ollama":
            self.runner = OllamaPromptRunner(model=model, base_url=base_url)
        elif selected_backend == "hf":
            self.runner = HFSeq2SeqPromptRunner(model=model)
        else:
            raise ValueError(f"Unsupported PROMPT_BACKEND: {selected_backend}")

    def run(self, prompt: str, temperature: float = 0.0, max_tokens: int = 16) -> str:
        return self.runner.run(prompt, temperature=temperature, max_tokens=max_tokens)
