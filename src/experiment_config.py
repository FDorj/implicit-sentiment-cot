from pathlib import Path
import os
import re


DEFAULT_DATA_PATH = "data/processed/semeval14_scapt_isa_only_clean.csv"


def data_path(default: str = DEFAULT_DATA_PATH) -> str:
    return os.getenv("DATA_PATH", default)


def parse_debug_n(default: int | None = 20) -> int | None:
    raw_value = os.getenv("DEBUG_N")
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"", "all", "none", "full"}:
        return None

    return int(raw_value)


def slugify_model_id(model_id: str) -> str:
    slug = model_id.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "model"


def get_model_id() -> str:
    backend = os.getenv("PROMPT_BACKEND", "ollama").strip().lower()
    if backend == "hf":
        return os.getenv("HF_MODEL", "google/flan-t5-large")
    if backend in {"openai_compatible", "openai-compatible", "openai"}:
        return os.getenv("OPENAI_COMPAT_MODEL", "Gemini-2.5-Flash")

    return os.getenv("OLLAMA_MODEL", "qwen3:8b")


def get_experiment_id(default: str | None = None) -> str | None:
    explicit_id = os.getenv("EXPERIMENT_ID")
    if explicit_id:
        return slugify_model_id(explicit_id)

    if default is not None:
        return slugify_model_id(default)

    return None


def result_path(stem: str, suffix: str, env_var: str | None = None, default_experiment_id: str | None = None) -> str:
    if env_var and os.getenv(env_var):
        return os.getenv(env_var, "")

    experiment_id = get_experiment_id(default=default_experiment_id)
    filename = f"{experiment_id}_{stem}_{suffix}" if experiment_id else f"{stem}_{suffix}"
    return str(Path("results") / filename)


def describe_runtime() -> str:
    backend = os.getenv("PROMPT_BACKEND", "ollama").strip().lower()
    return f"backend={backend}, model={get_model_id()}"
