import re
from pathlib import Path


VALID_OUTPUT_LABELS = ("positive", "negative", "neutral")


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def normalize_label(text: str) -> str:
    if not text:
        return "unknown"

    normalized = " ".join(str(text).strip().lower().split())
    matches = {
        label
        for label in VALID_OUTPUT_LABELS
        if re.search(rf"\b{re.escape(label)}\b", normalized)
    }
    return next(iter(matches)) if len(matches) == 1 else "unknown"
