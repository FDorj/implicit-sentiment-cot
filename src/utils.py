from pathlib import Path


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def normalize_label(text: str) -> str:
    if not text:
        return "unknown"

    t = text.strip().lower()

    if t == "positive" or t.startswith("positive"):
        return "positive"
    if t == "negative" or t.startswith("negative"):
        return "negative"
    if t == "neutral" or t.startswith("neutral"):
        return "neutral"

    if "positive" in t:
        return "positive"
    if "negative" in t:
        return "negative"
    if "neutral" in t:
        return "neutral"

    return "unknown"