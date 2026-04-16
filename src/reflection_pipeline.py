from src.prompt_runner import PromptRunner
from src.controller import normalize_confidence, normalize_error_type
from src.utils import load_prompt, normalize_label


def safe_prompt_value(value, fallback: str = "unknown") -> str:
    if value is None:
        return fallback

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return fallback

    return text


class SimpleReflectionPipeline:
    def __init__(
        self,
        reflection_prompt_path: str = "prompts/simple_reflection.txt",
        model: str | None = None,
    ):
        self.runner = PromptRunner(model=model)
        self.reflection_prompt_template = load_prompt(reflection_prompt_path)

    def reflect_label(
        self,
        sentence: str,
        target: str,
        aspect: str,
        opinion: str,
        polarity_reasoning: str,
        initial_label: str,
    ) -> tuple[str, str]:
        prompt = self.reflection_prompt_template.format(
            sentence=safe_prompt_value(sentence),
            target=safe_prompt_value(target),
            aspect=safe_prompt_value(aspect),
            opinion=safe_prompt_value(opinion),
            polarity_reasoning=safe_prompt_value(polarity_reasoning),
            initial_label=safe_prompt_value(initial_label),
        )

        raw_output = self.runner.run(
            prompt,
            temperature=0.0,
            max_tokens=8,
        ).strip()

        return raw_output, normalize_label(raw_output)

    def run_one(
        self,
        sentence: str,
        target: str,
        aspect: str,
        opinion: str,
        polarity_reasoning: str,
        initial_label: str,
    ) -> dict:
        raw_output, prediction = self.reflect_label(
            sentence=sentence,
            target=target,
            aspect=aspect,
            opinion=opinion,
            polarity_reasoning=polarity_reasoning,
            initial_label=initial_label,
        )

        return {
            "reflection_raw_output": raw_output,
            "reflection_prediction": prediction,
        }


def parse_diagnostic_output(raw_output: str) -> dict:
    parsed = {}
    for line in raw_output.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        parsed[key.strip().lower()] = value.strip()

    return {
        "error_type": normalize_error_type(parsed.get("error_type", "")),
        "diagnostic_label": normalize_label(parsed.get("label", "")),
        "confidence": normalize_confidence(parsed.get("confidence", "")),
    }


class ErrorTypeReflectionPipeline:
    def __init__(
        self,
        reflection_prompt_path: str = "prompts/error_type_reflection.txt",
        model: str | None = None,
    ):
        self.runner = PromptRunner(model=model)
        self.reflection_prompt_template = load_prompt(reflection_prompt_path)

    def diagnose(
        self,
        sentence: str,
        target: str,
        direct_label: str,
        aspect: str,
        opinion: str,
        polarity_reasoning: str,
        thor_label: str,
    ) -> dict:
        prompt = self.reflection_prompt_template.format(
            sentence=safe_prompt_value(sentence),
            target=safe_prompt_value(target),
            direct_label=safe_prompt_value(direct_label),
            aspect=safe_prompt_value(aspect),
            opinion=safe_prompt_value(opinion),
            polarity_reasoning=safe_prompt_value(polarity_reasoning),
            thor_label=safe_prompt_value(thor_label),
        )

        raw_output = self.runner.run(
            prompt,
            temperature=0.0,
            max_tokens=32,
        ).strip()

        parsed = parse_diagnostic_output(raw_output)
        parsed["diagnostic_raw_output"] = raw_output
        return parsed
