import json
import os
import re

from src.prompt_runner import PromptRunner
from src.controller import normalize_confidence, normalize_error_type
from src.utils import load_prompt, normalize_label


DIAGNOSTIC_REPAIR_PROMPT = """Sentence: "{sentence}"
Target: "{target}"

Direct label: "{direct_label}"
THOR aspect: "{aspect}"
THOR opinion clue: "{opinion}"
THOR polarity reasoning: "{polarity_reasoning}"
THOR label: "{thor_label}"

The previous diagnostic output was incomplete or not parseable:
{raw_output}

Return exactly one JSON object and no other text:
{{"error_type":"<allowed error type>","label":"<positive|negative|neutral>","confidence":"<low|medium|high>"}}

Allowed error_type values:
no_error, missed_implicit_negative, missed_implicit_positive, neutral_overinterpretation, target_scope_shift, aspect_opinion_mismatch, reasoning_label_inconsistency, insufficient_evidence
"""


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


def _extract_json_object(raw_output: str) -> dict:
    text = str(raw_output or "").strip()
    if not text:
        return {}

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not json_match:
        return {}

    try:
        value = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {}

    return value if isinstance(value, dict) else {}


def _regex_field(raw_output: str, field_names: list[str]) -> str:
    names = "|".join(re.escape(name) for name in field_names)
    match = re.search(
        rf"(?:{names})\s*[:=]\s*([A-Za-z_ -]+)",
        str(raw_output or ""),
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def parse_diagnostic_output(raw_output: str) -> dict:
    parsed = {}
    parsed.update(_extract_json_object(raw_output))

    for line in raw_output.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        parsed[key.strip().lower()] = value.strip()

    error_type = parsed.get("error_type", "")
    label = (
        parsed.get("label", "")
        or parsed.get("diagnostic_label", "")
        or parsed.get("corrected_label", "")
        or _regex_field(raw_output, ["label", "diagnostic_label", "corrected_label", "sentiment polarity", "polarity"])
    )
    confidence = parsed.get("confidence", "") or _regex_field(raw_output, ["confidence"])

    return {
        "error_type": normalize_error_type(error_type),
        "diagnostic_label": normalize_label(label),
        "confidence": normalize_confidence(confidence),
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
            max_tokens=int(os.getenv("ETC_DIAGNOSTIC_MAX_TOKENS", "512")),
        ).strip()

        parsed = parse_diagnostic_output(raw_output)
        if parsed["diagnostic_label"] == "unknown":
            repaired = self.repair_diagnostic_output(
                sentence=sentence,
                target=target,
                direct_label=direct_label,
                aspect=aspect,
                opinion=opinion,
                polarity_reasoning=polarity_reasoning,
                thor_label=thor_label,
                raw_output=raw_output,
            )
            if repaired:
                repair_raw_output, repair_parsed = repaired
                parsed = repair_parsed
                raw_output = f"{raw_output}\n\n[repair]\n{repair_raw_output}".strip()

        parsed["diagnostic_raw_output"] = raw_output
        return parsed

    def repair_diagnostic_output(
        self,
        sentence: str,
        target: str,
        direct_label: str,
        aspect: str,
        opinion: str,
        polarity_reasoning: str,
        thor_label: str,
        raw_output: str,
    ) -> tuple[str, dict] | None:
        retries = int(os.getenv("ETC_DIAGNOSTIC_REPAIR_RETRIES", "1"))
        if retries <= 0:
            return None

        prompt = DIAGNOSTIC_REPAIR_PROMPT.format(
            sentence=safe_prompt_value(sentence),
            target=safe_prompt_value(target),
            direct_label=safe_prompt_value(direct_label),
            aspect=safe_prompt_value(aspect),
            opinion=safe_prompt_value(opinion),
            polarity_reasoning=safe_prompt_value(polarity_reasoning),
            thor_label=safe_prompt_value(thor_label),
            raw_output=safe_prompt_value(raw_output, fallback="empty"),
        )

        for _ in range(retries):
            repair_raw_output = self.runner.run(
                prompt,
                temperature=0.0,
                max_tokens=int(os.getenv("ETC_DIAGNOSTIC_REPAIR_MAX_TOKENS", "512")),
            ).strip()
            repair_parsed = parse_diagnostic_output(repair_raw_output)
            if repair_parsed["diagnostic_label"] != "unknown":
                return repair_raw_output, repair_parsed

        return None
