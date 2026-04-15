from src.prompt_runner import PromptRunner
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
