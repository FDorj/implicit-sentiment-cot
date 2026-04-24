from src.prompt_runner import PromptRunner
from src.utils import load_prompt, normalize_label


def clean_short_text(text: str, max_words: int = 6, fallback: str = "general") -> str:
    if not text:
        return fallback

    cleaned = text.strip().splitlines()[0].strip()
    cleaned = cleaned.replace('"', "").replace("'", "")
    cleaned = " ".join(cleaned.split())

    if not cleaned:
        return fallback

    words = cleaned.split()
    return " ".join(words[:max_words])


class THORPipeline:
    def __init__(
        self,
        aspect_prompt_path: str = "prompts/thor_aspect.txt",
        opinion_prompt_path: str = "prompts/thor_opinion.txt",
        polarity_prompt_path: str = "prompts/thor_polarity.txt",
        polarity_label_prompt_path: str = "prompts/thor_polarity_label.txt",
        model: str | None = None,
        aspect_max_tokens: int = 6,
        opinion_max_tokens: int = 24,
        polarity_reasoning_max_tokens: int = 48,
        polarity_label_max_tokens: int = 3,
        aspect_max_words: int = 4,
        temperature: float = 0.0,
    ):
        self.runner = PromptRunner(model=model)
        self.aspect_prompt_template = load_prompt(aspect_prompt_path)
        self.opinion_prompt_template = load_prompt(opinion_prompt_path)
        self.polarity_prompt_template = load_prompt(polarity_prompt_path)
        self.polarity_label_prompt_template = load_prompt(polarity_label_prompt_path)
        self.aspect_max_tokens = aspect_max_tokens
        self.opinion_max_tokens = opinion_max_tokens
        self.polarity_reasoning_max_tokens = polarity_reasoning_max_tokens
        self.polarity_label_max_tokens = polarity_label_max_tokens
        self.aspect_max_words = aspect_max_words
        self.temperature = temperature

    def infer_aspect(self, sentence: str, target: str) -> str:
        prompt = self.aspect_prompt_template.format(
            sentence=sentence,
            target=target,
        )
        output = self.runner.run(
            prompt,
            temperature=self.temperature,
            max_tokens=self.aspect_max_tokens,
        ).strip()

        if not output:
            return "general"

        normalized = clean_short_text(output, max_words=self.aspect_max_words, fallback="general").lower()

        if normalized in {"none", "n/a", "na", "unknown"}:
            return "general"

        return normalized

    def infer_opinion(self, sentence: str, target: str, aspect: str) -> str:
        prompt = self.opinion_prompt_template.format(
            sentence=sentence,
            target=target,
            aspect=aspect,
        )
        output = self.runner.run(
            prompt,
            temperature=self.temperature,
            max_tokens=self.opinion_max_tokens,
        ).strip()

        if not output:
            return "no clear opinion"

        return output

    def infer_polarity_reasoning(self, sentence: str, target: str, aspect: str, opinion: str) -> str:
        prompt = self.polarity_prompt_template.format(
            sentence=sentence,
            target=target,
            aspect=aspect,
            opinion=opinion,
        )
        output = self.runner.run(
            prompt,
            temperature=self.temperature,
            max_tokens=self.polarity_reasoning_max_tokens,
        ).strip()

        return output

    def infer_polarity_label(self, sentence: str, target: str, polarity_reasoning: str) -> tuple[str, str]:
        prompt = self.polarity_label_prompt_template.format(
            sentence=sentence,
            target=target,
            polarity_reasoning=polarity_reasoning,
        )
        raw_output = self.runner.run(
            prompt,
            temperature=0.0,
            max_tokens=self.polarity_label_max_tokens,
        ).strip()

        return raw_output, normalize_label(raw_output)

    def run_one(self, sentence: str, target: str) -> dict:
        aspect = self.infer_aspect(sentence, target)
        if not aspect:
            aspect = "general"

        opinion = self.infer_opinion(sentence, target, aspect)
        if not opinion:
            opinion = "no clear opinion"

        polarity_reasoning = self.infer_polarity_reasoning(
            sentence=sentence,
            target=target,
            aspect=aspect,
            opinion=opinion,
        )
        raw_polarity_label_output, prediction = self.infer_polarity_label(
            sentence=sentence,
            target=target,
            polarity_reasoning=polarity_reasoning,
        )

        return {
            "aspect": aspect,
            "opinion": opinion,
            "polarity_reasoning": polarity_reasoning,
            "raw_polarity_output": raw_polarity_label_output,
            "prediction": prediction,
        }
