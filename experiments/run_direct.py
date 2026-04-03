from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.prompt_runner import PromptRunner
from src.utils import load_prompt, normalize_label
from src.evaluator import evaluate_predictions


DATA_PATH = "data/processed/semeval14_scapt_isa_only_clean.csv"
PROMPT_PATH = "prompts/direct_prompt.txt"
OUTPUT_PATH = "results/direct_isa_predictions.csv"
METRICS_PATH = "results/direct_isa_metrics.txt"

# برای تست اولیه:
DEBUG_N = None   # بعداً برای اجرای کامل، این را None کن


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if DEBUG_N is not None:
        df = df.head(DEBUG_N).copy()

    prompt_template = load_prompt(PROMPT_PATH)
    runner = PromptRunner()

    predictions = []
    raw_outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running direct baseline"):
        prompt = prompt_template.format(
            sentence=row["sentence"],
            target=row["target"],
        )

        raw_output = runner.run(prompt, temperature=0.0, max_tokens=16)
        pred = normalize_label(raw_output)

        raw_outputs.append(raw_output)
        predictions.append(pred)

    df["raw_output"] = raw_outputs
    df["prediction"] = predictions
    df.to_csv(OUTPUT_PATH, index=False)

    metrics = evaluate_predictions(df, gold_col="polarity", pred_col="prediction")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
        f.write(f"n_total: {metrics['n_total']}\n")
        f.write(f"n_eval: {metrics['n_eval']}\n")
        f.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {metrics['macro_f1']:.6f}\n\n")
        f.write(metrics["report"])

    print("\nDone.")
    print(f"Saved predictions to: {OUTPUT_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()