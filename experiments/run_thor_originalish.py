from pathlib import Path
import os
import sys

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import VALID_LABELS, evaluate_predictions
from src.experiment_config import describe_runtime, parse_debug_n, result_path
from src.thor_pipeline import THORPipeline


DATA_PATH = "data/processed/semeval14_scapt_isa_only_clean.csv"
OUTPUT_PATH = result_path("thor_originalish_isa", "predictions.csv", "THOR_ORIGINALISH_OUTPUT_PATH")
METRICS_PATH = result_path("thor_originalish_isa", "metrics.txt", "THOR_ORIGINALISH_METRICS_PATH")

DEBUG_N = parse_debug_n(default=20)
RESUME = os.getenv("RESUME_THOR_ORIGINALISH", "0") == "1"
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))

ASPECT_MAX_TOKENS = int(os.getenv("THOR_ASPECT_MAX_TOKENS", "16"))
OPINION_MAX_TOKENS = int(os.getenv("THOR_OPINION_MAX_TOKENS", "80"))
POLARITY_REASONING_MAX_TOKENS = int(os.getenv("THOR_POLARITY_REASONING_MAX_TOKENS", "80"))
POLARITY_LABEL_MAX_TOKENS = int(os.getenv("THOR_POLARITY_LABEL_MAX_TOKENS", "8"))
ASPECT_MAX_WORDS = int(os.getenv("THOR_ASPECT_MAX_WORDS", "8"))
TEMPERATURE = float(os.getenv("THOR_TEMPERATURE", "0.0"))

OUTPUT_COLUMNS = ["aspect", "opinion", "polarity_reasoning", "raw_polarity_output", "prediction"]


def save_outputs(df: pd.DataFrame):
    df.to_csv(OUTPUT_PATH, index=False)

    metrics = evaluate_predictions(df, gold_col="polarity", pred_col="prediction")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Runtime: {describe_runtime()}\n")
        f.write("Prompt variant: thor_originalish\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
        f.write(f"aspect_max_tokens: {ASPECT_MAX_TOKENS}\n")
        f.write(f"opinion_max_tokens: {OPINION_MAX_TOKENS}\n")
        f.write(f"polarity_reasoning_max_tokens: {POLARITY_REASONING_MAX_TOKENS}\n")
        f.write(f"polarity_label_max_tokens: {POLARITY_LABEL_MAX_TOKENS}\n")
        f.write(f"temperature: {TEMPERATURE:.2f}\n")
        f.write(f"n_total: {metrics['n_total']}\n")
        f.write(f"n_eval: {metrics['n_eval']}\n")
        f.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {metrics['macro_f1']:.6f}\n\n")
        f.write(metrics["report"])

    return metrics


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if DEBUG_N is not None:
        df = df.head(DEBUG_N).copy()

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    start_idx = 0
    if RESUME and Path(OUTPUT_PATH).exists():
        prev_df = pd.read_csv(OUTPUT_PATH)
        if len(prev_df) == len(df):
            for col in OUTPUT_COLUMNS:
                if col in prev_df.columns:
                    df[col] = prev_df[col]

            completed_mask = df["prediction"].isin(VALID_LABELS)
            start_idx = int(completed_mask.sum())

    pipeline = THORPipeline(
        aspect_prompt_path="prompts/thor_originalish_aspect.txt",
        opinion_prompt_path="prompts/thor_originalish_opinion.txt",
        polarity_prompt_path="prompts/thor_originalish_polarity.txt",
        polarity_label_prompt_path="prompts/thor_originalish_polarity_label.txt",
        aspect_max_tokens=ASPECT_MAX_TOKENS,
        opinion_max_tokens=OPINION_MAX_TOKENS,
        polarity_reasoning_max_tokens=POLARITY_REASONING_MAX_TOKENS,
        polarity_label_max_tokens=POLARITY_LABEL_MAX_TOKENS,
        aspect_max_words=ASPECT_MAX_WORDS,
        temperature=TEMPERATURE,
    )

    for idx, row in tqdm(
        df.iloc[start_idx:].iterrows(),
        total=len(df) - start_idx,
        desc="Running original-ish THOR baseline",
    ):
        result = pipeline.run_one(
            sentence=row["sentence"],
            target=row["target"],
        )

        for col in OUTPUT_COLUMNS:
            df.at[idx, col] = result[col]

        completed = idx + 1
        if completed % SAVE_EVERY == 0:
            save_outputs(df)

    metrics = save_outputs(df)

    print("\nDone.")
    print(f"Saved predictions to: {OUTPUT_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
