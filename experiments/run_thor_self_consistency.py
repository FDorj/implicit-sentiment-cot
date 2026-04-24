from collections import Counter
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
VARIANT = os.getenv("THOR_SC_VARIANT", "originalish").strip().lower()
SC_N = int(os.getenv("THOR_SC_N", "3"))
TEMPERATURE = float(os.getenv("THOR_SC_TEMPERATURE", "0.7"))
OUTPUT_STEM = f"thor_{VARIANT}_sc{SC_N}_isa"
OUTPUT_PATH = result_path(OUTPUT_STEM, "predictions.csv", "THOR_SC_OUTPUT_PATH")
METRICS_PATH = result_path(OUTPUT_STEM, "metrics.txt", "THOR_SC_METRICS_PATH")

DEBUG_N = parse_debug_n(default=5)
RESUME = os.getenv("RESUME_THOR_SC", "0") == "1"
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "5"))

OUTPUT_COLUMNS = [
    "sc_raw_runs",
    "sc_labels",
    "sc_vote_counts",
    "aspect",
    "opinion",
    "polarity_reasoning",
    "raw_polarity_output",
    "prediction",
]


def build_pipeline() -> THORPipeline:
    if VARIANT == "originalish":
        return THORPipeline(
            aspect_prompt_path="prompts/thor_originalish_aspect.txt",
            opinion_prompt_path="prompts/thor_originalish_opinion.txt",
            polarity_prompt_path="prompts/thor_originalish_polarity.txt",
            polarity_label_prompt_path="prompts/thor_originalish_polarity_label.txt",
            aspect_max_tokens=16,
            opinion_max_tokens=80,
            polarity_reasoning_max_tokens=80,
            polarity_label_max_tokens=8,
            aspect_max_words=8,
            temperature=TEMPERATURE,
        )

    if VARIANT == "simplified":
        return THORPipeline(temperature=TEMPERATURE)

    raise ValueError(f"Unsupported THOR_SC_VARIANT: {VARIANT}")


def choose_prediction(labels: list[str]) -> tuple[str, str]:
    valid_labels = [label for label in labels if label in VALID_LABELS]
    if not valid_labels:
        return "unknown", ""

    counts = Counter(valid_labels)
    winner, winner_count = counts.most_common(1)[0]
    tied_labels = [label for label, count in counts.items() if count == winner_count]
    if len(tied_labels) == 1:
        return winner, ";".join(f"{label}:{count}" for label, count in sorted(counts.items()))

    for label in labels:
        if label in tied_labels:
            return label, ";".join(f"{label}:{count}" for label, count in sorted(counts.items()))

    return winner, ";".join(f"{label}:{count}" for label, count in sorted(counts.items()))


def save_outputs(df: pd.DataFrame):
    df.to_csv(OUTPUT_PATH, index=False)
    metrics = evaluate_predictions(df, gold_col="polarity", pred_col="prediction")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Runtime: {describe_runtime()}\n")
        f.write(f"Variant: {VARIANT}\n")
        f.write(f"Self-consistency samples: {SC_N}\n")
        f.write(f"Temperature: {TEMPERATURE:.2f}\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
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

    pipeline = build_pipeline()

    for idx, row in tqdm(
        df.iloc[start_idx:].iterrows(),
        total=len(df) - start_idx,
        desc=f"Running THOR {VARIANT} self-consistency",
    ):
        run_results = []
        labels = []
        for _ in range(SC_N):
            result = pipeline.run_one(
                sentence=row["sentence"],
                target=row["target"],
            )
            run_results.append(result)
            labels.append(result["prediction"])

        prediction, vote_counts = choose_prediction(labels)
        representative = next(
            (result for result in run_results if result["prediction"] == prediction),
            run_results[0],
        )

        df.at[idx, "sc_raw_runs"] = repr(run_results)
        df.at[idx, "sc_labels"] = ",".join(labels)
        df.at[idx, "sc_vote_counts"] = vote_counts
        df.at[idx, "aspect"] = representative["aspect"]
        df.at[idx, "opinion"] = representative["opinion"]
        df.at[idx, "polarity_reasoning"] = representative["polarity_reasoning"]
        df.at[idx, "raw_polarity_output"] = representative["raw_polarity_output"]
        df.at[idx, "prediction"] = prediction

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
