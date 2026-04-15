from pathlib import Path
import os
import sys

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import VALID_LABELS, evaluate_predictions
from src.reflection_pipeline import SimpleReflectionPipeline


SOURCE_PATH = os.getenv("THOR_PREDICTIONS_PATH", "results/thor_isa_predictions.csv")
OUTPUT_PATH = os.getenv("SIMPLE_REFLECTION_OUTPUT_PATH", "results/simple_reflection_isa_predictions.csv")
METRICS_PATH = os.getenv("SIMPLE_REFLECTION_METRICS_PATH", "results/simple_reflection_isa_metrics.txt")

debug_n_raw = os.getenv("DEBUG_N")
if debug_n_raw is None:
    DEBUG_N = 20
elif debug_n_raw.strip().lower() in {"", "all", "none", "full"}:
    DEBUG_N = None
else:
    DEBUG_N = int(debug_n_raw)

RESUME = os.getenv("RESUME_SIMPLE_REFLECTION", "0") == "1"
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))

REQUIRED_COLUMNS = [
    "sentence",
    "target",
    "polarity",
    "aspect",
    "opinion",
    "polarity_reasoning",
    "prediction",
]
OUTPUT_COLUMNS = ["reflection_raw_output", "reflection_prediction"]


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def save_outputs(df: pd.DataFrame):
    df.to_csv(OUTPUT_PATH, index=False)

    metrics = evaluate_predictions(df, gold_col="polarity", pred_col="reflection_prediction")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Source: {SOURCE_PATH}\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
        f.write(f"n_total: {metrics['n_total']}\n")
        f.write(f"n_eval: {metrics['n_eval']}\n")
        f.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {metrics['macro_f1']:.6f}\n\n")
        f.write(metrics["report"])

    return metrics


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_PATH)
    require_columns(df, REQUIRED_COLUMNS, SOURCE_PATH)

    if DEBUG_N is not None:
        df = df.head(DEBUG_N).copy()

    df = df.rename(columns={"prediction": "thor_prediction"})

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

            completed_mask = df["reflection_prediction"].isin(VALID_LABELS)
            start_idx = int(completed_mask.sum())

    pipeline = SimpleReflectionPipeline()

    for idx, row in tqdm(
        df.iloc[start_idx:].iterrows(),
        total=len(df) - start_idx,
        desc="Running simple reflection",
    ):
        result = pipeline.run_one(
            sentence=row["sentence"],
            target=row["target"],
            aspect=row["aspect"],
            opinion=row["opinion"],
            polarity_reasoning=row["polarity_reasoning"],
            initial_label=row["thor_prediction"],
        )

        df.at[idx, "reflection_raw_output"] = result["reflection_raw_output"]
        df.at[idx, "reflection_prediction"] = result["reflection_prediction"]

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
