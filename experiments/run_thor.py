from pathlib import Path
import os
import sys
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thor_pipeline import THORPipeline
from src.evaluator import evaluate_predictions


DATA_PATH = "data/processed/semeval14_scapt_isa_only_clean.csv"
OUTPUT_PATH = "results/thor_isa_predictions.csv"
METRICS_PATH = "results/thor_isa_metrics.txt"

debug_n_raw = os.getenv("DEBUG_N")
if debug_n_raw is None:
    DEBUG_N = 20
elif debug_n_raw.strip().lower() in {"", "all", "none", "full"}:
    DEBUG_N = None
else:
    DEBUG_N = int(debug_n_raw)

RESUME = os.getenv("RESUME_THOR", "0") == "1"
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))


def save_outputs(df: pd.DataFrame):
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

    return metrics


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if DEBUG_N is not None:
        df = df.head(DEBUG_N).copy()

    for col in ["aspect", "opinion", "polarity_reasoning", "raw_polarity_output", "prediction"]:
        if col not in df.columns:
            df[col] = pd.NA

    start_idx = 0
    if RESUME and Path(OUTPUT_PATH).exists():
        prev_df = pd.read_csv(OUTPUT_PATH)
        if len(prev_df) == len(df):
            for col in ["aspect", "opinion", "polarity_reasoning", "raw_polarity_output", "prediction"]:
                if col in prev_df.columns:
                    df[col] = prev_df[col]

            completed_mask = df["prediction"].isin(["positive", "negative", "neutral"])
            start_idx = int(completed_mask.sum())

    pipeline = THORPipeline()

    for idx, row in tqdm(
        df.iloc[start_idx:].iterrows(),
        total=len(df) - start_idx,
        desc="Running THOR baseline",
    ):
        result = pipeline.run_one(
            sentence=row["sentence"],
            target=row["target"],
        )

        df.at[idx, "aspect"] = result["aspect"]
        df.at[idx, "opinion"] = result["opinion"]
        df.at[idx, "polarity_reasoning"] = result["polarity_reasoning"]
        df.at[idx, "raw_polarity_output"] = result["raw_polarity_output"]
        df.at[idx, "prediction"] = result["prediction"]

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
