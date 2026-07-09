from pathlib import Path
import os
import sys
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_runner import PromptRunner
from src.utils import load_prompt, normalize_label
from src.evaluator import VALID_LABELS, evaluate_predictions
from src.experiment_config import data_path, describe_runtime, parse_debug_n, result_path


DATA_PATH = data_path()
PROMPT_PATH = "prompts/direct_prompt.txt"
OUTPUT_PATH = result_path("direct_isa", "predictions.csv", "DIRECT_OUTPUT_PATH")
METRICS_PATH = result_path("direct_isa", "metrics.txt", "DIRECT_METRICS_PATH")

DEBUG_N = parse_debug_n(default=None)
DATA_SPLIT = os.getenv("DATA_SPLIT", "").strip().lower()
RESUME = os.getenv("RESUME_DIRECT", "0") == "1"
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))
KEY_COLS = ["id", "source_sentence_id", "sentence", "target", "from", "to", "polarity"]
OUTPUT_COLUMNS = ["raw_output", "prediction"]


def select_experiment_rows(
    df: pd.DataFrame,
    data_split: str = "",
    debug_n: int | None = None,
) -> pd.DataFrame:
    selected_df = df
    if data_split:
        selected_df = selected_df[selected_df["split"].astype(str).str.lower() == data_split]
        if selected_df.empty:
            raise ValueError(f"No rows found for DATA_SPLIT={data_split!r}.")

    if debug_n is not None:
        selected_df = selected_df.head(debug_n)

    return selected_df.copy()


def attach_previous_outputs(df: pd.DataFrame, previous_df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = pd.NA

    previous_cols = [*KEY_COLS, *OUTPUT_COLUMNS]
    missing = [col for col in previous_cols if col not in previous_df.columns]
    if missing:
        raise ValueError(f"Previous direct output is missing required columns: {missing}")

    merged = output_df.drop(columns=OUTPUT_COLUMNS, errors="ignore").merge(
        previous_df[previous_cols],
        on=KEY_COLS,
        how="left",
    )
    return merged


def completed_prediction_mask(df: pd.DataFrame) -> pd.Series:
    return df["prediction"].isin(VALID_LABELS)


def save_outputs(df: pd.DataFrame) -> dict:
    df.to_csv(OUTPUT_PATH, index=False)
    metrics = evaluate_predictions(df, gold_col="polarity", pred_col="prediction")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Runtime: {describe_runtime()}\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
        f.write(f"data_split: {DATA_SPLIT or 'all'}\n")
        f.write(f"debug_n: {DEBUG_N if DEBUG_N is not None else 'all'}\n")
        f.write(f"resume: {RESUME}\n")
        f.write(f"save_every: {SAVE_EVERY}\n")
        f.write(f"n_total: {metrics['n_total']}\n")
        f.write(f"n_eval: {metrics['n_eval']}\n")
        f.write(f"n_invalid: {metrics['n_invalid']}\n")
        f.write(f"valid_prediction_rate: {metrics['valid_prediction_rate']:.6f}\n")
        f.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {metrics['macro_f1']:.6f}\n\n")
        f.write(metrics["report"])

    return metrics


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = select_experiment_rows(df, data_split=DATA_SPLIT, debug_n=DEBUG_N)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    if RESUME and Path(OUTPUT_PATH).exists():
        previous_df = pd.read_csv(OUTPUT_PATH)
        df = attach_previous_outputs(df, previous_df)

    prompt_template = load_prompt(PROMPT_PATH)
    runner = PromptRunner()

    completed_mask = completed_prediction_mask(df)
    completed_before_run = int(completed_mask.sum())
    pending_indices = df[~completed_mask].index.tolist()

    for completed_now, idx in enumerate(
        tqdm(pending_indices, total=len(pending_indices), desc="Running direct baseline"),
        start=1,
    ):
        row = df.loc[idx]
        prompt = prompt_template.format(
            sentence=row["sentence"],
            target=row["target"],
        )

        raw_output = runner.run(prompt, temperature=0.0, max_tokens=16)
        pred = normalize_label(raw_output)

        df.at[idx, "raw_output"] = raw_output
        df.at[idx, "prediction"] = pred

        completed_total = completed_before_run + completed_now
        if completed_total % SAVE_EVERY == 0:
            save_outputs(df)

    metrics = save_outputs(df)

    print("\nDone.")
    print(f"Saved predictions to: {OUTPUT_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
