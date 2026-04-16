from pathlib import Path
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import CORRECTABLE_ERROR_TYPES, normalize_error_type, select_final_label
from src.evaluator import evaluate_predictions
from src.experiment_config import result_path


ETC_PATH = result_path("etc_isa", "predictions.csv", "ETC_PREDICTIONS_PATH")
OUTPUT_PATH = result_path("etc_selected_isa", "predictions.csv", "ETC_SELECTED_OUTPUT_PATH")
METRICS_PATH = result_path("etc_selected_isa", "metrics.txt", "ETC_SELECTED_METRICS_PATH")

FALLBACK_POLICY = os.getenv("ETC_SELECTED_FALLBACK_POLICY", "direct").strip().lower()
TRUST_NO_ERROR_ON_DISAGREEMENT = os.getenv("ETC_SELECTED_TRUST_NO_ERROR_ON_DISAGREEMENT", "0") == "1"
MIN_CORRECTABLE_CONFIDENCE = os.getenv("ETC_SELECTED_MIN_CORRECTABLE_CONFIDENCE", "high").strip().lower()
ACCEPT_ERROR_TYPES_RAW = os.getenv("ETC_SELECTED_ACCEPT_ERROR_TYPES", "missed_implicit_positive").strip().lower()

REQUIRED_COLUMNS = [
    "polarity",
    "direct_prediction",
    "thor_prediction",
    "diagnostic_label",
    "error_type",
    "diagnostic_confidence",
]


def parse_accepted_error_types(raw_value: str) -> set[str]:
    if raw_value in {"", "all", "*"}:
        return set(CORRECTABLE_ERROR_TYPES)

    return {
        normalize_error_type(value)
        for value in raw_value.split(",")
        if value.strip()
    } & CORRECTABLE_ERROR_TYPES


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)

    accepted_error_types = parse_accepted_error_types(ACCEPT_ERROR_TYPES_RAW)
    df = pd.read_csv(ETC_PATH)
    require_columns(df, REQUIRED_COLUMNS, ETC_PATH)

    predictions = []
    decisions = []
    for _, row in df.iterrows():
        prediction, decision = select_final_label(
            direct_label=row["direct_prediction"],
            thor_label=row["thor_prediction"],
            proposed_label=row["diagnostic_label"],
            error_type=row["error_type"],
            confidence=row["diagnostic_confidence"],
            fallback_policy=FALLBACK_POLICY,
            trust_no_error_on_disagreement=TRUST_NO_ERROR_ON_DISAGREEMENT,
            min_correctable_confidence=MIN_CORRECTABLE_CONFIDENCE,
            accepted_error_types=accepted_error_types,
        )
        predictions.append(prediction)
        decisions.append(decision)

    df["selected_prediction"] = predictions
    df["selected_controller_decision"] = decisions
    df.to_csv(OUTPUT_PATH, index=False)

    metrics = evaluate_predictions(df, gold_col="polarity", pred_col="selected_prediction")
    split_metrics = []
    for split in ["train", "test"]:
        split_df = df[df["split"] == split].copy()
        split_result = evaluate_predictions(split_df, gold_col="polarity", pred_col="selected_prediction")
        split_metrics.append((split, split_result))

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"ETC predictions: {ETC_PATH}\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
        f.write(f"fallback_policy: {FALLBACK_POLICY}\n")
        f.write(f"trust_no_error_on_disagreement: {TRUST_NO_ERROR_ON_DISAGREEMENT}\n")
        f.write(f"min_correctable_confidence: {MIN_CORRECTABLE_CONFIDENCE}\n")
        f.write(f"accepted_error_types: {','.join(sorted(accepted_error_types))}\n")
        f.write(f"n_total: {metrics['n_total']}\n")
        f.write(f"n_eval: {metrics['n_eval']}\n")
        f.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {metrics['macro_f1']:.6f}\n\n")
        f.write(metrics["report"])
        f.write("\n")

        for split, split_result in split_metrics:
            f.write(f"\n{split} metrics\n")
            f.write(f"n_total: {split_result['n_total']}\n")
            f.write(f"n_eval: {split_result['n_eval']}\n")
            f.write(f"accuracy: {split_result['accuracy']:.6f}\n")
            f.write(f"macro_f1: {split_result['macro_f1']:.6f}\n")

    print("Done.")
    print(f"Saved selected ETC predictions to: {OUTPUT_PATH}")
    print(f"Saved selected ETC metrics to: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
