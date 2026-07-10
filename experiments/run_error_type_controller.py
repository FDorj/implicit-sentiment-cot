from pathlib import Path
import os
import sys

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import CORRECTABLE_ERROR_TYPES, normalize_error_type, select_final_label
from src.evaluator import VALID_LABELS, evaluate_predictions
from src.experiment_config import describe_runtime, parse_debug_n, result_path
from src.reflection_pipeline import ErrorTypeReflectionPipeline


DIRECT_PATH = result_path("direct_isa", "predictions.csv", "DIRECT_PREDICTIONS_PATH")
THOR_PATH = result_path("thor_isa", "predictions.csv", "THOR_PREDICTIONS_PATH")
OUTPUT_PATH = result_path("etc_isa", "predictions.csv", "ETC_OUTPUT_PATH")
METRICS_PATH = result_path("etc_isa", "metrics.txt", "ETC_METRICS_PATH")

DEBUG_N = parse_debug_n(default=20)
RESUME = os.getenv("RESUME_ERROR_TYPE", "0") == "1"
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))
RERUN_INVALID_DIAGNOSTICS = os.getenv("RERUN_INVALID_DIAGNOSTICS", "0") == "1"
TRIGGER_POLICY = os.getenv("ETC_TRIGGER_POLICY", "disagreement").strip().lower()
FALLBACK_POLICY = os.getenv("ETC_FALLBACK_POLICY", "direct").strip().lower()
TRUST_NO_ERROR_ON_DISAGREEMENT = os.getenv("ETC_TRUST_NO_ERROR_ON_DISAGREEMENT", "0") == "1"
MIN_CORRECTABLE_CONFIDENCE = os.getenv("ETC_MIN_CORRECTABLE_CONFIDENCE", "medium").strip().lower()
ACCEPT_ERROR_TYPES_RAW = os.getenv("ETC_ACCEPT_ERROR_TYPES", "all").strip().lower()

KEY_COLS = ["id", "source_sentence_id", "sentence", "target", "from", "to", "polarity"]
OUTPUT_COLUMNS = [
    "diagnostic_raw_output",
    "error_type",
    "diagnostic_label",
    "diagnostic_confidence",
    "controller_prediction",
    "controller_decision",
    "diagnostic_triggered",
]


def parse_accepted_error_types(raw_value: str) -> set[str]:
    if raw_value in {"", "all", "*"}:
        return set(CORRECTABLE_ERROR_TYPES)

    return {
        normalize_error_type(value)
        for value in raw_value.split(",")
        if value.strip()
    } & CORRECTABLE_ERROR_TYPES


ACCEPTED_ERROR_TYPES = parse_accepted_error_types(ACCEPT_ERROR_TYPES_RAW)


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_inputs() -> pd.DataFrame:
    direct_df = pd.read_csv(DIRECT_PATH)
    thor_df = pd.read_csv(THOR_PATH)

    require_columns(direct_df, KEY_COLS + ["domain", "split", "raw_output", "prediction"], DIRECT_PATH)
    require_columns(
        thor_df,
        KEY_COLS
        + [
            "aspect",
            "opinion",
            "polarity_reasoning",
            "raw_polarity_output",
            "prediction",
        ],
        THOR_PATH,
    )

    for source, df in [(DIRECT_PATH, direct_df), (THOR_PATH, thor_df)]:
        if df.duplicated(KEY_COLS).any():
            raise ValueError(f"{KEY_COLS} must be unique in {source}.")

    direct_cols = [*KEY_COLS, "domain", "split", "raw_output", "prediction"]
    thor_cols = [
        *KEY_COLS,
        "aspect",
        "opinion",
        "polarity_reasoning",
        "raw_polarity_output",
        "prediction",
    ]

    df = direct_df[direct_cols].merge(
        thor_df[thor_cols],
        on=KEY_COLS,
        how="inner",
        suffixes=("_direct", "_thor"),
    )

    if len(df) != len(direct_df) or len(df) != len(thor_df):
        raise ValueError(
            "Direct and THOR prediction files do not align one-to-one. "
            f"direct={len(direct_df)}, thor={len(thor_df)}, merged={len(df)}"
        )

    return df.rename(
        columns={
            "raw_output": "direct_raw_output",
            "prediction_direct": "direct_prediction",
            "raw_polarity_output": "thor_raw_polarity_output",
            "prediction_thor": "thor_prediction",
        }
    )


def should_trigger(row: pd.Series) -> bool:
    if TRIGGER_POLICY == "all":
        return True
    if TRIGGER_POLICY == "disagreement":
        return row["direct_prediction"] != row["thor_prediction"]
    raise ValueError(f"Unsupported ETC_TRIGGER_POLICY: {TRIGGER_POLICY}")


def apply_no_diagnostic(row: pd.Series) -> dict:
    final_label, decision = select_final_label(
        direct_label=row["direct_prediction"],
        thor_label=row["thor_prediction"],
        proposed_label=row["thor_prediction"],
        error_type="no_error",
        confidence="high",
        fallback_policy=FALLBACK_POLICY,
    )
    return {
        "diagnostic_raw_output": "",
        "error_type": "no_error",
        "diagnostic_label": row["thor_prediction"],
        "diagnostic_confidence": "high",
        "controller_prediction": final_label,
        "controller_decision": decision,
        "diagnostic_triggered": False,
    }


def completed_error_type_mask(df: pd.DataFrame, rerun_invalid_diagnostics: bool = False) -> pd.Series:
    completed_mask = df["controller_prediction"].isin(VALID_LABELS)
    if not rerun_invalid_diagnostics:
        return completed_mask

    triggered = df["diagnostic_triggered"].fillna(False).astype(bool)
    valid_diagnostic = df["diagnostic_label"].isin(VALID_LABELS)
    return completed_mask & (~triggered | valid_diagnostic)


def pending_error_type_indices(df: pd.DataFrame, rerun_invalid_diagnostics: bool = False) -> list:
    completed_mask = completed_error_type_mask(
        df,
        rerun_invalid_diagnostics=rerun_invalid_diagnostics,
    )
    return df.index[~completed_mask].tolist()


def save_outputs(df: pd.DataFrame):
    df.to_csv(OUTPUT_PATH, index=False)
    metrics = evaluate_predictions(df, gold_col="polarity", pred_col="controller_prediction")

    triggered_count = int(df["diagnostic_triggered"].fillna(False).sum())
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Direct predictions: {DIRECT_PATH}\n")
        f.write(f"THOR predictions: {THOR_PATH}\n")
        f.write(f"Runtime: {describe_runtime()}\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
        f.write(f"trigger_policy: {TRIGGER_POLICY}\n")
        f.write(f"fallback_policy: {FALLBACK_POLICY}\n")
        f.write(f"trust_no_error_on_disagreement: {TRUST_NO_ERROR_ON_DISAGREEMENT}\n")
        f.write(f"min_correctable_confidence: {MIN_CORRECTABLE_CONFIDENCE}\n")
        f.write(f"accepted_error_types: {','.join(sorted(ACCEPTED_ERROR_TYPES))}\n")
        f.write(f"n_total: {metrics['n_total']}\n")
        f.write(f"n_eval: {metrics['n_eval']}\n")
        f.write(f"n_diagnostic_triggered: {triggered_count}\n")
        f.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {metrics['macro_f1']:.6f}\n\n")
        f.write(metrics["report"])

    return metrics


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    df = load_inputs()
    if DEBUG_N is not None:
        df = df.head(DEBUG_N).copy()

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    pending_indices = df.index.tolist()
    if RESUME and Path(OUTPUT_PATH).exists():
        prev_df = pd.read_csv(OUTPUT_PATH)
        if len(prev_df) == len(df):
            for col in OUTPUT_COLUMNS:
                if col in prev_df.columns:
                    df[col] = prev_df[col]

            pending_indices = pending_error_type_indices(
                df,
                rerun_invalid_diagnostics=RERUN_INVALID_DIAGNOSTICS,
            )

    pipeline = ErrorTypeReflectionPipeline()

    for completed, (idx, row) in enumerate(
        tqdm(
            df.loc[pending_indices].iterrows(),
            total=len(pending_indices),
            desc="Running ETC-ISA controller",
        ),
        start=1,
    ):
        if should_trigger(row):
            diagnosis = pipeline.diagnose(
                sentence=row["sentence"],
                target=row["target"],
                direct_label=row["direct_prediction"],
                aspect=row["aspect"],
                opinion=row["opinion"],
                polarity_reasoning=row["polarity_reasoning"],
                thor_label=row["thor_prediction"],
            )
            final_label, decision = select_final_label(
                direct_label=row["direct_prediction"],
                thor_label=row["thor_prediction"],
                proposed_label=diagnosis["diagnostic_label"],
                error_type=diagnosis["error_type"],
                confidence=diagnosis["confidence"],
                fallback_policy=FALLBACK_POLICY,
                trust_no_error_on_disagreement=TRUST_NO_ERROR_ON_DISAGREEMENT,
                min_correctable_confidence=MIN_CORRECTABLE_CONFIDENCE,
                accepted_error_types=ACCEPTED_ERROR_TYPES,
            )

            updates = {
                "diagnostic_raw_output": diagnosis["diagnostic_raw_output"],
                "error_type": diagnosis["error_type"],
                "diagnostic_label": diagnosis["diagnostic_label"],
                "diagnostic_confidence": diagnosis["confidence"],
                "controller_prediction": final_label,
                "controller_decision": decision,
                "diagnostic_triggered": True,
            }
        else:
            updates = apply_no_diagnostic(row)

        for col, value in updates.items():
            df.at[idx, col] = value

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
