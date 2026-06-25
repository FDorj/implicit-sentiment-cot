from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluator import VALID_LABELS, evaluate_predictions


KEY_COLS = ["id", "source_sentence_id", "sentence", "target", "from", "to", "polarity"]
SPLITS = ["overall", "train", "test"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path: str | Path
    pred_col: str
    note: str = ""


def require_columns(df: pd.DataFrame, columns: list[str], source: str | Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_prediction_frame(path: str | Path, required_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, required_columns, path)
    return df


def compute_method_metrics(methods: list[MethodSpec], splits: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_splits = splits or SPLITS

    for method in methods:
        df = load_prediction_frame(method.path, ["polarity", "split", method.pred_col])
        for split in selected_splits:
            split_df = df if split == "overall" else df[df["split"].astype(str).str.lower() == split].copy()
            if split_df.empty:
                continue

            metrics = evaluate_predictions(split_df, gold_col="polarity", pred_col=method.pred_col)
            rows.append(
                {
                    "method": method.name,
                    "split": split,
                    "n_total": metrics["n_total"],
                    "n_eval": metrics["n_eval"],
                    "unknown_labels": metrics["n_total"] - metrics["n_eval"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "prediction_column": method.pred_col,
                    "source_file": str(method.path),
                    "note": method.note,
                }
            )

    return pd.DataFrame(rows)


def count_duplicate_keys(df: pd.DataFrame) -> int:
    return int(df.duplicated(KEY_COLS).sum())


def count_alignment_mismatches(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> int:
    merged = left_df[[*KEY_COLS, left_col]].merge(
        right_df[[*KEY_COLS, right_col]],
        on=KEY_COLS,
        how="inner",
    )
    return int((merged[left_col] != merged[right_col]).sum())


def validate_final_chain(
    direct_path: str | Path,
    thor_path: str | Path,
    etc_path: str | Path,
    selected_path: str | Path,
) -> dict[str, Any]:
    direct_df = load_prediction_frame(direct_path, [*KEY_COLS, "prediction"])
    thor_df = load_prediction_frame(thor_path, [*KEY_COLS, "prediction"])
    etc_required = [
        *KEY_COLS,
        "direct_prediction",
        "thor_prediction",
        "diagnostic_label",
        "diagnostic_confidence",
        "error_type",
        "controller_prediction",
    ]
    etc_df = load_prediction_frame(etc_path, etc_required)
    selected_df = load_prediction_frame(selected_path, [*KEY_COLS, "selected_prediction"])

    row_counts = {
        "direct": len(direct_df),
        "thor": len(thor_df),
        "etc": len(etc_df),
        "selected": len(selected_df),
    }
    duplicate_key_counts = {
        "direct": count_duplicate_keys(direct_df),
        "thor": count_duplicate_keys(thor_df),
        "etc": count_duplicate_keys(etc_df),
        "selected": count_duplicate_keys(selected_df),
    }

    direct_etc_rows = len(
        direct_df[KEY_COLS].merge(etc_df[KEY_COLS], on=KEY_COLS, how="inner")
    )
    thor_etc_rows = len(
        thor_df[KEY_COLS].merge(etc_df[KEY_COLS], on=KEY_COLS, how="inner")
    )
    etc_selected_rows = len(
        etc_df[KEY_COLS].merge(selected_df[KEY_COLS], on=KEY_COLS, how="inner")
    )

    diagnostic_columns = ["diagnostic_label", "diagnostic_confidence", "error_type"]

    return {
        "row_count": row_counts["selected"],
        "row_counts": row_counts,
        "all_row_counts_equal": len(set(row_counts.values())) == 1,
        "duplicate_key_counts": duplicate_key_counts,
        "has_duplicate_keys": any(count > 0 for count in duplicate_key_counts.values()),
        "direct_etc_aligned_rows": direct_etc_rows,
        "thor_etc_aligned_rows": thor_etc_rows,
        "etc_selected_aligned_rows": etc_selected_rows,
        "direct_alignment_mismatches": count_alignment_mismatches(
            direct_df, etc_df, "prediction", "direct_prediction"
        ),
        "thor_alignment_mismatches": count_alignment_mismatches(
            thor_df, etc_df, "prediction", "thor_prediction"
        ),
        "diagnostic_columns_present": all(column in etc_df.columns for column in diagnostic_columns),
        "selected_valid_labels": int(selected_df["selected_prediction"].isin(VALID_LABELS).sum()),
    }


def metrics_to_markdown(metrics_df: pd.DataFrame) -> str:
    display_columns = ["method", "split", "n_eval", "accuracy", "macro_f1", "unknown_labels"]
    lines = [
        "| method | split | n_eval | accuracy | macro_f1 | unknown_labels |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for _, row in metrics_df[display_columns].iterrows():
        lines.append(
            "| {method} | {split} | {n_eval} | {accuracy:.6f} | {macro_f1:.6f} | {unknown_labels} |".format(
                method=row["method"],
                split=row["split"],
                n_eval=int(row["n_eval"]),
                accuracy=float(row["accuracy"]),
                macro_f1=float(row["macro_f1"]),
                unknown_labels=int(row["unknown_labels"]),
            )
        )

    return "\n".join(lines) + "\n"


def validation_to_text(summary: dict[str, Any]) -> str:
    lines = [
        "Final pipeline validation",
        f"row_count: {summary['row_count']}",
        f"all_row_counts_equal: {summary['all_row_counts_equal']}",
        f"row_counts: {summary['row_counts']}",
        f"duplicate_key_counts: {summary['duplicate_key_counts']}",
        f"has_duplicate_keys: {summary['has_duplicate_keys']}",
        f"direct_etc_aligned_rows: {summary['direct_etc_aligned_rows']}",
        f"thor_etc_aligned_rows: {summary['thor_etc_aligned_rows']}",
        f"etc_selected_aligned_rows: {summary['etc_selected_aligned_rows']}",
        f"direct_alignment_mismatches: {summary['direct_alignment_mismatches']}",
        f"thor_alignment_mismatches: {summary['thor_alignment_mismatches']}",
        f"diagnostic_columns_present: {summary['diagnostic_columns_present']}",
        f"selected_valid_labels: {summary['selected_valid_labels']}",
    ]
    return "\n".join(lines) + "\n"
