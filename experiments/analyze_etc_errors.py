from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import VALID_LABELS, evaluate_predictions
from src.experiment_config import result_path


DIRECT_PATH = result_path("direct_isa", "predictions.csv", "DIRECT_PREDICTIONS_PATH")
THOR_PATH = result_path("thor_isa", "predictions.csv", "THOR_PREDICTIONS_PATH")
ETC_PATH = result_path("etc_isa", "predictions.csv", "ETC_PREDICTIONS_PATH")
OUTPUT_PATH = result_path("etc_error_analysis", "predictions.csv", "ETC_ERROR_ANALYSIS_OUTPUT_PATH")
EXAMPLES_PATH = result_path("etc_error_examples", "predictions.csv", "ETC_ERROR_EXAMPLES_OUTPUT_PATH")
METRICS_PATH = result_path("etc_error_analysis", "metrics.txt", "ETC_ERROR_ANALYSIS_METRICS_PATH")

KEY_COLS = ["id", "source_sentence_id", "sentence", "target", "from", "to", "polarity"]
EXAMPLES_PER_GROUP = 12


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def format_count_table(counts: pd.Series, total: int) -> str:
    lines = []
    for name, count in counts.items():
        pct = count / total * 100 if total else 0.0
        lines.append(f"{name}: {count} ({pct:.2f}%)")
    return "\n".join(lines)


def format_confusion(title: str, y_true: pd.Series, y_pred: pd.Series) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
    table = pd.DataFrame(
        matrix,
        index=[f"gold_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    )
    return f"{title}\n{table.to_string()}"


def etc_change_group(row: pd.Series) -> str:
    thor_correct = bool(row["thor_correct"])
    etc_correct = bool(row["etc_correct"])
    changed = bool(row["etc_changed_thor"])

    if not changed and etc_correct:
        return "unchanged_correct"
    if not changed and not etc_correct:
        return "unchanged_wrong"
    if not thor_correct and etc_correct:
        return "fixed_by_etc"
    if thor_correct and not etc_correct:
        return "broken_by_etc"
    return "changed_still_wrong"


def build_examples(analysis_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for group_name in [
        "fixed_by_etc",
        "broken_by_etc",
        "changed_still_wrong",
        "unchanged_wrong",
    ]:
        group_df = analysis_df[analysis_df["etc_change_group"] == group_name].copy()
        if group_df.empty:
            continue

        examples = (
            group_df.sort_values(["polarity", "id"])
            .groupby("polarity", group_keys=False)
            .head(max(1, EXAMPLES_PER_GROUP // len(VALID_LABELS)))
            .head(EXAMPLES_PER_GROUP)
        )
        frames.append(examples)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)

    direct_df = pd.read_csv(DIRECT_PATH)
    thor_df = pd.read_csv(THOR_PATH)
    etc_df = pd.read_csv(ETC_PATH)

    require_columns(direct_df, KEY_COLS + ["domain", "split", "prediction", "raw_output"], DIRECT_PATH)
    require_columns(thor_df, KEY_COLS + ["prediction"], THOR_PATH)
    require_columns(
        etc_df,
        KEY_COLS
        + [
            "direct_prediction",
            "thor_prediction",
            "controller_prediction",
            "controller_decision",
            "diagnostic_triggered",
            "error_type",
            "diagnostic_label",
            "diagnostic_confidence",
        ],
        ETC_PATH,
    )

    for source, df in [(DIRECT_PATH, direct_df), (THOR_PATH, thor_df), (ETC_PATH, etc_df)]:
        if df.duplicated(KEY_COLS).any():
            raise ValueError(f"{KEY_COLS} must be unique in {source}.")

    direct_cols = [*KEY_COLS, "domain", "split", "raw_output", "prediction"]
    thor_cols = [*KEY_COLS, "prediction"]
    etc_cols = [
        *KEY_COLS,
        "controller_prediction",
        "controller_decision",
        "diagnostic_triggered",
        "error_type",
        "diagnostic_label",
        "diagnostic_confidence",
    ]

    analysis_df = (
        direct_df[direct_cols]
        .merge(thor_df[thor_cols], on=KEY_COLS, how="inner", suffixes=("_direct", "_thor"))
        .merge(etc_df[etc_cols], on=KEY_COLS, how="inner")
    )

    if len(analysis_df) != len(direct_df) or len(analysis_df) != len(thor_df) or len(analysis_df) != len(etc_df):
        raise ValueError(
            "Prediction files do not align one-to-one. "
            f"direct={len(direct_df)}, thor={len(thor_df)}, etc={len(etc_df)}, merged={len(analysis_df)}"
        )

    analysis_df = analysis_df.rename(
        columns={
            "raw_output": "direct_raw_output",
            "prediction_direct": "direct_prediction",
            "prediction_thor": "thor_prediction",
            "controller_prediction": "etc_prediction",
        }
    )

    analysis_df["direct_correct"] = analysis_df["direct_prediction"] == analysis_df["polarity"]
    analysis_df["thor_correct"] = analysis_df["thor_prediction"] == analysis_df["polarity"]
    analysis_df["etc_correct"] = analysis_df["etc_prediction"] == analysis_df["polarity"]
    analysis_df["etc_changed_thor"] = analysis_df["etc_prediction"] != analysis_df["thor_prediction"]
    analysis_df["etc_changed_direct"] = analysis_df["etc_prediction"] != analysis_df["direct_prediction"]
    analysis_df["etc_change_group"] = analysis_df.apply(etc_change_group, axis=1)

    analysis_df.to_csv(OUTPUT_PATH, index=False)
    build_examples(analysis_df).to_csv(EXAMPLES_PATH, index=False)

    direct_eval_df = analysis_df.rename(columns={"direct_prediction": "prediction"})
    thor_eval_df = analysis_df.rename(columns={"thor_prediction": "prediction"})
    etc_eval_df = analysis_df.rename(columns={"etc_prediction": "prediction"})

    direct_metrics = evaluate_predictions(direct_eval_df, gold_col="polarity", pred_col="prediction")
    thor_metrics = evaluate_predictions(thor_eval_df, gold_col="polarity", pred_col="prediction")
    etc_metrics = evaluate_predictions(etc_eval_df, gold_col="polarity", pred_col="prediction")

    group_counts = analysis_df["etc_change_group"].value_counts()
    decision_counts = analysis_df["controller_decision"].value_counts()
    error_type_counts = analysis_df["error_type"].value_counts()
    triggered_counts = analysis_df["diagnostic_triggered"].map({True: "triggered", False: "not_triggered"}).value_counts()
    label_group_counts = pd.crosstab(analysis_df["polarity"], analysis_df["etc_change_group"])
    domain_group_counts = pd.crosstab(analysis_df["domain"], analysis_df["etc_change_group"])

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Direct predictions: {DIRECT_PATH}\n")
        f.write(f"THOR predictions: {THOR_PATH}\n")
        f.write(f"ETC predictions: {ETC_PATH}\n")
        f.write(f"Analysis output: {OUTPUT_PATH}\n")
        f.write(f"Examples output: {EXAMPLES_PATH}\n")
        f.write(f"n_total: {len(analysis_df)}\n\n")

        f.write("Direct metrics\n")
        f.write(f"accuracy: {direct_metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {direct_metrics['macro_f1']:.6f}\n\n")

        f.write("THOR metrics\n")
        f.write(f"accuracy: {thor_metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {thor_metrics['macro_f1']:.6f}\n\n")

        f.write("ETC-ISA metrics\n")
        f.write(f"accuracy: {etc_metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {etc_metrics['macro_f1']:.6f}\n\n")

        f.write("Diagnostic trigger counts\n")
        f.write(format_count_table(triggered_counts, len(analysis_df)))
        f.write("\n\n")

        f.write("Controller decisions\n")
        f.write(format_count_table(decision_counts, len(analysis_df)))
        f.write("\n\n")

        f.write("Diagnosed error types\n")
        f.write(format_count_table(error_type_counts, len(analysis_df)))
        f.write("\n\n")

        f.write("ETC change groups vs THOR\n")
        f.write(format_count_table(group_counts, len(analysis_df)))
        f.write("\n\n")

        f.write(format_confusion("ETC-ISA confusion matrix", analysis_df["polarity"], analysis_df["etc_prediction"]))
        f.write("\n\n")

        f.write("ETC change groups by gold label\n")
        f.write(label_group_counts.to_string())
        f.write("\n\n")

        f.write("ETC change groups by domain\n")
        f.write(domain_group_counts.to_string())
        f.write("\n")

    print("Done.")
    print(f"Saved analysis to: {OUTPUT_PATH}")
    print(f"Saved examples to: {EXAMPLES_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
