from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import VALID_LABELS, evaluate_predictions


DIRECT_PATH = "results/direct_isa_predictions.csv"
THOR_PATH = "results/thor_isa_predictions.csv"
OUTPUT_PATH = "results/baseline_error_analysis_predictions.csv"
EXAMPLES_PATH = "results/baseline_error_examples_predictions.csv"
METRICS_PATH = "results/baseline_error_analysis_metrics.txt"

KEY_COLS = ["id", "source_sentence_id", "sentence", "target", "from", "to", "polarity"]
EXAMPLES_PER_GROUP = 12


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def correctness_group(row: pd.Series) -> str:
    direct_correct = bool(row["direct_correct"])
    thor_correct = bool(row["thor_correct"])

    if direct_correct and thor_correct:
        return "both_correct"
    if direct_correct and not thor_correct:
        return "direct_correct_thor_wrong"
    if not direct_correct and thor_correct:
        return "thor_correct_direct_wrong"
    return "both_wrong"


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


def build_examples(analysis_df: pd.DataFrame) -> pd.DataFrame:
    example_frames = []
    for group_name in [
        "direct_correct_thor_wrong",
        "thor_correct_direct_wrong",
        "both_wrong",
        "both_correct",
    ]:
        group_df = analysis_df[analysis_df["comparison_group"] == group_name].copy()
        if group_df.empty:
            continue

        # Keep examples deterministic and varied across gold labels.
        examples = (
            group_df.sort_values(["polarity", "id"])
            .groupby("polarity", group_keys=False)
            .head(max(1, EXAMPLES_PER_GROUP // len(VALID_LABELS)))
            .head(EXAMPLES_PER_GROUP)
        )
        example_frames.append(examples)

    if not example_frames:
        return pd.DataFrame()

    return pd.concat(example_frames, ignore_index=True)


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)

    direct_df = pd.read_csv(DIRECT_PATH)
    thor_df = pd.read_csv(THOR_PATH)

    require_columns(direct_df, KEY_COLS + ["domain", "split", "prediction", "raw_output"], DIRECT_PATH)
    require_columns(
        thor_df,
        KEY_COLS
        + [
            "prediction",
            "aspect",
            "opinion",
            "polarity_reasoning",
            "raw_polarity_output",
        ],
        THOR_PATH,
    )

    if direct_df.duplicated(KEY_COLS).any() or thor_df.duplicated(KEY_COLS).any():
        raise ValueError(f"{KEY_COLS} must be unique in both prediction files.")

    direct_cols = [
        *KEY_COLS,
        "domain",
        "split",
        "raw_output",
        "prediction",
    ]
    thor_cols = [
        *KEY_COLS,
        "aspect",
        "opinion",
        "polarity_reasoning",
        "raw_polarity_output",
        "prediction",
    ]

    comparison_df = direct_df[direct_cols].merge(
        thor_df[thor_cols],
        on=KEY_COLS,
        how="inner",
        suffixes=("_direct", "_thor"),
    )

    if len(comparison_df) != len(direct_df) or len(comparison_df) != len(thor_df):
        raise ValueError(
            "Prediction files do not align one-to-one. "
            f"direct={len(direct_df)}, thor={len(thor_df)}, merged={len(comparison_df)}"
        )

    comparison_df = comparison_df.rename(
        columns={
            "raw_output": "direct_raw_output",
            "prediction_direct": "direct_prediction",
            "raw_polarity_output": "thor_raw_polarity_output",
            "prediction_thor": "thor_prediction",
        }
    )

    comparison_df["direct_correct"] = comparison_df["direct_prediction"] == comparison_df["polarity"]
    comparison_df["thor_correct"] = comparison_df["thor_prediction"] == comparison_df["polarity"]
    comparison_df["models_agree"] = comparison_df["direct_prediction"] == comparison_df["thor_prediction"]
    comparison_df["comparison_group"] = comparison_df.apply(correctness_group, axis=1)

    comparison_df.to_csv(OUTPUT_PATH, index=False)

    examples_df = build_examples(comparison_df)
    examples_df.to_csv(EXAMPLES_PATH, index=False)

    direct_eval_df = comparison_df.rename(columns={"direct_prediction": "prediction"})
    thor_eval_df = comparison_df.rename(columns={"thor_prediction": "prediction"})
    direct_metrics = evaluate_predictions(direct_eval_df, gold_col="polarity", pred_col="prediction")
    thor_metrics = evaluate_predictions(thor_eval_df, gold_col="polarity", pred_col="prediction")

    group_counts = comparison_df["comparison_group"].value_counts()
    agreement_counts = comparison_df["models_agree"].map({True: "agree", False: "disagree"}).value_counts()
    domain_group_counts = pd.crosstab(comparison_df["domain"], comparison_df["comparison_group"])
    label_group_counts = pd.crosstab(comparison_df["polarity"], comparison_df["comparison_group"])

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Direct predictions: {DIRECT_PATH}\n")
        f.write(f"THOR predictions: {THOR_PATH}\n")
        f.write(f"Analysis output: {OUTPUT_PATH}\n")
        f.write(f"Examples output: {EXAMPLES_PATH}\n")
        f.write(f"n_total: {len(comparison_df)}\n\n")

        f.write("Direct metrics\n")
        f.write(f"accuracy: {direct_metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {direct_metrics['macro_f1']:.6f}\n\n")

        f.write("THOR metrics\n")
        f.write(f"accuracy: {thor_metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {thor_metrics['macro_f1']:.6f}\n\n")

        f.write("Correctness groups\n")
        f.write(format_count_table(group_counts, len(comparison_df)))
        f.write("\n\n")

        f.write("Model agreement\n")
        f.write(format_count_table(agreement_counts, len(comparison_df)))
        f.write("\n\n")

        f.write(format_confusion("Direct confusion matrix", comparison_df["polarity"], comparison_df["direct_prediction"]))
        f.write("\n\n")
        f.write(format_confusion("THOR confusion matrix", comparison_df["polarity"], comparison_df["thor_prediction"]))
        f.write("\n\n")

        f.write("Correctness groups by gold label\n")
        f.write(label_group_counts.to_string())
        f.write("\n\n")

        f.write("Correctness groups by domain\n")
        f.write(domain_group_counts.to_string())
        f.write("\n")

    print("Done.")
    print(f"Saved analysis to: {OUTPUT_PATH}")
    print(f"Saved examples to: {EXAMPLES_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
