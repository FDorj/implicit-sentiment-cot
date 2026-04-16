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
REFLECTION_PATH = "results/simple_reflection_isa_predictions.csv"
OUTPUT_PATH = "results/reflection_error_analysis_predictions.csv"
EXAMPLES_PATH = "results/reflection_error_examples_predictions.csv"
METRICS_PATH = "results/reflection_error_analysis_metrics.txt"

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


def reflection_change_group(row: pd.Series) -> str:
    thor_correct = bool(row["thor_correct"])
    reflection_correct = bool(row["reflection_correct"])
    changed = bool(row["reflection_changed_thor"])

    if not changed and reflection_correct:
        return "unchanged_correct"
    if not changed and not reflection_correct:
        return "unchanged_wrong"
    if not thor_correct and reflection_correct:
        return "fixed_by_reflection"
    if thor_correct and not reflection_correct:
        return "broken_by_reflection"
    return "changed_still_wrong"


def build_examples(analysis_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for group_name in [
        "fixed_by_reflection",
        "broken_by_reflection",
        "changed_still_wrong",
        "unchanged_wrong",
    ]:
        group_df = analysis_df[analysis_df["reflection_change_group"] == group_name].copy()
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
    reflection_df = pd.read_csv(REFLECTION_PATH)

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
    require_columns(reflection_df, KEY_COLS + ["thor_prediction", "reflection_raw_output", "reflection_prediction"], REFLECTION_PATH)

    for source, df in [(DIRECT_PATH, direct_df), (THOR_PATH, thor_df), (REFLECTION_PATH, reflection_df)]:
        if df.duplicated(KEY_COLS).any():
            raise ValueError(f"{KEY_COLS} must be unique in {source}.")

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
    reflection_cols = [
        *KEY_COLS,
        "reflection_raw_output",
        "reflection_prediction",
    ]

    analysis_df = (
        direct_df[direct_cols]
        .merge(thor_df[thor_cols], on=KEY_COLS, how="inner", suffixes=("_direct", "_thor"))
        .merge(reflection_df[reflection_cols], on=KEY_COLS, how="inner")
    )

    if len(analysis_df) != len(direct_df) or len(analysis_df) != len(thor_df) or len(analysis_df) != len(reflection_df):
        raise ValueError(
            "Prediction files do not align one-to-one. "
            f"direct={len(direct_df)}, thor={len(thor_df)}, reflection={len(reflection_df)}, merged={len(analysis_df)}"
        )

    analysis_df = analysis_df.rename(
        columns={
            "raw_output": "direct_raw_output",
            "prediction_direct": "direct_prediction",
            "raw_polarity_output": "thor_raw_polarity_output",
            "prediction_thor": "thor_prediction",
        }
    )

    analysis_df["direct_correct"] = analysis_df["direct_prediction"] == analysis_df["polarity"]
    analysis_df["thor_correct"] = analysis_df["thor_prediction"] == analysis_df["polarity"]
    analysis_df["reflection_correct"] = analysis_df["reflection_prediction"] == analysis_df["polarity"]
    analysis_df["reflection_changed_thor"] = analysis_df["reflection_prediction"] != analysis_df["thor_prediction"]
    analysis_df["reflection_change_group"] = analysis_df.apply(reflection_change_group, axis=1)

    analysis_df.to_csv(OUTPUT_PATH, index=False)
    build_examples(analysis_df).to_csv(EXAMPLES_PATH, index=False)

    direct_eval_df = analysis_df.rename(columns={"direct_prediction": "prediction"})
    thor_eval_df = analysis_df.rename(columns={"thor_prediction": "prediction"})
    reflection_eval_df = analysis_df.rename(columns={"reflection_prediction": "prediction"})

    direct_metrics = evaluate_predictions(direct_eval_df, gold_col="polarity", pred_col="prediction")
    thor_metrics = evaluate_predictions(thor_eval_df, gold_col="polarity", pred_col="prediction")
    reflection_metrics = evaluate_predictions(reflection_eval_df, gold_col="polarity", pred_col="prediction")

    group_counts = analysis_df["reflection_change_group"].value_counts()
    changed_counts = analysis_df["reflection_changed_thor"].map({True: "changed", False: "unchanged"}).value_counts()
    label_group_counts = pd.crosstab(analysis_df["polarity"], analysis_df["reflection_change_group"])
    domain_group_counts = pd.crosstab(analysis_df["domain"], analysis_df["reflection_change_group"])

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Direct predictions: {DIRECT_PATH}\n")
        f.write(f"THOR predictions: {THOR_PATH}\n")
        f.write(f"Reflection predictions: {REFLECTION_PATH}\n")
        f.write(f"Analysis output: {OUTPUT_PATH}\n")
        f.write(f"Examples output: {EXAMPLES_PATH}\n")
        f.write(f"n_total: {len(analysis_df)}\n\n")

        f.write("Direct metrics\n")
        f.write(f"accuracy: {direct_metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {direct_metrics['macro_f1']:.6f}\n\n")

        f.write("THOR metrics\n")
        f.write(f"accuracy: {thor_metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {thor_metrics['macro_f1']:.6f}\n\n")

        f.write("Simple reflection metrics\n")
        f.write(f"accuracy: {reflection_metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {reflection_metrics['macro_f1']:.6f}\n\n")

        f.write("Reflection changes vs THOR\n")
        f.write(format_count_table(changed_counts, len(analysis_df)))
        f.write("\n\n")

        f.write("Reflection change groups\n")
        f.write(format_count_table(group_counts, len(analysis_df)))
        f.write("\n\n")

        f.write(format_confusion("Simple reflection confusion matrix", analysis_df["polarity"], analysis_df["reflection_prediction"]))
        f.write("\n\n")

        f.write("Reflection change groups by gold label\n")
        f.write(label_group_counts.to_string())
        f.write("\n\n")

        f.write("Reflection change groups by domain\n")
        f.write(domain_group_counts.to_string())
        f.write("\n")

    print("Done.")
    print(f"Saved analysis to: {OUTPUT_PATH}")
    print(f"Saved examples to: {EXAMPLES_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
