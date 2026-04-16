from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import select_final_label
from src.evaluator import evaluate_predictions
from src.experiment_config import result_path


ETC_PATH = result_path("etc_isa", "predictions.csv", "ETC_PREDICTIONS_PATH")
OUTPUT_PATH = result_path("etc_policy_ablation", "predictions.csv", "ETC_POLICY_ABLATION_OUTPUT_PATH")
METRICS_PATH = result_path("etc_policy_ablation", "metrics.txt", "ETC_POLICY_ABLATION_METRICS_PATH")

POLICIES = [
    {
        "name": "direct_baseline",
        "mode": "copy",
        "source_col": "direct_prediction",
    },
    {
        "name": "thor_baseline",
        "mode": "copy",
        "source_col": "thor_prediction",
    },
    {
        "name": "etc_direct_medium_no_trust",
        "fallback_policy": "direct",
        "trust_no_error_on_disagreement": False,
        "min_correctable_confidence": "medium",
    },
    {
        "name": "etc_direct_high_no_trust",
        "fallback_policy": "direct",
        "trust_no_error_on_disagreement": False,
        "min_correctable_confidence": "high",
    },
    {
        "name": "etc_direct_high_positive_only",
        "fallback_policy": "direct",
        "trust_no_error_on_disagreement": False,
        "min_correctable_confidence": "high",
        "accepted_error_types": {"missed_implicit_positive"},
    },
    {
        "name": "etc_direct_high_negative_only",
        "fallback_policy": "direct",
        "trust_no_error_on_disagreement": False,
        "min_correctable_confidence": "high",
        "accepted_error_types": {"missed_implicit_negative"},
    },
    {
        "name": "etc_direct_medium_trust_no_error",
        "fallback_policy": "direct",
        "trust_no_error_on_disagreement": True,
        "min_correctable_confidence": "medium",
    },
    {
        "name": "etc_thor_medium_no_trust",
        "fallback_policy": "thor",
        "trust_no_error_on_disagreement": False,
        "min_correctable_confidence": "medium",
    },
    {
        "name": "etc_thor_high_no_trust",
        "fallback_policy": "thor",
        "trust_no_error_on_disagreement": False,
        "min_correctable_confidence": "high",
    },
]

REQUIRED_COLUMNS = [
    "polarity",
    "direct_prediction",
    "thor_prediction",
    "diagnostic_label",
    "error_type",
    "diagnostic_confidence",
]


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def apply_policy(row: pd.Series, policy: dict) -> tuple[str, str]:
    if policy.get("mode") == "copy":
        return row[policy["source_col"]], f"copy_{policy['source_col']}"

    return select_final_label(
        direct_label=row["direct_prediction"],
        thor_label=row["thor_prediction"],
        proposed_label=row["diagnostic_label"],
        error_type=row["error_type"],
        confidence=row["diagnostic_confidence"],
        fallback_policy=policy["fallback_policy"],
        trust_no_error_on_disagreement=policy["trust_no_error_on_disagreement"],
        min_correctable_confidence=policy["min_correctable_confidence"],
        accepted_error_types=policy.get("accepted_error_types"),
    )


def summarize_policy(df: pd.DataFrame, predictions: list[str], policy_name: str, split: str) -> dict:
    if split == "overall":
        split_df = df
        split_predictions = predictions
    else:
        split_mask = df["split"] == split
        split_df = df[split_mask]
        split_predictions = pd.Series(predictions, index=df.index).loc[split_df.index].tolist()

    metrics = evaluate_predictions(
        pd.DataFrame(
            {
                "polarity": split_df["polarity"],
                "prediction": split_predictions,
            }
        ),
        gold_col="polarity",
        pred_col="prediction",
    )

    return {
        "policy": policy_name,
        "split": split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "n_eval": metrics["n_eval"],
        "changed_vs_thor": int(sum(pd.Series(split_predictions).to_numpy() != split_df["thor_prediction"].to_numpy())),
        "changed_vs_direct": int(sum(pd.Series(split_predictions).to_numpy() != split_df["direct_prediction"].to_numpy())),
    }


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ETC_PATH)
    require_columns(df, REQUIRED_COLUMNS, ETC_PATH)

    summary_rows = []
    output_df = df.copy()

    for policy in POLICIES:
        policy_name = policy["name"]
        predictions = []
        decisions = []

        for _, row in df.iterrows():
            prediction, decision = apply_policy(row, policy)
            predictions.append(prediction)
            decisions.append(decision)

        pred_col = f"{policy_name}_prediction"
        decision_col = f"{policy_name}_decision"
        output_df[pred_col] = predictions
        output_df[decision_col] = decisions

        for split in ["overall", "train", "test"]:
            summary_rows.append(summarize_policy(df, predictions, policy_name, split))

    summary_df = pd.DataFrame(summary_rows).sort_values(["split", "macro_f1", "accuracy"], ascending=False)
    output_df.to_csv(OUTPUT_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"ETC predictions: {ETC_PATH}\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
        f.write(f"n_total: {len(df)}\n\n")
        f.write(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        f.write("\n")

    print("Done.")
    print(f"Saved policy predictions to: {OUTPUT_PATH}")
    print(f"Saved policy metrics to: {METRICS_PATH}")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
