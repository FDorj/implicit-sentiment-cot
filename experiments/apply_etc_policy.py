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

POLICY_MODE = os.getenv("ETC_SELECTED_POLICY_MODE", "train_calibrated").strip().lower()

FALLBACK_POLICY = os.getenv("ETC_SELECTED_FALLBACK_POLICY", "direct").strip().lower()
TRUST_NO_ERROR_ON_DISAGREEMENT = os.getenv("ETC_SELECTED_TRUST_NO_ERROR_ON_DISAGREEMENT", "0") == "1"
MIN_CORRECTABLE_CONFIDENCE = os.getenv("ETC_SELECTED_MIN_CORRECTABLE_CONFIDENCE", "high").strip().lower()
ACCEPT_ERROR_TYPES_RAW = os.getenv("ETC_SELECTED_ACCEPT_ERROR_TYPES", "missed_implicit_positive").strip().lower()

TRAIN_SPLIT = os.getenv("ETC_SELECTED_TRAIN_SPLIT", "train").strip().lower()
KEY_COLUMNS_RAW = os.getenv(
    "ETC_SELECTED_KEY_COLUMNS",
    "direct_prediction,error_type,diagnostic_confidence,domain",
).strip()
CANDIDATE_SOURCES_RAW = os.getenv(
    "ETC_SELECTED_CANDIDATE_SOURCES",
    "direct,thor,diagnostic",
).strip()
DEFAULT_SOURCE = os.getenv("ETC_SELECTED_DEFAULT_SOURCE", "direct").strip().lower()

BASE_REQUIRED_COLUMNS = [
    "polarity",
    "split",
    "direct_prediction",
    "thor_prediction",
    "diagnostic_label",
    "error_type",
    "diagnostic_confidence",
]
SOURCE_TO_COLUMN = {
    "direct": "direct_prediction",
    "thor": "thor_prediction",
    "diagnostic": "diagnostic_label",
}


def parse_csv_list(raw_value: str) -> list[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


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


def validate_sources(candidate_sources: list[str], default_source: str) -> None:
    invalid = [source for source in [*candidate_sources, default_source] if source not in SOURCE_TO_COLUMN]
    if invalid:
        raise ValueError(f"Unsupported ETC_SELECTED source(s): {sorted(set(invalid))}")


def learn_calibrated_policy(
    df: pd.DataFrame,
    key_columns: list[str],
    candidate_sources: list[str],
    default_source: str,
    train_split: str,
) -> dict[tuple, str]:
    train_df = df[df["split"].astype(str).str.lower() == train_split].copy()
    if train_df.empty:
        raise ValueError(f"No rows found for ETC_SELECTED_TRAIN_SPLIT={train_split!r}")

    policy: dict[tuple, str] = {}
    for key, group in train_df.groupby(key_columns, dropna=False):
        best_source = default_source
        best_score = -1
        for source in candidate_sources:
            pred_col = SOURCE_TO_COLUMN[source]
            score = int((group[pred_col] == group["polarity"]).sum())
            if score > best_score:
                best_score = score
                best_source = source

        policy[key if isinstance(key, tuple) else (key,)] = best_source

    return policy


def apply_manual_policy(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    accepted_error_types = parse_accepted_error_types(ACCEPT_ERROR_TYPES_RAW)
    predictions = []
    decisions = []
    sources = []
    policy_keys = []

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
        sources.append("manual_controller")
        policy_keys.append("")

    return predictions, decisions, sources, policy_keys


def apply_train_calibrated_policy(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str], dict[tuple, str]]:
    key_columns = parse_csv_list(KEY_COLUMNS_RAW)
    candidate_sources = parse_csv_list(CANDIDATE_SOURCES_RAW)
    validate_sources(candidate_sources, DEFAULT_SOURCE)
    require_columns(df, key_columns, ETC_PATH)

    policy = learn_calibrated_policy(
        df=df,
        key_columns=key_columns,
        candidate_sources=candidate_sources,
        default_source=DEFAULT_SOURCE,
        train_split=TRAIN_SPLIT,
    )

    predictions = []
    decisions = []
    sources = []
    policy_keys = []

    for _, row in df.iterrows():
        key = tuple(row[col] for col in key_columns)
        source = policy.get(key, DEFAULT_SOURCE)
        prediction = row[SOURCE_TO_COLUMN[source]]

        predictions.append(prediction)
        decisions.append(f"train_calibrated_use_{source}")
        sources.append(source)
        policy_keys.append("|".join(f"{col}={row[col]}" for col in key_columns))

    return predictions, decisions, sources, policy_keys, policy


def write_split_metrics(f, df: pd.DataFrame) -> None:
    for split in ["train", "test"]:
        split_df = df[df["split"].astype(str).str.lower() == split].copy()
        if split_df.empty:
            continue

        split_result = evaluate_predictions(split_df, gold_col="polarity", pred_col="selected_prediction")
        f.write(f"\n{split} metrics\n")
        f.write(f"n_total: {split_result['n_total']}\n")
        f.write(f"n_eval: {split_result['n_eval']}\n")
        f.write(f"accuracy: {split_result['accuracy']:.6f}\n")
        f.write(f"macro_f1: {split_result['macro_f1']:.6f}\n")


def write_domain_metrics(f, df: pd.DataFrame) -> None:
    if "domain" not in df.columns:
        return

    f.write("\ndomain metrics\n")
    for domain, domain_df in df.groupby("domain"):
        domain_result = evaluate_predictions(domain_df, gold_col="polarity", pred_col="selected_prediction")
        f.write(f"{domain}: accuracy={domain_result['accuracy']:.6f}, macro_f1={domain_result['macro_f1']:.6f}, n={domain_result['n_eval']}\n")


def write_change_summary(f, df: pd.DataFrame) -> None:
    changed_vs_direct = df["selected_prediction"] != df["direct_prediction"]
    changed_vs_thor = df["selected_prediction"] != df["thor_prediction"]

    gain_vs_direct = (df["direct_prediction"] != df["polarity"]) & (df["selected_prediction"] == df["polarity"])
    loss_vs_direct = (df["direct_prediction"] == df["polarity"]) & (df["selected_prediction"] != df["polarity"])

    f.write("\nchange summary vs direct/thor\n")
    f.write(f"changed_vs_direct: {int(changed_vs_direct.sum())}\n")
    f.write(f"changed_vs_thor: {int(changed_vs_thor.sum())}\n")
    f.write(f"gain_vs_direct: {int(gain_vs_direct.sum())}\n")
    f.write(f"loss_vs_direct: {int(loss_vs_direct.sum())}\n")


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ETC_PATH)
    require_columns(df, BASE_REQUIRED_COLUMNS, ETC_PATH)

    learned_policy: dict[tuple, str] = {}
    if POLICY_MODE == "manual":
        predictions, decisions, sources, policy_keys = apply_manual_policy(df)
    elif POLICY_MODE == "train_calibrated":
        predictions, decisions, sources, policy_keys, learned_policy = apply_train_calibrated_policy(df)
    else:
        raise ValueError(f"Unsupported ETC_SELECTED_POLICY_MODE: {POLICY_MODE}")

    df["selected_prediction"] = predictions
    df["selected_controller_decision"] = decisions
    df["selected_source"] = sources
    df["selected_policy_key"] = policy_keys
    df.to_csv(OUTPUT_PATH, index=False)

    metrics = evaluate_predictions(df, gold_col="polarity", pred_col="selected_prediction")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"ETC predictions: {ETC_PATH}\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
        f.write(f"policy_mode: {POLICY_MODE}\n")

        if POLICY_MODE == "manual":
            accepted_error_types = parse_accepted_error_types(ACCEPT_ERROR_TYPES_RAW)
            f.write(f"fallback_policy: {FALLBACK_POLICY}\n")
            f.write(f"trust_no_error_on_disagreement: {TRUST_NO_ERROR_ON_DISAGREEMENT}\n")
            f.write(f"min_correctable_confidence: {MIN_CORRECTABLE_CONFIDENCE}\n")
            f.write(f"accepted_error_types: {','.join(sorted(accepted_error_types))}\n")
        else:
            f.write(f"train_split: {TRAIN_SPLIT}\n")
            f.write(f"key_columns: {','.join(parse_csv_list(KEY_COLUMNS_RAW))}\n")
            f.write(f"candidate_sources: {','.join(parse_csv_list(CANDIDATE_SOURCES_RAW))}\n")
            f.write(f"default_source: {DEFAULT_SOURCE}\n")

        f.write(f"n_total: {metrics['n_total']}\n")
        f.write(f"n_eval: {metrics['n_eval']}\n")
        f.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {metrics['macro_f1']:.6f}\n\n")
        f.write(metrics["report"])
        f.write("\n")

        write_split_metrics(f, df)
        write_domain_metrics(f, df)
        write_change_summary(f, df)

        f.write("\nselected source counts\n")
        f.write(df["selected_source"].value_counts().to_string())
        f.write("\n")

        if learned_policy:
            f.write("\nlearned policy\n")
            for key, source in sorted(learned_policy.items()):
                key_text = ", ".join(
                    f"{column}={value}"
                    for column, value in zip(parse_csv_list(KEY_COLUMNS_RAW), key)
                )
                f.write(f"{key_text} -> {source}\n")

    print("Done.")
    print(f"Saved selected ETC predictions to: {OUTPUT_PATH}")
    print(f"Saved selected ETC metrics to: {METRICS_PATH}")
    print(f"Policy mode: {POLICY_MODE}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
