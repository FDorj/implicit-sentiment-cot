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
GUARDED_KEY_COLUMNS_RAW = os.getenv(
    "ETC_SELECTED_GUARDED_KEY_COLUMNS",
    "direct_prediction,thor_prediction,error_type,diagnostic_confidence,domain",
).strip()
GUARDED_MIN_SUPPORT = int(os.getenv("ETC_SELECTED_MIN_SUPPORT", "10"))
GUARDED_MIN_MARGIN_DEFAULT = int(os.getenv("ETC_SELECTED_MIN_MARGIN_DEFAULT", "2"))
GUARDED_MIN_MARGIN_SECOND = int(os.getenv("ETC_SELECTED_MIN_MARGIN_SECOND", "1"))
GUARDED_MIN_RELATIVE_GAIN = float(os.getenv("ETC_SELECTED_MIN_RELATIVE_GAIN", "0.05"))

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


def score_sources(group: pd.DataFrame, candidate_sources: list[str]) -> dict[str, int]:
    return {
        source: int((group[SOURCE_TO_COLUMN[source]] == group["polarity"]).sum())
        for source in candidate_sources
    }


def best_source_by_score(scores: dict[str, int], candidate_sources: list[str]) -> tuple[str, int]:
    best_source = candidate_sources[0]
    best_score = -1
    for source in candidate_sources:
        score = scores[source]
        if score > best_score:
            best_score = score
            best_source = source

    return best_source, best_score


def learn_guarded_calibrated_policy(
    df: pd.DataFrame,
    key_columns: list[str],
    candidate_sources: list[str],
    default_source: str,
    train_split: str,
    min_support: int,
    min_margin_default: int,
    min_margin_second: int,
    min_relative_gain: float,
) -> tuple[dict[tuple, str], dict[tuple, dict]]:
    train_df = df[df["split"].astype(str).str.lower() == train_split].copy()
    if train_df.empty:
        raise ValueError(f"No rows found for ETC_SELECTED_TRAIN_SPLIT={train_split!r}")

    policy: dict[tuple, str] = {}
    metadata: dict[tuple, dict] = {}
    score_sources_order = list(dict.fromkeys([*candidate_sources, default_source]))

    for key, group in train_df.groupby(key_columns, dropna=False):
        normalized_key = key if isinstance(key, tuple) else (key,)
        support = len(group)
        scores = score_sources(group, score_sources_order)
        best_source, best_score = best_source_by_score(scores, score_sources_order)
        sorted_scores = sorted(scores.values(), reverse=True)
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else best_score
        default_score = scores[default_source]
        margin_default = best_score - default_score
        margin_second = best_score - second_score
        relative_gain = (best_score / support) - (default_score / support)

        selected_source = best_source
        fallback_reason = ""
        if best_source != default_source:
            if support < min_support:
                selected_source = default_source
                fallback_reason = "low_support"
            elif margin_default < min_margin_default:
                selected_source = default_source
                fallback_reason = "low_default_margin"
            elif margin_second < min_margin_second:
                selected_source = default_source
                fallback_reason = "low_second_margin"
            elif relative_gain < min_relative_gain:
                selected_source = default_source
                fallback_reason = "low_relative_gain"

        policy[normalized_key] = selected_source
        metadata[normalized_key] = {
            "source": selected_source,
            "support": support,
            "scores": scores,
            "best_source": best_source,
            "best_score": best_score,
            "second_score": second_score,
            "default_score": default_score,
            "margin_default": margin_default,
            "margin_second": margin_second,
            "relative_gain": relative_gain,
            "fallback_reason": fallback_reason,
        }

    return policy, metadata


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


def apply_guarded_train_calibrated_policy(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
    candidate_sources: list[str] | None = None,
    default_source: str = DEFAULT_SOURCE,
    train_split: str = TRAIN_SPLIT,
    min_support: int = GUARDED_MIN_SUPPORT,
    min_margin_default: int = GUARDED_MIN_MARGIN_DEFAULT,
    min_margin_second: int = GUARDED_MIN_MARGIN_SECOND,
    min_relative_gain: float = GUARDED_MIN_RELATIVE_GAIN,
) -> tuple[list[str], list[str], list[str], list[str], dict[tuple, str], dict[tuple, dict]]:
    selected_key_columns = key_columns or parse_csv_list(GUARDED_KEY_COLUMNS_RAW)
    selected_candidate_sources = candidate_sources or parse_csv_list(CANDIDATE_SOURCES_RAW)
    validate_sources(selected_candidate_sources, default_source)
    require_columns(df, selected_key_columns, ETC_PATH)

    policy, policy_metadata = learn_guarded_calibrated_policy(
        df=df,
        key_columns=selected_key_columns,
        candidate_sources=selected_candidate_sources,
        default_source=default_source,
        train_split=train_split,
        min_support=min_support,
        min_margin_default=min_margin_default,
        min_margin_second=min_margin_second,
        min_relative_gain=min_relative_gain,
    )

    predictions = []
    decisions = []
    sources = []
    policy_keys = []

    for _, row in df.iterrows():
        key = tuple(row[col] for col in selected_key_columns)
        source = policy.get(key, default_source)
        prediction = row[SOURCE_TO_COLUMN[source]]
        metadata = policy_metadata.get(key)
        fallback_reason = "unseen_profile" if metadata is None else metadata["fallback_reason"]
        decision = f"guarded_train_calibrated_use_{source}"
        if fallback_reason:
            decision = f"{decision}_{fallback_reason}"

        predictions.append(prediction)
        decisions.append(decision)
        sources.append(source)
        policy_keys.append("|".join(f"{col}={row[col]}" for col in selected_key_columns))

    return predictions, decisions, sources, policy_keys, policy, policy_metadata


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
    learned_policy_metadata: dict[tuple, dict] = {}
    if POLICY_MODE == "manual":
        predictions, decisions, sources, policy_keys = apply_manual_policy(df)
    elif POLICY_MODE == "train_calibrated":
        predictions, decisions, sources, policy_keys, learned_policy = apply_train_calibrated_policy(df)
    elif POLICY_MODE == "guarded_train_calibrated":
        (
            predictions,
            decisions,
            sources,
            policy_keys,
            learned_policy,
            learned_policy_metadata,
        ) = apply_guarded_train_calibrated_policy(df)
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
            key_columns = parse_csv_list(KEY_COLUMNS_RAW)
            if POLICY_MODE == "guarded_train_calibrated":
                key_columns = parse_csv_list(GUARDED_KEY_COLUMNS_RAW)
            f.write(f"key_columns: {','.join(key_columns)}\n")
            f.write(f"candidate_sources: {','.join(parse_csv_list(CANDIDATE_SOURCES_RAW))}\n")
            f.write(f"default_source: {DEFAULT_SOURCE}\n")
            if POLICY_MODE == "guarded_train_calibrated":
                f.write(f"min_support: {GUARDED_MIN_SUPPORT}\n")
                f.write(f"min_margin_default: {GUARDED_MIN_MARGIN_DEFAULT}\n")
                f.write(f"min_margin_second: {GUARDED_MIN_MARGIN_SECOND}\n")
                f.write(f"min_relative_gain: {GUARDED_MIN_RELATIVE_GAIN:.6f}\n")

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
                key_columns = parse_csv_list(KEY_COLUMNS_RAW)
                if POLICY_MODE == "guarded_train_calibrated":
                    key_columns = parse_csv_list(GUARDED_KEY_COLUMNS_RAW)
                key_text = ", ".join(
                    f"{column}={value}"
                    for column, value in zip(key_columns, key)
                )
                metadata = learned_policy_metadata.get(key)
                if metadata:
                    scores_text = ",".join(
                        f"{score_source}:{score}"
                        for score_source, score in sorted(metadata["scores"].items())
                    )
                    f.write(
                        f"{key_text} -> {source} "
                        f"(support={metadata['support']}, scores={scores_text}, "
                        f"best={metadata['best_source']}:{metadata['best_score']}, "
                        f"margin_default={metadata['margin_default']}, "
                        f"margin_second={metadata['margin_second']}, "
                        f"relative_gain={metadata['relative_gain']:.6f}, "
                        f"fallback_reason={metadata['fallback_reason']})\n"
                    )
                else:
                    f.write(f"{key_text} -> {source}\n")

    print("Done.")
    print(f"Saved selected ETC predictions to: {OUTPUT_PATH}")
    print(f"Saved selected ETC metrics to: {METRICS_PATH}")
    print(f"Policy mode: {POLICY_MODE}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
