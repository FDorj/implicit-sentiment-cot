from pathlib import Path
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.apply_etc_policy import (  # noqa: E402
    SOURCE_TO_COLUMN,
    learn_calibrated_policy,
    learn_guarded_calibrated_policy,
    parse_csv_list,
    require_columns,
    validate_sources,
)
from src.evaluator import evaluate_predictions  # noqa: E402


ETC_PATH = os.getenv(
    "ETC_PREDICTIONS_PATH",
    str(Path("results") / "etc_thor_originalish_sc3_isa_predictions.csv"),
)
OUTPUT_CSV_PATH = os.getenv(
    "GUARDED_ETC_POLICY_ABLATION_CSV",
    str(Path("results") / "guarded_etc_policy_ablation_metrics.csv"),
)
OUTPUT_TXT_PATH = os.getenv(
    "GUARDED_ETC_POLICY_ABLATION_TXT",
    str(Path("results") / "guarded_etc_policy_ablation_metrics.txt"),
)

TRAIN_SPLIT = os.getenv("ETC_SELECTED_TRAIN_SPLIT", "train").strip().lower()
CANDIDATE_SOURCES = parse_csv_list(os.getenv("ETC_SELECTED_CANDIDATE_SOURCES", "direct,thor,diagnostic"))
DEFAULT_SOURCE = os.getenv("ETC_SELECTED_DEFAULT_SOURCE", "direct").strip().lower()

CURRENT_KEY_COLUMNS = ["direct_prediction", "error_type", "diagnostic_confidence", "domain"]
RICHER_KEY_COLUMNS = [
    "direct_prediction",
    "thor_prediction",
    "error_type",
    "diagnostic_confidence",
    "domain",
]
MIN_SUPPORT_CANDIDATES = [5, 10, 20]
MIN_MARGIN_DEFAULT_CANDIDATES = [1, 2, 3]
MIN_MARGIN_SECOND_CANDIDATES = [0, 1, 2]
MIN_RELATIVE_GAIN_CANDIDATES = [0.03, 0.05]

REQUIRED_COLUMNS = sorted(
    {
        "polarity",
        "split",
        "direct_prediction",
        "thor_prediction",
        "diagnostic_label",
        "error_type",
        "diagnostic_confidence",
        "domain",
    }
)


def format_gain(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_policy_configs() -> list[dict]:
    configs = [
        {
            "name": "current_key_unguarded",
            "mode": "unguarded",
            "key_columns": CURRENT_KEY_COLUMNS,
            "key_variant": "current",
        },
        {
            "name": "richer_key_unguarded",
            "mode": "unguarded",
            "key_columns": RICHER_KEY_COLUMNS,
            "key_variant": "richer",
        },
    ]

    key_variants = [
        ("current", CURRENT_KEY_COLUMNS),
        ("richer", RICHER_KEY_COLUMNS),
    ]
    for key_variant, key_columns in key_variants:
        for min_support in MIN_SUPPORT_CANDIDATES:
            for min_margin_default in MIN_MARGIN_DEFAULT_CANDIDATES:
                for min_margin_second in MIN_MARGIN_SECOND_CANDIDATES:
                    for min_relative_gain in MIN_RELATIVE_GAIN_CANDIDATES:
                        configs.append(
                            {
                                "name": (
                                    f"guarded_{key_variant}"
                                    f"_s{min_support}"
                                    f"_md{min_margin_default}"
                                    f"_ms{min_margin_second}"
                                    f"_rg{format_gain(min_relative_gain)}"
                                ),
                                "mode": "guarded",
                                "key_columns": key_columns,
                                "key_variant": key_variant,
                                "min_support": min_support,
                                "min_margin_default": min_margin_default,
                                "min_margin_second": min_margin_second,
                                "min_relative_gain": min_relative_gain,
                            }
                        )

    return configs


def apply_learned_policy(
    df: pd.DataFrame,
    key_columns: list[str],
    policy: dict[tuple, str],
    default_source: str,
    policy_metadata: dict[tuple, dict] | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    predictions = []
    decisions = []
    sources = []
    policy_keys = []

    for _, row in df.iterrows():
        key = tuple(row[col] for col in key_columns)
        source = policy.get(key, default_source)
        metadata = policy_metadata.get(key) if policy_metadata else None
        fallback_reason = ""
        if policy_metadata is not None:
            fallback_reason = "unseen_profile" if metadata is None else metadata["fallback_reason"]

        decision = f"use_{source}"
        if fallback_reason:
            decision = f"{decision}_{fallback_reason}"

        predictions.append(row[SOURCE_TO_COLUMN[source]])
        decisions.append(decision)
        sources.append(source)
        policy_keys.append("|".join(f"{column}={row[column]}" for column in key_columns))

    return predictions, decisions, sources, policy_keys


def run_policy_config(
    df: pd.DataFrame,
    config: dict,
    candidate_sources: list[str],
    default_source: str,
    train_split: str,
) -> tuple[list[str], list[str], list[str], list[str], dict[tuple, str], dict[tuple, dict]]:
    key_columns = config["key_columns"]
    if config["mode"] == "unguarded":
        policy = learn_calibrated_policy(
            df=df,
            key_columns=key_columns,
            candidate_sources=candidate_sources,
            default_source=default_source,
            train_split=train_split,
        )
        predictions, decisions, sources, policy_keys = apply_learned_policy(
            df=df,
            key_columns=key_columns,
            policy=policy,
            default_source=default_source,
        )
        return predictions, decisions, sources, policy_keys, policy, {}

    policy, policy_metadata = learn_guarded_calibrated_policy(
        df=df,
        key_columns=key_columns,
        candidate_sources=candidate_sources,
        default_source=default_source,
        train_split=train_split,
        min_support=config["min_support"],
        min_margin_default=config["min_margin_default"],
        min_margin_second=config["min_margin_second"],
        min_relative_gain=config["min_relative_gain"],
    )
    predictions, decisions, sources, policy_keys = apply_learned_policy(
        df=df,
        key_columns=key_columns,
        policy=policy,
        default_source=default_source,
        policy_metadata=policy_metadata,
    )
    return predictions, decisions, sources, policy_keys, policy, policy_metadata


def summarize_predictions(
    df: pd.DataFrame,
    predictions: list[str],
    sources: list[str],
    config: dict,
    split: str,
) -> dict:
    prediction_series = pd.Series(predictions, index=df.index)
    source_series = pd.Series(sources, index=df.index)

    if split == "overall":
        split_df = df
        split_predictions = prediction_series
        split_sources = source_series
    else:
        split_mask = df["split"].astype(str).str.lower() == split
        split_df = df[split_mask]
        split_predictions = prediction_series.loc[split_df.index]
        split_sources = source_series.loc[split_df.index]

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
    source_counts = split_sources.value_counts().to_dict()

    return {
        "policy": config["name"],
        "mode": config["mode"],
        "key_variant": config["key_variant"],
        "split": split,
        "n_eval": metrics["n_eval"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "direct_count": int(source_counts.get("direct", 0)),
        "thor_count": int(source_counts.get("thor", 0)),
        "diagnostic_count": int(source_counts.get("diagnostic", 0)),
        "min_support": config.get("min_support", ""),
        "min_margin_default": config.get("min_margin_default", ""),
        "min_margin_second": config.get("min_margin_second", ""),
        "min_relative_gain": config.get("min_relative_gain", ""),
    }


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    validate_sources(CANDIDATE_SOURCES, DEFAULT_SOURCE)

    df = pd.read_csv(ETC_PATH)
    require_columns(df, REQUIRED_COLUMNS, ETC_PATH)

    summary_rows = []
    for config in build_policy_configs():
        require_columns(df, config["key_columns"], ETC_PATH)
        predictions, _, sources, _, _, _ = run_policy_config(
            df=df,
            config=config,
            candidate_sources=CANDIDATE_SOURCES,
            default_source=DEFAULT_SOURCE,
            train_split=TRAIN_SPLIT,
        )
        for split in ["overall", "train", "test"]:
            summary_rows.append(summarize_predictions(df, predictions, sources, config, split))

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["split", "macro_f1", "accuracy", "policy"],
        ascending=[True, False, False, True],
    )
    summary_df.to_csv(OUTPUT_CSV_PATH, index=False)

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(f"ETC predictions: {ETC_PATH}\n")
        f.write(f"Output CSV: {OUTPUT_CSV_PATH}\n")
        f.write(f"train_split: {TRAIN_SPLIT}\n")
        f.write(f"candidate_sources: {','.join(CANDIDATE_SOURCES)}\n")
        f.write(f"default_source: {DEFAULT_SOURCE}\n")
        f.write(f"n_configs: {len(build_policy_configs())}\n\n")
        f.write(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        f.write("\n")

    print("Done.")
    print(f"Saved guarded policy ablation CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved guarded policy ablation text to: {OUTPUT_TXT_PATH}")
    print(summary_df[summary_df["split"] == "test"].head(20).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
