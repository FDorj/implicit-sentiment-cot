from pathlib import Path
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablate_guarded_etc_policy import (  # noqa: E402
    CURRENT_KEY_COLUMNS,
    RICHER_KEY_COLUMNS,
    apply_learned_policy,
    format_gain,
    run_policy_config,
)
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
TUNING_CSV_PATH = os.getenv(
    "VALIDATION_TUNED_ETC_POLICY_CSV",
    str(Path("results") / "validation_tuned_guarded_etc_policy_ablation.csv"),
)
OUTPUT_PATH = os.getenv(
    "VALIDATION_TUNED_ETC_SELECTED_OUTPUT_PATH",
    str(Path("results") / "validation_tuned_guarded_etc_selected_predictions.csv"),
)
METRICS_PATH = os.getenv(
    "VALIDATION_TUNED_ETC_SELECTED_METRICS_PATH",
    str(Path("results") / "validation_tuned_guarded_etc_selected_metrics.txt"),
)

TRAIN_SPLIT = os.getenv("ETC_SELECTED_TRAIN_SPLIT", "train").strip().lower()
CANDIDATE_SOURCES = parse_csv_list(os.getenv("ETC_SELECTED_CANDIDATE_SOURCES", "direct,thor,diagnostic"))
DEFAULT_SOURCE = os.getenv("ETC_SELECTED_DEFAULT_SOURCE", "direct").strip().lower()
VALIDATION_FRACTION = float(os.getenv("ETC_TUNING_VALIDATION_FRACTION", "0.333333"))
VALIDATION_SEED = int(os.getenv("ETC_TUNING_VALIDATION_SEED", "42"))
VALIDATION_SEEDS = [
    int(value)
    for value in parse_csv_list(os.getenv("ETC_TUNING_VALIDATION_SEEDS", "0,1,2,3,4,5,6,7,8,9"))
]

MIN_SUPPORT_CANDIDATES = [
    int(value)
    for value in parse_csv_list(os.getenv("ETC_TUNING_MIN_SUPPORTS", "2,3,5,10,20"))
]
MIN_MARGIN_DEFAULT_CANDIDATES = [
    int(value)
    for value in parse_csv_list(os.getenv("ETC_TUNING_MIN_MARGIN_DEFAULTS", "1,2,3"))
]
MIN_MARGIN_SECOND_CANDIDATES = [
    int(value)
    for value in parse_csv_list(os.getenv("ETC_TUNING_MIN_MARGIN_SECONDS", "0,1,2"))
]
MIN_RELATIVE_GAIN_CANDIDATES = [
    float(value)
    for value in parse_csv_list(os.getenv("ETC_TUNING_MIN_RELATIVE_GAINS", "0.0,0.03,0.05"))
]

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


def build_validation_policy_configs() -> list[dict]:
    configs = [
        {
            "name": "current_unguarded",
            "mode": "unguarded",
            "key_columns": CURRENT_KEY_COLUMNS,
            "key_variant": "current",
        },
        {
            "name": "rich_unguarded",
            "mode": "unguarded",
            "key_columns": RICHER_KEY_COLUMNS,
            "key_variant": "rich",
        },
    ]

    for key_variant, key_columns in [("current", CURRENT_KEY_COLUMNS), ("rich", RICHER_KEY_COLUMNS)]:
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


def assign_tuning_splits(
    df: pd.DataFrame,
    train_split: str,
    validation_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    tuned_df = df.copy()
    tuned_df["original_split"] = tuned_df["split"]
    tuned_df["tuning_split"] = tuned_df["split"].astype(str).str.lower()

    train_mask = tuned_df["tuning_split"] == train_split
    train_df = tuned_df[train_mask]
    if train_df.empty:
        raise ValueError(f"No rows found for train_split={train_split!r}")

    stratify_columns = [column for column in ["domain", "polarity"] if column in train_df.columns]
    if stratify_columns:
        validation_indices = []
        for _, group in train_df.groupby(stratify_columns, dropna=False):
            n_validation = max(1, int(round(len(group) * validation_fraction)))
            validation_indices.extend(
                group.sample(n=n_validation, random_state=random_state).index.tolist()
            )
    else:
        n_validation = max(1, int(round(len(train_df) * validation_fraction)))
        validation_indices = train_df.sample(n=n_validation, random_state=random_state).index.tolist()

    tuned_df.loc[train_mask, "tuning_split"] = "calibration"
    tuned_df.loc[validation_indices, "tuning_split"] = "validation"
    tuned_df["split"] = tuned_df["tuning_split"]
    return tuned_df


def summarize_split(
    df: pd.DataFrame,
    predictions: list[str],
    sources: list[str],
    config: dict,
    split: str,
) -> dict:
    prediction_series = pd.Series(predictions, index=df.index)
    source_series = pd.Series(sources, index=df.index)
    split_mask = df["split"].astype(str).str.lower() == split
    split_df = df[split_mask]
    split_predictions = prediction_series.loc[split_df.index]
    split_sources = source_series.loc[split_df.index]
    metrics = evaluate_predictions(
        pd.DataFrame({"polarity": split_df["polarity"], "prediction": split_predictions}),
        gold_col="polarity",
        pred_col="prediction",
    )
    source_counts = split_sources.value_counts().to_dict()

    return {
        "policy": config["name"],
        "mode": config["mode"],
        "key_variant": config["key_variant"],
        "split": split,
        "validation_seed": config.get("validation_seed", ""),
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


def select_best_validation_config(summary_df: pd.DataFrame) -> dict:
    validation_rows = summary_df[summary_df["split"] == "validation"].copy()
    if validation_rows.empty:
        raise ValueError("No validation rows found for policy selection")
    if "n_eval" not in validation_rows.columns:
        validation_rows["n_eval"] = 1
    if "mode" not in validation_rows.columns:
        validation_rows["mode"] = ""
    for column in ["min_support", "min_margin_default", "min_margin_second", "min_relative_gain"]:
        if column not in validation_rows.columns:
            validation_rows[column] = pd.NA
        validation_rows[column] = pd.to_numeric(validation_rows[column], errors="coerce")

    validation_rows = (
        validation_rows.groupby("policy", as_index=False)
        .agg(
            {
                "macro_f1": "mean",
                "accuracy": "mean",
                "n_eval": "sum",
                "mode": "first",
                "min_support": "min",
                "min_margin_default": "min",
                "min_margin_second": "min",
                "min_relative_gain": "min",
            }
        )
    )
    validation_rows["mode_rank"] = validation_rows["mode"].map({"guarded": 0}).fillna(1)
    validation_rows["min_support_rank"] = validation_rows["min_support"].fillna(999999)
    validation_rows["min_margin_default_rank"] = validation_rows["min_margin_default"].fillna(999999)
    validation_rows["min_margin_second_rank"] = validation_rows["min_margin_second"].fillna(999999)
    validation_rows["min_relative_gain_rank"] = validation_rows["min_relative_gain"].fillna(999999.0)
    validation_rows = validation_rows.sort_values(
        [
            "macro_f1",
            "accuracy",
            "mode_rank",
            "min_support_rank",
            "min_margin_default_rank",
            "min_margin_second_rank",
            "min_relative_gain_rank",
            "policy",
        ],
        ascending=[False, False, True, True, True, True, True, True],
    )
    return validation_rows.iloc[0].to_dict()


def learn_policy_from_full_train(
    df: pd.DataFrame,
    config: dict,
    candidate_sources: list[str],
    default_source: str,
    train_split: str,
) -> tuple[dict[tuple, str], dict[tuple, dict]]:
    key_columns = config["key_columns"]
    if config["mode"] == "unguarded":
        policy = learn_calibrated_policy(
            df=df,
            key_columns=key_columns,
            candidate_sources=candidate_sources,
            default_source=default_source,
            train_split=train_split,
        )
        return policy, {}

    return learn_guarded_calibrated_policy(
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


def write_final_metrics(
    path: str,
    df: pd.DataFrame,
    selected_config: dict,
    tuning_summary: pd.DataFrame,
) -> None:
    metrics = evaluate_predictions(df, gold_col="polarity", pred_col="selected_prediction")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ETC predictions: {ETC_PATH}\n")
        f.write(f"Tuning CSV: {TUNING_CSV_PATH}\n")
        f.write(f"Output: {OUTPUT_PATH}\n")
        f.write("selection_basis: validation_macro_f1_then_accuracy\n")
        f.write(f"validation_fraction: {VALIDATION_FRACTION:.6f}\n")
        f.write(f"validation_seeds: {','.join(str(seed) for seed in VALIDATION_SEEDS)}\n")
        f.write(f"selected_policy: {selected_config['policy']}\n")
        f.write(f"selected_mean_validation_accuracy: {selected_config['accuracy']:.6f}\n")
        f.write(f"selected_mean_validation_macro_f1: {selected_config['macro_f1']:.6f}\n")
        f.write(f"selected_validation_n_eval_sum: {int(selected_config['n_eval'])}\n")
        f.write(f"n_tuned_configs: {tuning_summary['policy'].nunique()}\n")
        f.write(f"n_total: {metrics['n_total']}\n")
        f.write(f"n_eval: {metrics['n_eval']}\n")
        f.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {metrics['macro_f1']:.6f}\n\n")
        f.write(metrics["report"])
        f.write("\n")

        for split in ["train", "test"]:
            split_df = df[df["split"].astype(str).str.lower() == split]
            split_metrics = evaluate_predictions(split_df, gold_col="polarity", pred_col="selected_prediction")
            f.write(f"\n{split} metrics\n")
            f.write(f"n_total: {split_metrics['n_total']}\n")
            f.write(f"n_eval: {split_metrics['n_eval']}\n")
            f.write(f"accuracy: {split_metrics['accuracy']:.6f}\n")
            f.write(f"macro_f1: {split_metrics['macro_f1']:.6f}\n")

        f.write("\nselected source counts\n")
        f.write(df["selected_source"].value_counts().to_string())
        f.write("\n")


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    validate_sources(CANDIDATE_SOURCES, DEFAULT_SOURCE)

    original_df = pd.read_csv(ETC_PATH)
    require_columns(original_df, REQUIRED_COLUMNS, ETC_PATH)

    tuning_rows = []
    configs = build_validation_policy_configs()
    for validation_seed in VALIDATION_SEEDS:
        tuning_df = assign_tuning_splits(
            df=original_df,
            train_split=TRAIN_SPLIT,
            validation_fraction=VALIDATION_FRACTION,
            random_state=validation_seed,
        )
        for config in configs:
            config_with_seed = {**config, "validation_seed": validation_seed}
            require_columns(tuning_df, config["key_columns"], ETC_PATH)
            predictions, _, sources, _, _, _ = run_policy_config(
                df=tuning_df,
                config=config,
                candidate_sources=CANDIDATE_SOURCES,
                default_source=DEFAULT_SOURCE,
                train_split="calibration",
            )
            for split in ["calibration", "validation"]:
                tuning_rows.append(summarize_split(tuning_df, predictions, sources, config_with_seed, split))

    tuning_summary = pd.DataFrame(tuning_rows)
    tuning_summary = tuning_summary.sort_values(
        ["split", "macro_f1", "accuracy", "policy"],
        ascending=[True, False, False, True],
    )
    tuning_summary.to_csv(TUNING_CSV_PATH, index=False)

    selected_validation = select_best_validation_config(tuning_summary)
    selected_config = next(config for config in configs if config["name"] == selected_validation["policy"])

    final_df = original_df.copy()
    policy, policy_metadata = learn_policy_from_full_train(
        df=final_df,
        config=selected_config,
        candidate_sources=CANDIDATE_SOURCES,
        default_source=DEFAULT_SOURCE,
        train_split=TRAIN_SPLIT,
    )
    predictions, decisions, sources, policy_keys = apply_learned_policy(
        df=final_df,
        key_columns=selected_config["key_columns"],
        policy=policy,
        default_source=DEFAULT_SOURCE,
        policy_metadata=policy_metadata if selected_config["mode"] == "guarded" else None,
    )
    final_df["selected_prediction"] = predictions
    final_df["selected_controller_decision"] = decisions
    final_df["selected_source"] = sources
    final_df["selected_policy_key"] = policy_keys
    final_df.to_csv(OUTPUT_PATH, index=False)

    write_final_metrics(METRICS_PATH, final_df, selected_validation, tuning_summary)

    metrics = evaluate_predictions(final_df, gold_col="polarity", pred_col="selected_prediction")
    test_metrics = evaluate_predictions(
        final_df[final_df["split"].astype(str).str.lower() == "test"],
        gold_col="polarity",
        pred_col="selected_prediction",
    )
    print("Done.")
    print(f"Saved tuning summary to: {TUNING_CSV_PATH}")
    print(f"Saved selected predictions to: {OUTPUT_PATH}")
    print(f"Saved selected metrics to: {METRICS_PATH}")
    print(f"Selected policy: {selected_validation['policy']}")
    print(f"Validation Macro-F1: {selected_validation['macro_f1']:.6f}")
    print(f"Overall Macro-F1: {metrics['macro_f1']:.6f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
