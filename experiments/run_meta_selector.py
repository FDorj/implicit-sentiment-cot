from pathlib import Path
import math
import os
import sys

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import evaluate_predictions  # noqa: E402
from src.final_results import KEY_COLS  # noqa: E402


ETC_PATH = os.getenv(
    "META_SELECTOR_ETC_PATH",
    str(Path("results") / "etc_thor_originalish_sc3_isa_predictions.csv"),
)
THOR_SC_PATH = os.getenv(
    "META_SELECTOR_THOR_SC_PATH",
    str(Path("results") / "thor_originalish_sc3_isa_predictions.csv"),
)
CURRENT_SELECTED_PATH = os.getenv(
    "META_SELECTOR_CURRENT_SELECTED_PATH",
    str(Path("results") / "etc_thor_originalish_sc3_selected_isa_predictions.csv"),
)
OUTPUT_PREDICTIONS_PATH = os.getenv(
    "META_SELECTOR_OUTPUT_PATH",
    str(Path("results") / "meta_selector_predictions.csv"),
)
OUTPUT_METRICS_PATH = os.getenv(
    "META_SELECTOR_METRICS_PATH",
    str(Path("results") / "meta_selector_metrics.txt"),
)
OUTPUT_ABLATION_PATH = os.getenv(
    "META_SELECTOR_ABLATION_PATH",
    str(Path("results") / "meta_selector_ablation_metrics.csv"),
)

SOURCE_TO_COLUMN = {
    "direct": "direct_prediction",
    "thor": "thor_prediction",
    "diagnostic": "diagnostic_label",
}
SOURCE_ORDER = ["direct", "thor", "diagnostic"]
PROFILE_KEY_CONFIGS = [
    (
        "current_profile",
        ["direct_prediction", "error_type", "diagnostic_confidence", "domain"],
    ),
    (
        "rich_profile",
        [
            "direct_prediction",
            "thor_prediction",
            "error_type",
            "diagnostic_confidence",
            "domain",
        ],
    ),
]
REQUIRED_COLUMNS = [
    "polarity",
    "split",
    "domain",
    "direct_prediction",
    "thor_prediction",
    "diagnostic_label",
    "controller_prediction",
    "error_type",
    "diagnostic_confidence",
]
CATEGORICAL_FEATURES = [
    "direct_prediction",
    "thor_prediction",
    "diagnostic_label",
    "controller_prediction",
    "error_type",
    "diagnostic_confidence",
    "domain",
    "diagnostic_triggered",
    "direct_thor_agreement",
    "direct_diagnostic_agreement",
    "thor_diagnostic_agreement",
    "all_sources_agree",
]
NUMERIC_FEATURES = [
    "sc_top_vote_count",
    "sc_vote_margin",
    "sc_top_vote_share",
    "sc_vote_unique_labels",
]


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def parse_vote_counts(raw_value: object) -> dict[str, int]:
    text = str(raw_value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return {}

    counts: dict[str, int] = {}
    for part in text.split(";"):
        if ":" not in part:
            continue
        label, value = part.split(":", 1)
        label = label.strip()
        try:
            counts[label] = int(value.strip())
        except ValueError:
            continue

    return counts


def vote_features(raw_value: object) -> dict[str, float | int]:
    counts = parse_vote_counts(raw_value)
    if not counts:
        return {
            "sc_top_vote_count": 0,
            "sc_vote_margin": 0,
            "sc_top_vote_share": 0.0,
            "sc_vote_unique_labels": 0,
        }

    sorted_counts = sorted(counts.values(), reverse=True)
    top_count = sorted_counts[0]
    second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
    total = sum(sorted_counts)
    return {
        "sc_top_vote_count": top_count,
        "sc_vote_margin": top_count - second_count,
        "sc_top_vote_share": top_count / total if total else 0.0,
        "sc_vote_unique_labels": len(counts),
    }


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            features[column] = df[column].astype(str).fillna("unknown")
        else:
            features[column] = "unknown"

    features["diagnostic_triggered"] = df.get("diagnostic_triggered", False).map(
        lambda value: yes_no(str(value).strip().lower() in {"1", "true", "yes"})
    )
    features["direct_thor_agreement"] = (
        df["direct_prediction"] == df["thor_prediction"]
    ).map(yes_no)
    features["direct_diagnostic_agreement"] = (
        df["direct_prediction"] == df["diagnostic_label"]
    ).map(yes_no)
    features["thor_diagnostic_agreement"] = (
        df["thor_prediction"] == df["diagnostic_label"]
    ).map(yes_no)
    features["all_sources_agree"] = (
        (df["direct_prediction"] == df["thor_prediction"])
        & (df["direct_prediction"] == df["diagnostic_label"])
    ).map(yes_no)

    vote_series = df["sc_vote_counts"] if "sc_vote_counts" in df.columns else pd.Series("", index=df.index)
    vote_feature_rows = [vote_features(value) for value in vote_series]
    vote_feature_df = pd.DataFrame(vote_feature_rows, index=df.index)
    for column in NUMERIC_FEATURES:
        features[column] = vote_feature_df[column]

    return features[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


def profile_feature_columns(prefix: str) -> tuple[list[str], list[str]]:
    categorical_columns = [f"{prefix}_best_source"]
    numeric_columns = [
        f"{prefix}_support",
        f"{prefix}_best_correct_count",
        f"{prefix}_margin_vs_direct",
        f"{prefix}_margin_vs_second",
    ]
    for source in SOURCE_ORDER:
        numeric_columns.extend(
            [
                f"{prefix}_{source}_correct_count",
                f"{prefix}_{source}_correct_rate",
                f"{prefix}_{source}_margin_vs_direct",
            ]
        )

    return categorical_columns, numeric_columns


def build_profile_calibration_features(
    df: pd.DataFrame,
    train_indices: list[int],
    key_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    categorical_columns, numeric_columns = profile_feature_columns(prefix)
    output = pd.DataFrame(index=df.index)
    train_df = df.loc[train_indices]
    rows = []

    for key, group in train_df.groupby(key_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        scores = {
            source: int((group[SOURCE_TO_COLUMN[source]] == group["polarity"]).sum())
            for source in SOURCE_ORDER
        }
        sorted_scores = sorted(
            scores.items(),
            key=lambda item: (-item[1], SOURCE_ORDER.index(item[0])),
        )
        best_source, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

        row = dict(zip(key_columns, key))
        row[f"{prefix}_support"] = len(group)
        row[f"{prefix}_best_source"] = best_source
        row[f"{prefix}_best_correct_count"] = best_score
        row[f"{prefix}_margin_vs_direct"] = best_score - scores["direct"]
        row[f"{prefix}_margin_vs_second"] = best_score - second_score

        for source in SOURCE_ORDER:
            row[f"{prefix}_{source}_correct_count"] = scores[source]
            row[f"{prefix}_{source}_correct_rate"] = scores[source] / len(group)
            row[f"{prefix}_{source}_margin_vs_direct"] = scores[source] - scores["direct"]

        rows.append(row)

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        for column in categorical_columns:
            output[column] = "unknown"
        for column in numeric_columns:
            output[column] = 0
        return output[categorical_columns + numeric_columns]

    merged = df[key_columns].merge(stats_df, on=key_columns, how="left")
    merged.index = df.index
    for column in categorical_columns:
        output[column] = merged[column].fillna("unknown").astype(str)
    for column in numeric_columns:
        output[column] = merged[column].fillna(0)

    return output[categorical_columns + numeric_columns]


def build_calibrated_meta_features(
    df: pd.DataFrame,
    train_indices: list[int],
) -> pd.DataFrame:
    features = [build_meta_features(df)]
    for prefix, key_columns in PROFILE_KEY_CONFIGS:
        features.append(
            build_profile_calibration_features(
                df=df,
                train_indices=train_indices,
                key_columns=key_columns,
                prefix=prefix,
            )
        )

    return pd.concat(features, axis=1)


def infer_feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [
        column for column in features.columns if is_numeric_dtype(features[column])
    ]
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]
    return categorical_features, numeric_features


def oracle_select_sources(
    df: pd.DataFrame,
    source_to_column: dict[str, str] | None = None,
    default_source: str = "direct",
) -> tuple[list[str], list[str]]:
    selected_predictions = []
    selected_sources = []
    source_map = source_to_column or SOURCE_TO_COLUMN

    for _, row in df.iterrows():
        selected_source = default_source
        for source in SOURCE_ORDER:
            column = source_map[source]
            if row[column] == row["polarity"]:
                selected_source = source
                break

        selected_sources.append(selected_source)
        selected_predictions.append(row[source_map[selected_source]])

    return selected_predictions, selected_sources


def predictions_from_sources(df: pd.DataFrame, sources: list[str]) -> list[str]:
    return [
        row[SOURCE_TO_COLUMN[source]]
        for source, (_, row) in zip(sources, df.iterrows())
    ]


def metric_row(df: pd.DataFrame, predictions: list[str], method: str, split: str, note: str = "") -> dict:
    if split == "overall":
        split_df = df
        split_predictions = predictions
    else:
        split_mask = df["split"].astype(str).str.lower() == split
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
        "method": method,
        "split": split,
        "n_eval": metrics["n_eval"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "note": note,
    }


def make_model_configs(
    categorical_features: list[str],
    numeric_features: list[str],
) -> list[tuple[str, Pipeline]]:
    def make_pipeline(classifier) -> Pipeline:
        preprocessor = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("numeric", StandardScaler(), numeric_features),
            ]
        )
        return Pipeline(
            [
                ("preprocess", preprocessor),
                ("classifier", classifier),
            ]
        )

    return [
        (
            "logistic_balanced",
            make_pipeline(
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                )
            ),
        ),
        (
            "tree_depth3_leaf10",
            make_pipeline(
                DecisionTreeClassifier(
                    max_depth=3,
                    min_samples_leaf=10,
                    random_state=42,
                )
            ),
        ),
        (
            "tree_depth5_leaf10",
            make_pipeline(
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=10,
                    random_state=42,
                )
            ),
        ),
        (
            "tree_depth5_leaf20",
            make_pipeline(
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=20,
                    random_state=42,
                )
            ),
        ),
        (
            "gb_depth2_n100_lr0p05",
            make_pipeline(
                GradientBoostingClassifier(
                    max_depth=2,
                    n_estimators=100,
                    learning_rate=0.05,
                    random_state=42,
                )
            ),
        ),
    ]


def final_label_metrics_for_sources(df: pd.DataFrame, sources: list[str], indices: list[int]) -> tuple[float, float]:
    indexed_df = df.loc[indices]
    indexed_sources = list(pd.Series(sources, index=indices).loc[indices])
    predictions = predictions_from_sources(indexed_df, indexed_sources)
    return (
        accuracy_score(indexed_df["polarity"], predictions),
        f1_score(indexed_df["polarity"], predictions, average="macro"),
    )


def select_meta_model(
    df: pd.DataFrame,
    oracle_sources: list[str],
) -> tuple[str, Pipeline, pd.DataFrame]:
    train_mask = df["split"].astype(str).str.lower() == "train"
    train_indices = df[train_mask].index.to_list()
    y_train = pd.Series(oracle_sources, index=df.index).loc[train_indices]

    stratify = y_train if y_train.value_counts().min() >= 2 else None
    calibration_indices, validation_indices = train_test_split(
        train_indices,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    validation_features = build_calibrated_meta_features(df, calibration_indices)
    categorical_features, numeric_features = infer_feature_columns(validation_features)
    rows = []
    best_name = ""
    best_model: Pipeline | None = None
    best_macro_f1 = -math.inf
    for name, model in make_model_configs(categorical_features, numeric_features):
        model.fit(
            validation_features.loc[calibration_indices],
            y_train.loc[calibration_indices],
        )
        validation_sources = model.predict(
            validation_features.loc[validation_indices]
        ).tolist()
        validation_accuracy, validation_macro_f1 = final_label_metrics_for_sources(
            df=df,
            sources=validation_sources,
            indices=validation_indices,
        )
        rows.append(
            {
                "model": name,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
            }
        )
        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No meta-selector model was trained.")

    final_features = build_calibrated_meta_features(df, train_indices)
    final_categorical, final_numeric = infer_feature_columns(final_features)
    final_model = dict(make_model_configs(final_categorical, final_numeric))[best_name]
    final_model.fit(final_features.loc[train_indices], y_train)
    return best_name, final_model, pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "validation_accuracy"],
        ascending=False,
    )


def attach_sc_vote_columns(df: pd.DataFrame, thor_sc_path: str) -> pd.DataFrame:
    path = Path(thor_sc_path)
    if not path.exists() or "sc_vote_counts" in df.columns:
        return df

    thor_df = pd.read_csv(path)
    if not all(column in thor_df.columns for column in [*KEY_COLS, "sc_vote_counts"]):
        return df

    return df.merge(
        thor_df[[*KEY_COLS, "sc_vote_counts", "sc_labels"]],
        on=KEY_COLS,
        how="left",
    )


def attach_current_selected(df: pd.DataFrame, current_selected_path: str) -> pd.DataFrame:
    path = Path(current_selected_path)
    if not path.exists():
        return df

    selected_df = pd.read_csv(path)
    if not all(column in selected_df.columns for column in [*KEY_COLS, "selected_prediction"]):
        return df

    selected_columns = [*KEY_COLS, "selected_prediction"]
    rename_columns = {"selected_prediction": "current_selected_prediction"}
    if "selected_source" in selected_df.columns:
        selected_columns.append("selected_source")
        rename_columns["selected_source"] = "current_selected_source"

    return df.merge(
        selected_df[selected_columns],
        on=KEY_COLS,
        how="left",
    ).rename(columns=rename_columns)


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ETC_PATH)
    require_columns(df, REQUIRED_COLUMNS, ETC_PATH)
    df = attach_sc_vote_columns(df, THOR_SC_PATH)
    df = attach_current_selected(df, CURRENT_SELECTED_PATH)

    oracle_predictions, oracle_sources = oracle_select_sources(df)
    train_indices = df[df["split"].astype(str).str.lower() == "train"].index.to_list()
    features = build_calibrated_meta_features(df, train_indices)
    best_model_name, meta_model, validation_df = select_meta_model(df, oracle_sources)
    meta_sources = meta_model.predict(features).tolist()
    meta_predictions = predictions_from_sources(df, meta_sources)

    output_df = df.copy()
    output_df["oracle_source"] = oracle_sources
    output_df["oracle_prediction"] = oracle_predictions
    output_df["meta_selector_source"] = meta_sources
    output_df["meta_selector_prediction"] = meta_predictions
    output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)

    summary_rows = []
    comparison_methods = [
        ("direct_baseline", df["direct_prediction"].tolist(), ""),
        ("thor_baseline", df["thor_prediction"].tolist(), ""),
        ("diagnostic_label", df["diagnostic_label"].tolist(), ""),
        ("controller_prediction", df["controller_prediction"].tolist(), ""),
        ("oracle_source_upper_bound", oracle_predictions, "uses gold labels to choose source"),
        ("meta_selector", meta_predictions, f"selected validation model={best_model_name}"),
    ]
    if "current_selected_prediction" in df.columns:
        comparison_methods.insert(
            4,
            ("current_final_selected", df["current_selected_prediction"].tolist(), ""),
        )

    for method, predictions, note in comparison_methods:
        for split in ["overall", "train", "test"]:
            summary_rows.append(metric_row(df, predictions, method, split, note=note))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_ABLATION_PATH, index=False)

    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"ETC predictions: {ETC_PATH}\n")
        f.write(f"THOR SC predictions: {THOR_SC_PATH}\n")
        f.write(f"Current selected predictions: {CURRENT_SELECTED_PATH}\n")
        f.write(f"Output predictions: {OUTPUT_PREDICTIONS_PATH}\n")
        f.write(f"Output metrics CSV: {OUTPUT_ABLATION_PATH}\n")
        f.write(f"Selected meta model: {best_model_name}\n\n")
        f.write("Feature sets: base agreement/vote features + train-calibrated current/rich profile stats\n")
        f.write(f"Profile keys: {PROFILE_KEY_CONFIGS}\n\n")
        f.write("validation model selection\n")
        f.write(validation_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        f.write("\n\nsummary metrics\n")
        f.write(
            summary_df.sort_values(["split", "macro_f1", "accuracy"], ascending=[True, False, False])
            .to_string(index=False, float_format=lambda value: f"{value:.6f}")
        )
        test_summary = summary_df[summary_df["split"] == "test"].set_index("method")
        if {"oracle_source_upper_bound", "current_final_selected", "meta_selector", "direct_baseline"}.issubset(
            test_summary.index
        ):
            f.write("\n\ntest macro-F1 gaps\n")
            oracle_f1 = test_summary.loc["oracle_source_upper_bound", "macro_f1"]
            current_f1 = test_summary.loc["current_final_selected", "macro_f1"]
            meta_f1 = test_summary.loc["meta_selector", "macro_f1"]
            direct_f1 = test_summary.loc["direct_baseline", "macro_f1"]
            f.write(f"oracle_minus_current_final: {oracle_f1 - current_f1:.6f}\n")
            f.write(f"oracle_minus_meta_selector: {oracle_f1 - meta_f1:.6f}\n")
            f.write(f"meta_selector_minus_direct: {meta_f1 - direct_f1:.6f}\n")
            f.write(f"current_final_minus_meta_selector: {current_f1 - meta_f1:.6f}\n")
        f.write("\n\noracle source counts\n")
        f.write(output_df["oracle_source"].value_counts().to_string())
        f.write("\n\nmeta-selector source counts\n")
        f.write(output_df["meta_selector_source"].value_counts().to_string())
        f.write("\n")

    print("Done.")
    print(f"Saved meta-selector predictions to: {OUTPUT_PREDICTIONS_PATH}")
    print(f"Saved meta-selector metrics to: {OUTPUT_METRICS_PATH}")
    print(f"Selected meta model: {best_model_name}")
    print(
        summary_df[summary_df["split"] == "test"]
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()
