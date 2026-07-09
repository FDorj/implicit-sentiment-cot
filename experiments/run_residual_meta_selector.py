from pathlib import Path
import math
import os
import re
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

from experiments.run_meta_selector import (  # noqa: E402
    SOURCE_ORDER,
    SOURCE_TO_COLUMN,
    build_calibrated_meta_features,
)
from src.evaluator import evaluate_predictions  # noqa: E402


INPUT_PATH = os.getenv(
    "RESIDUAL_META_SELECTOR_INPUT_PATH",
    str(Path("results") / "meta_selector_predictions.csv"),
)
OUTPUT_PREDICTIONS_PATH = os.getenv(
    "RESIDUAL_META_SELECTOR_OUTPUT_PATH",
    str(Path("results") / "residual_meta_selector_predictions.csv"),
)
OUTPUT_METRICS_PATH = os.getenv(
    "RESIDUAL_META_SELECTOR_METRICS_PATH",
    str(Path("results") / "residual_meta_selector_metrics.txt"),
)
OUTPUT_ABLATION_PATH = os.getenv(
    "RESIDUAL_META_SELECTOR_ABLATION_PATH",
    str(Path("results") / "residual_meta_selector_ablation_metrics.csv"),
)

REQUIRED_COLUMNS = [
    "polarity",
    "split",
    "direct_prediction",
    "thor_prediction",
    "diagnostic_label",
    "current_selected_source",
    "current_selected_prediction",
]
THRESHOLDS = [value / 100 for value in range(0, 51)]
NEGATION_CUES = {
    "no",
    "not",
    "never",
    "none",
    "neither",
    "nor",
    "without",
    "cannot",
    "cant",
    "can't",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "didnt",
    "didn't",
    "wont",
    "won't",
    "isnt",
    "isn't",
    "wasnt",
    "wasn't",
    "arent",
    "aren't",
    "werent",
    "weren't",
    "couldnt",
    "couldn't",
    "wouldnt",
    "wouldn't",
    "shouldnt",
    "shouldn't",
}
CONTRAST_CUES = {
    "but",
    "however",
    "though",
    "although",
    "yet",
    "despite",
    "while",
    "whereas",
}


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def infer_feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [
        column for column in features.columns if is_numeric_dtype(features[column])
    ]
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]
    return categorical_features, numeric_features


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def safe_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def target_start_index(row: pd.Series, sentence: str) -> int | None:
    from_value = safe_int(row.get("from"))
    if from_value is not None and from_value >= 0:
        return from_value

    target = str(row.get("target", "") or "").strip().lower()
    if not target:
        return None
    index = sentence.lower().find(target)
    return index if index >= 0 else None


def position_bucket(position_ratio: float | None) -> str:
    if position_ratio is None:
        return "unknown"
    if position_ratio < 0.33:
        return "early"
    if position_ratio < 0.66:
        return "middle"
    return "late"


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        sentence = str(row.get("sentence", "") or "")
        tokens = tokenize(sentence)
        start_index = target_start_index(row, sentence)
        ratio = start_index / len(sentence) if sentence and start_index is not None else None
        rows.append(
            {
                "sentence_char_length": len(sentence),
                "sentence_word_count": len(tokens),
                "target_position_ratio": ratio if ratio is not None else 0.0,
                "target_position_bucket": position_bucket(ratio),
                "has_negation_cue": yes_no(any(token in NEGATION_CUES for token in tokens)),
                "has_contrast_cue": yes_no(any(token in CONTRAST_CUES for token in tokens)),
                "has_exclamation": yes_no("!" in sentence),
                "has_question": yes_no("?" in sentence),
                "negation_count": sum(1 for token in tokens if token in NEGATION_CUES),
                "contrast_count": sum(1 for token in tokens if token in CONTRAST_CUES),
            }
        )

    return pd.DataFrame(rows, index=df.index)


def build_residual_row_features(
    df: pd.DataFrame,
    train_indices: list[int],
) -> pd.DataFrame:
    return pd.concat(
        [
            build_calibrated_meta_features(df, train_indices),
            build_text_features(df),
        ],
        axis=1,
    )


def build_candidate_rows(
    df: pd.DataFrame,
    row_features: pd.DataFrame,
    indices: list[int],
) -> pd.DataFrame:
    parts = []
    rows = df.loc[indices]

    for source in SOURCE_ORDER:
        candidate_rows = row_features.loc[indices].copy()
        candidate_predictions = rows[SOURCE_TO_COLUMN[source]].astype(str)

        candidate_rows["candidate_source"] = source
        candidate_rows["candidate_prediction"] = candidate_predictions.values
        candidate_rows["current_selected_source"] = rows["current_selected_source"].astype(str).values
        candidate_rows["current_selected_prediction"] = rows["current_selected_prediction"].astype(str).values
        candidate_rows["candidate_is_current"] = [
            yes_no(candidate_source == current_source)
            for candidate_source, current_source in zip(
                candidate_rows["candidate_source"],
                candidate_rows["current_selected_source"],
            )
        ]
        candidate_rows["candidate_agrees_current"] = [
            yes_no(candidate_prediction == current_prediction)
            for candidate_prediction, current_prediction in zip(
                candidate_rows["candidate_prediction"],
                candidate_rows["current_selected_prediction"],
            )
        ]
        candidate_rows["candidate_agrees_direct"] = [
            yes_no(candidate_prediction == direct_prediction)
            for candidate_prediction, direct_prediction in zip(
                candidate_rows["candidate_prediction"],
                rows["direct_prediction"].astype(str),
            )
        ]
        candidate_rows["candidate_agrees_thor"] = [
            yes_no(candidate_prediction == thor_prediction)
            for candidate_prediction, thor_prediction in zip(
                candidate_rows["candidate_prediction"],
                rows["thor_prediction"].astype(str),
            )
        ]
        candidate_rows["candidate_agrees_diagnostic"] = [
            yes_no(candidate_prediction == diagnostic_prediction)
            for candidate_prediction, diagnostic_prediction in zip(
                candidate_rows["candidate_prediction"],
                rows["diagnostic_label"].astype(str),
            )
        ]
        candidate_rows["current_to_candidate_label"] = [
            f"{current_prediction}->{candidate_prediction}"
            for current_prediction, candidate_prediction in zip(
                candidate_rows["current_selected_prediction"],
                candidate_rows["candidate_prediction"],
            )
        ]
        candidate_rows["direct_to_thor_label"] = [
            f"{direct_prediction}->{thor_prediction}"
            for direct_prediction, thor_prediction in zip(
                rows["direct_prediction"].astype(str),
                rows["thor_prediction"].astype(str),
            )
        ]
        candidate_rows["direct_to_diagnostic_label"] = [
            f"{direct_prediction}->{diagnostic_prediction}"
            for direct_prediction, diagnostic_prediction in zip(
                rows["direct_prediction"].astype(str),
                rows["diagnostic_label"].astype(str),
            )
        ]
        candidate_rows["thor_to_diagnostic_label"] = [
            f"{thor_prediction}->{diagnostic_prediction}"
            for thor_prediction, diagnostic_prediction in zip(
                rows["thor_prediction"].astype(str),
                rows["diagnostic_label"].astype(str),
            )
        ]
        candidate_rows["source_index"] = SOURCE_ORDER.index(source)
        candidate_rows["row_index"] = indices
        candidate_rows["candidate_correct"] = (
            rows[SOURCE_TO_COLUMN[source]].values == rows["polarity"].values
        )
        parts.append(candidate_rows)

    return pd.concat(parts, ignore_index=True)


def select_residual_predictions(
    df: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    selected_predictions = {}
    selected_sources = {}

    for row_index, group in candidate_scores.groupby("row_index"):
        row = df.loc[row_index]
        current_source = row["current_selected_source"]
        current_prediction = row["current_selected_prediction"]
        current_rows = group[group["candidate_source"] == current_source]
        if current_rows.empty:
            selected_predictions[row_index] = current_prediction
            selected_sources[row_index] = current_source
            continue

        current_score = float(current_rows["score"].iloc[0])
        best = group.sort_values("score", ascending=False).iloc[0]
        score_margin = float(best["score"]) - current_score
        if best["candidate_source"] != current_source and score_margin >= threshold:
            selected_predictions[row_index] = best["candidate_prediction"]
            selected_sources[row_index] = best["candidate_source"]
        else:
            selected_predictions[row_index] = current_prediction
            selected_sources[row_index] = current_source

    return pd.Series(selected_predictions), pd.Series(selected_sources)


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
                    max_iter=3000,
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
                    class_weight="balanced",
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
                    class_weight="balanced",
                    random_state=42,
                )
            ),
        ),
        (
            "gb_depth2_n80_lr0p05",
            make_pipeline(
                GradientBoostingClassifier(
                    max_depth=2,
                    n_estimators=80,
                    learning_rate=0.05,
                    random_state=42,
                )
            ),
        ),
    ]


def score_candidates(model: Pipeline, candidate_rows: pd.DataFrame) -> pd.DataFrame:
    model_input = candidate_rows.drop(columns=["candidate_correct", "row_index"])
    classifier = model.named_steps["classifier"]
    true_class_index = list(classifier.classes_).index(True)
    scores = model.predict_proba(model_input)[:, true_class_index]
    output = candidate_rows[
        ["row_index", "candidate_source", "candidate_prediction"]
    ].copy()
    output["score"] = scores
    return output


def evaluate_predictions_for_indices(
    df: pd.DataFrame,
    predictions: pd.Series,
    indices: list[int],
) -> tuple[float, float]:
    indexed_df = df.loc[indices]
    indexed_predictions = predictions.loc[indices]
    return (
        accuracy_score(indexed_df["polarity"], indexed_predictions),
        f1_score(indexed_df["polarity"], indexed_predictions, average="macro"),
    )


def residual_threshold_metrics(
    df: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    indices: list[int],
    threshold: float,
) -> dict:
    predictions, sources = select_residual_predictions(
        df=df,
        candidate_scores=candidate_scores,
        threshold=threshold,
    )
    indexed_df = df.loc[indices]
    indexed_predictions = predictions.loc[indices]
    current_predictions = indexed_df["current_selected_prediction"]
    accuracy, macro_f1 = evaluate_predictions_for_indices(df, predictions, indices)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "overrides": int((indexed_predictions != current_predictions).sum()),
        "gains": int(
            (
                (current_predictions != indexed_df["polarity"])
                & (indexed_predictions == indexed_df["polarity"])
            ).sum()
        ),
        "losses": int(
            (
                (current_predictions == indexed_df["polarity"])
                & (indexed_predictions != indexed_df["polarity"])
            ).sum()
        ),
        "selected_sources": sources,
        "predictions": predictions,
    }


def residual_candidate_allowed(
    metrics: dict,
    baseline_macro_f1: float,
    min_macro_f1_gain: float = 0.0,
    max_validation_losses: int = 0,
    min_validation_gains: int = 2,
) -> bool:
    return (
        metrics["macro_f1"] > baseline_macro_f1 + min_macro_f1_gain
        and metrics["losses"] <= max_validation_losses
        and metrics["gains"] >= min_validation_gains
    )


def select_residual_model(
    df: pd.DataFrame,
) -> tuple[str, float, Pipeline, pd.DataFrame, pd.DataFrame]:
    train_indices = df[df["split"].astype(str).str.lower() == "train"].index.to_list()
    current_wrong = (
        df.loc[train_indices, "current_selected_prediction"]
        != df.loc[train_indices, "polarity"]
    )
    stratify = current_wrong if current_wrong.value_counts().min() >= 2 else None
    calibration_indices, validation_indices = train_test_split(
        train_indices,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    validation_features = build_residual_row_features(df, calibration_indices)
    calibration_candidates = build_candidate_rows(
        df, validation_features, calibration_indices
    )
    validation_candidates = build_candidate_rows(df, validation_features, validation_indices)
    x_calibration = calibration_candidates.drop(columns=["candidate_correct", "row_index"])
    y_calibration = calibration_candidates["candidate_correct"]
    categorical_features, numeric_features = infer_feature_columns(x_calibration)
    model_configs = make_model_configs(categorical_features, numeric_features)
    baseline_predictions = df["current_selected_prediction"]
    baseline_accuracy, baseline_macro_f1 = evaluate_predictions_for_indices(
        df=df,
        predictions=baseline_predictions,
        indices=validation_indices,
    )

    rows = [
        {
            "model": "current_final_noop",
            "threshold": math.inf,
            "validation_accuracy": baseline_accuracy,
            "validation_macro_f1": baseline_macro_f1,
            "validation_overrides": 0,
            "validation_gains": 0,
            "validation_losses": 0,
            "guard_allowed": True,
        }
    ]
    best_model_name = "current_final_noop"
    best_threshold = math.inf
    best_score = (baseline_macro_f1, baseline_accuracy, 0, 0, 0)
    for model_name, model in model_configs:
        model.fit(x_calibration, y_calibration)
        validation_scores = score_candidates(model, validation_candidates)
        for threshold in THRESHOLDS:
            metrics = residual_threshold_metrics(
                df=df,
                candidate_scores=validation_scores,
                indices=validation_indices,
                threshold=threshold,
            )
            guard_allowed = residual_candidate_allowed(metrics, baseline_macro_f1)
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "validation_accuracy": metrics["accuracy"],
                    "validation_macro_f1": metrics["macro_f1"],
                    "validation_overrides": metrics["overrides"],
                    "validation_gains": metrics["gains"],
                    "validation_losses": metrics["losses"],
                    "guard_allowed": guard_allowed,
                }
            )
            candidate_score = (
                metrics["macro_f1"],
                metrics["accuracy"],
                -metrics["losses"],
                metrics["gains"],
                -metrics["overrides"],
            )
            if guard_allowed and candidate_score > best_score:
                best_score = candidate_score
                best_model_name = model_name
                best_threshold = threshold

    final_features = build_residual_row_features(df, train_indices)
    train_candidates = build_candidate_rows(df, final_features, train_indices)
    x_train = train_candidates.drop(columns=["candidate_correct", "row_index"])
    y_train = train_candidates["candidate_correct"]
    final_categorical, final_numeric = infer_feature_columns(x_train)
    final_model_name = (
        best_model_name
        if best_model_name != "current_final_noop"
        else make_model_configs(final_categorical, final_numeric)[0][0]
    )
    final_model = dict(make_model_configs(final_categorical, final_numeric))[final_model_name]
    final_model.fit(x_train, y_train)
    all_candidates = build_candidate_rows(df, final_features, df.index.to_list())
    all_scores = score_candidates(final_model, all_candidates)

    return (
        best_model_name,
        best_threshold,
        final_model,
        pd.DataFrame(rows).sort_values(
            ["guard_allowed", "validation_macro_f1", "validation_accuracy", "validation_losses"],
            ascending=[False, False, False, True],
        ),
        all_scores,
    )


def metric_row(
    df: pd.DataFrame,
    predictions: list[str] | pd.Series,
    method: str,
    split: str,
    note: str = "",
) -> dict:
    if split == "overall":
        split_df = df
        split_predictions = list(predictions)
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


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    require_columns(df, REQUIRED_COLUMNS, INPUT_PATH)
    best_model_name, best_threshold, _, validation_df, all_scores = select_residual_model(df)
    residual_predictions, residual_sources = select_residual_predictions(
        df=df,
        candidate_scores=all_scores,
        threshold=best_threshold,
    )

    output_df = df.copy()
    output_df["residual_selected_prediction"] = residual_predictions.loc[df.index].values
    output_df["residual_selected_source"] = residual_sources.loc[df.index].values
    output_df["residual_changed_prediction"] = (
        output_df["residual_selected_prediction"]
        != output_df["current_selected_prediction"]
    )
    output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)

    summary_rows = []
    comparison_methods = [
        ("current_final_selected", df["current_selected_prediction"].tolist(), ""),
        (
            "residual_meta_selector",
            output_df["residual_selected_prediction"].tolist(),
            f"candidate scorer={best_model_name}; threshold={best_threshold:.2f}",
        ),
    ]
    if "meta_selector_prediction" in df.columns:
        comparison_methods.append(
            ("source_classifier_meta_selector", df["meta_selector_prediction"].tolist(), "")
        )
    if "oracle_prediction" in df.columns:
        comparison_methods.append(
            ("oracle_source_upper_bound", df["oracle_prediction"].tolist(), "uses gold labels")
        )
    comparison_methods.append(("direct_baseline", df["direct_prediction"].tolist(), ""))

    for method, predictions, note in comparison_methods:
        for split in ["overall", "train", "test"]:
            summary_rows.append(metric_row(df, predictions, method, split, note=note))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_ABLATION_PATH, index=False)

    test_df = output_df[output_df["split"].astype(str).str.lower() == "test"]
    test_overrides = int(test_df["residual_changed_prediction"].sum())
    test_gains = int(
        (
            (test_df["current_selected_prediction"] != test_df["polarity"])
            & (test_df["residual_selected_prediction"] == test_df["polarity"])
        ).sum()
    )
    test_losses = int(
        (
            (test_df["current_selected_prediction"] == test_df["polarity"])
            & (test_df["residual_selected_prediction"] != test_df["polarity"])
        ).sum()
    )

    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Input predictions: {INPUT_PATH}\n")
        f.write(f"Output predictions: {OUTPUT_PREDICTIONS_PATH}\n")
        f.write(f"Output metrics CSV: {OUTPUT_ABLATION_PATH}\n")
        f.write(f"Selected residual model: {best_model_name}\n")
        f.write(f"Selected validation threshold: {best_threshold:.2f}\n")
        f.write("Selection rule: keep current final unless best candidate score exceeds current score by threshold.\n\n")
        f.write("validation model/threshold selection top rows\n")
        f.write(
            validation_df.head(20).to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )
        f.write("\n\nsummary metrics\n")
        f.write(
            summary_df.sort_values(["split", "macro_f1", "accuracy"], ascending=[True, False, False])
            .to_string(index=False, float_format=lambda value: f"{value:.6f}")
        )
        f.write("\n\ntest residual changes\n")
        f.write(f"test_overrides: {test_overrides}\n")
        f.write(f"test_gains_vs_current: {test_gains}\n")
        f.write(f"test_losses_vs_current: {test_losses}\n")
        f.write("\n")

    print("Done.")
    print(f"Saved residual predictions to: {OUTPUT_PREDICTIONS_PATH}")
    print(f"Saved residual metrics to: {OUTPUT_METRICS_PATH}")
    print(f"Selected residual model: {best_model_name}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(
        summary_df[summary_df["split"] == "test"]
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"test_overrides={test_overrides}, test_gains={test_gains}, test_losses={test_losses}")


if __name__ == "__main__":
    main()
