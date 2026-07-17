from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import VALID_LABELS  # noqa: E402
from src.experiment_config import DEFAULT_DATA_PATH  # noqa: E402


SEED = 42
CANDIDATE_C_VALUES = (0.1, 1.0, 10.0)
REQUIRED_COLUMNS = ("split", "sentence", "target", "polarity")
DEFAULT_PREDICTIONS_PATH = Path("results/tfidf_logreg_predictions.csv")
DEFAULT_CV_PATH = Path("results/tfidf_logreg_cv_results.csv")
DEFAULT_METRICS_PATH = Path("results/tfidf_logreg_metrics.txt")


def validate_dataset(df: pd.DataFrame) -> None:
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_counts = df.loc[:, REQUIRED_COLUMNS].isna().sum()
    invalid_counts = {
        name: int(count)
        for name, count in missing_counts.items()
        if int(count) > 0
    }
    if invalid_counts:
        raise ValueError(f"Missing values in required columns: {invalid_counts}")

    splits = set(df["split"].astype(str).str.strip().str.lower())
    if splits != {"train", "test"}:
        raise ValueError(
            f"Expected exactly train/test splits, found: {sorted(splits)}"
        )

    labels = set(df["polarity"].astype(str).str.strip().str.lower())
    unsupported = sorted(labels - set(VALID_LABELS))
    if unsupported:
        raise ValueError(f"Unsupported labels: {unsupported}")


def normalize_text(value: object) -> str:
    return " ".join(str(value).split())


def build_model_text(df: pd.DataFrame) -> pd.Series:
    targets = df["target"].map(normalize_text)
    sentences = df["sentence"].map(normalize_text)
    return targets + " [SEP] " + sentences


def make_pipeline(c_value: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def select_best_candidate(results: pd.DataFrame) -> float:
    ranked = results.sort_values(
        ["mean_macro_f1", "mean_accuracy", "c"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["c"])


def cross_validate_candidates(
    texts: pd.Series,
    labels: pd.Series,
    c_values: tuple[float, ...] = CANDIDATE_C_VALUES,
) -> pd.DataFrame:
    normalized_texts = texts.reset_index(drop=True)
    normalized_labels = labels.astype(str).str.strip().str.lower().reset_index(drop=True)
    class_counts = normalized_labels.value_counts()
    insufficient = {
        label: int(class_counts.get(label, 0))
        for label in VALID_LABELS
        if int(class_counts.get(label, 0)) < 5
    }
    if insufficient:
        raise ValueError(
            "Five-fold stratified CV requires at least five training rows per "
            f"class; found: {insufficient}"
        )

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    rows: list[dict[str, object]] = []
    for c_value in c_values:
        fold_macro_f1: list[float] = []
        fold_accuracy: list[float] = []
        for train_indices, validation_indices in splitter.split(
            normalized_texts,
            normalized_labels,
        ):
            model = make_pipeline(float(c_value))
            model.fit(
                normalized_texts.iloc[train_indices],
                normalized_labels.iloc[train_indices],
            )
            predictions = model.predict(normalized_texts.iloc[validation_indices])
            gold = normalized_labels.iloc[validation_indices]
            fold_macro_f1.append(
                float(
                    f1_score(
                        gold,
                        predictions,
                        labels=VALID_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                )
            )
            fold_accuracy.append(float(accuracy_score(gold, predictions)))

        rows.append(
            {
                "c": float(c_value),
                "fold_count": len(fold_macro_f1),
                "fold_macro_f1": json.dumps(fold_macro_f1),
                "mean_macro_f1": float(np.mean(fold_macro_f1)),
                "std_macro_f1": float(np.std(fold_macro_f1)),
                "fold_accuracy": json.dumps(fold_accuracy),
                "mean_accuracy": float(np.mean(fold_accuracy)),
                "std_accuracy": float(np.std(fold_accuracy)),
            }
        )
    return pd.DataFrame(rows)


def format_metrics(
    *,
    data_path: Path,
    predictions_path: Path,
    cv_path: Path,
    train_rows: int,
    test_rows: int,
    selected_c: float,
    test_accuracy: float,
    test_macro_f1: float,
    gold: pd.Series,
    predictions: np.ndarray,
) -> str:
    matrix = confusion_matrix(gold, predictions, labels=VALID_LABELS)
    report = classification_report(
        gold,
        predictions,
        labels=VALID_LABELS,
        digits=6,
        zero_division=0,
    )
    lines = [
        f"data_path: {data_path}",
        f"predictions_path: {predictions_path}",
        f"cv_results_path: {cv_path}",
        f"train_rows: {train_rows}",
        f"test_rows: {test_rows}",
        f"seed: {SEED}",
        f"candidate_c_values: {list(CANDIDATE_C_VALUES)}",
        f"selected_c: {selected_c}",
        "selection_metric: mean validation Macro-F1; tie-break validation Accuracy then smaller C",
        "cv: StratifiedKFold(n_splits=5, shuffle=True, random_state=42), train only",
        "model_text: normalized target + ' [SEP] ' + sentence",
        "word_tfidf: ngram_range=(1, 2), sublinear_tf=True",
        "char_tfidf: analyzer=char_wb, ngram_range=(3, 5), sublinear_tf=True, min_df=2",
        "logistic_regression: class_weight=balanced, max_iter=2000, random_state=42",
        f"test_accuracy: {test_accuracy:.12f}",
        f"test_macro_f1: {test_macro_f1:.12f}",
        f"label_order: {VALID_LABELS}",
        "confusion_matrix:",
        np.array2string(matrix),
        "classification_report:",
        report,
    ]
    return "\n".join(lines).rstrip() + "\n"


def run_experiment(
    data_path: Path,
    predictions_path: Path,
    cv_path: Path,
    metrics_path: Path,
) -> dict[str, object]:
    df = pd.read_csv(data_path)
    validate_dataset(df)

    normalized_split = df["split"].astype(str).str.strip().str.lower()
    df = df.copy()
    df["polarity"] = df["polarity"].astype(str).str.strip().str.lower()
    train_df = df[normalized_split == "train"].copy()
    test_df = df[normalized_split == "test"].copy()

    train_texts = build_model_text(train_df)
    test_texts = build_model_text(test_df)
    cv_results = cross_validate_candidates(train_texts, train_df["polarity"])
    selected_c = select_best_candidate(cv_results)

    model = make_pipeline(selected_c)
    model.fit(train_texts, train_df["polarity"])
    predictions = model.predict(test_texts)
    test_accuracy = float(accuracy_score(test_df["polarity"], predictions))
    test_macro_f1 = float(
        f1_score(
            test_df["polarity"],
            predictions,
            labels=VALID_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    output_df = test_df.copy()
    output_df["model_text"] = test_texts
    output_df["prediction"] = predictions

    for path in (predictions_path, cv_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(predictions_path, index=False)
    cv_results.to_csv(cv_path, index=False)
    metrics_path.write_text(
        format_metrics(
            data_path=data_path,
            predictions_path=predictions_path,
            cv_path=cv_path,
            train_rows=len(train_df),
            test_rows=len(test_df),
            selected_c=selected_c,
            test_accuracy=test_accuracy,
            test_macro_f1=test_macro_f1,
            gold=test_df["polarity"],
            predictions=predictions,
        ),
        encoding="utf-8",
    )

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "seed": SEED,
        "candidate_c_values": list(CANDIDATE_C_VALUES),
        "selected_c": selected_c,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a train-tuned TF-IDF plus logistic-regression baseline."
    )
    parser.add_argument("--data-path", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )
    parser.add_argument("--cv-path", type=Path, default=DEFAULT_CV_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        args.data_path,
        args.predictions_path,
        args.cv_path,
        args.metrics_path,
    )
    print(f"Selected C: {summary['selected_c']}")
    print(f"Test Accuracy: {summary['test_accuracy']:.6f}")
    print(f"Test Macro-F1: {summary['test_macro_f1']:.6f}")
    print(f"Saved predictions: {args.predictions_path}")
    print(f"Saved CV results: {args.cv_path}")
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
