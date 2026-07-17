import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from experiments.run_tfidf_logreg_baseline import (
    VALID_LABELS,
    build_model_text,
    cross_validate_candidates,
    run_experiment,
    select_best_candidate,
    validate_dataset,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "train",
                "sentence": "  Service   was slow. ",
                "target": " staff ",
                "polarity": "negative",
            },
            {
                "split": "train",
                "sentence": "It works.",
                "target": "battery",
                "polarity": "positive",
            },
            {
                "split": "train",
                "sentence": "It exists.",
                "target": "screen",
                "polarity": "neutral",
            },
            {
                "split": "test",
                "sentence": "Average item.",
                "target": "item",
                "polarity": "neutral",
            },
        ]
    )


def balanced_training_data(repeats: int = 5) -> pd.DataFrame:
    rows = []
    phrases = {
        "positive": "excellent reliable pleasant",
        "negative": "awful broken unpleasant",
        "neutral": "ordinary available standard",
    }
    for label, phrase in phrases.items():
        for index in range(repeats):
            rows.append(
                {
                    "split": "train",
                    "sentence": f"{phrase} {index}",
                    "target": "item",
                    "polarity": label,
                }
            )
        rows.append(
            {
                "split": "test",
                "sentence": phrase,
                "target": "item",
                "polarity": label,
            }
        )
    return pd.DataFrame(rows)


class TfidfLogregBaselineTests(unittest.TestCase):
    def test_build_model_text_normalizes_target_and_sentence(self):
        df = sample_frame()

        self.assertEqual(
            build_model_text(df).iloc[0],
            "staff [SEP] Service was slow.",
        )

    def test_validate_dataset_rejects_missing_required_values(self):
        df = sample_frame()
        df.loc[0, "sentence"] = None

        with self.assertRaisesRegex(ValueError, "Missing values"):
            validate_dataset(df)

    def test_validate_dataset_rejects_unknown_labels(self):
        df = sample_frame()
        df.loc[0, "polarity"] = "mixed"

        with self.assertRaisesRegex(ValueError, "Unsupported labels"):
            validate_dataset(df)

    def test_validate_dataset_requires_exact_train_and_test_splits(self):
        df = sample_frame()
        df.loc[df["split"] == "test", "split"] = "validation"

        with self.assertRaisesRegex(ValueError, "exactly train/test"):
            validate_dataset(df)

    def test_select_best_candidate_uses_smaller_c_after_metric_tie(self):
        results = pd.DataFrame(
            [
                {"c": 1.0, "mean_macro_f1": 0.5, "mean_accuracy": 0.6},
                {"c": 0.1, "mean_macro_f1": 0.5, "mean_accuracy": 0.6},
                {"c": 10.0, "mean_macro_f1": 0.4, "mean_accuracy": 0.9},
            ]
        )

        self.assertEqual(select_best_candidate(results), 0.1)

    def test_cross_validate_candidates_returns_one_deterministic_row_per_c(self):
        df = balanced_training_data()
        train = df[df["split"] == "train"]

        results = cross_validate_candidates(
            build_model_text(train),
            train["polarity"],
            c_values=(0.1, 1.0),
        )

        self.assertEqual(results["c"].tolist(), [0.1, 1.0])
        self.assertEqual(results["fold_count"].tolist(), [5, 5])
        self.assertTrue(results["mean_macro_f1"].between(0.0, 1.0).all())
        self.assertTrue(results["mean_accuracy"].between(0.0, 1.0).all())

    def test_run_experiment_writes_predictions_cv_and_reproducible_metrics(self):
        root = Path("results")
        data_path = root / "_test_tfidf_logreg_data.csv"
        predictions_path = root / "_test_tfidf_logreg_predictions.csv"
        cv_path = root / "_test_tfidf_logreg_cv.csv"
        metrics_path = root / "_test_tfidf_logreg_metrics.txt"
        for path in (data_path, predictions_path, cv_path, metrics_path):
            self.addCleanup(path.unlink, missing_ok=True)
        balanced_training_data().to_csv(data_path, index=False)

        summary = run_experiment(
            data_path,
            predictions_path,
            cv_path,
            metrics_path,
        )
        predictions = pd.read_csv(predictions_path)
        cv_results = pd.read_csv(cv_path)

        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(cv_results), 3)
        self.assertAlmostEqual(
            summary["test_accuracy"],
            accuracy_score(predictions["polarity"], predictions["prediction"]),
        )
        self.assertAlmostEqual(
            summary["test_macro_f1"],
            f1_score(
                predictions["polarity"],
                predictions["prediction"],
                labels=VALID_LABELS,
                average="macro",
                zero_division=0,
            ),
        )
        metrics_text = metrics_path.read_text(encoding="utf-8")
        self.assertIn(f"selected_c: {summary['selected_c']}", metrics_text)
        self.assertIn("test_rows: 3", metrics_text)


if __name__ == "__main__":
    unittest.main()
