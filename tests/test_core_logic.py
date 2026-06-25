import unittest
from io import StringIO

import pandas as pd

from experiments.apply_etc_policy import apply_train_calibrated_policy
from src.controller import select_final_label
from src.reflection_pipeline import parse_diagnostic_output
from src.utils import normalize_label


def csv_buffer(df: pd.DataFrame) -> StringIO:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


class CoreLogicTests(unittest.TestCase):
    def test_normalize_label_handles_extra_text(self):
        self.assertEqual(normalize_label("positive"), "positive")
        self.assertEqual(normalize_label("negative sentiment"), "negative")
        self.assertEqual(normalize_label("The answer is neutral."), "neutral")
        self.assertEqual(normalize_label("unclear"), "unknown")

    def test_parse_diagnostic_output_normalizes_fields(self):
        parsed = parse_diagnostic_output(
            "error_type=missed implicit negative\n"
            "label=Negative\n"
            "confidence=HIGH\n"
        )

        self.assertEqual(parsed["error_type"], "missed_implicit_negative")
        self.assertEqual(parsed["diagnostic_label"], "negative")
        self.assertEqual(parsed["confidence"], "high")

    def test_controller_keeps_shared_direct_and_thor_label(self):
        label, decision = select_final_label(
            direct_label="positive",
            thor_label="positive",
            proposed_label="negative",
            error_type="missed_implicit_negative",
            confidence="high",
        )

        self.assertEqual(label, "positive")
        self.assertEqual(decision, "agreement_keep_shared_label")

    def test_controller_accepts_high_confidence_correctable_error(self):
        label, decision = select_final_label(
            direct_label="neutral",
            thor_label="positive",
            proposed_label="negative",
            error_type="missed_implicit_negative",
            confidence="high",
            min_correctable_confidence="medium",
        )

        self.assertEqual(label, "negative")
        self.assertEqual(decision, "accept_missed_implicit_negative")

    def test_train_calibrated_policy_prefers_best_train_source(self):
        df = pd.DataFrame(
            [
                {
                    "split": "train",
                    "domain": "laptop",
                    "polarity": "neutral",
                    "direct_prediction": "positive",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "positive",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                },
                {
                    "split": "train",
                    "domain": "laptop",
                    "polarity": "neutral",
                    "direct_prediction": "positive",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "positive",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                },
                {
                    "split": "test",
                    "domain": "laptop",
                    "polarity": "neutral",
                    "direct_prediction": "positive",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "positive",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                },
            ]
        )

        predictions, decisions, sources, policy_keys, learned_policy = apply_train_calibrated_policy(df)

        self.assertEqual(sources, ["thor", "thor", "thor"])
        self.assertEqual(predictions, ["neutral", "neutral", "neutral"])
        self.assertEqual(decisions, ["train_calibrated_use_thor"] * 3)
        self.assertIn(("positive", "no_error", "high", "laptop"), learned_policy)


class FinalResultsHelperTests(unittest.TestCase):
    def test_compute_method_metrics_returns_overall_train_and_test_rows(self):
        from src.final_results import MethodSpec, compute_method_metrics

        path = csv_buffer(
            pd.DataFrame(
                [
                    {"split": "train", "polarity": "positive", "prediction": "positive"},
                    {"split": "train", "polarity": "negative", "prediction": "positive"},
                    {"split": "test", "polarity": "neutral", "prediction": "neutral"},
                ]
            )
        )
        metrics = compute_method_metrics(
            [MethodSpec(name="Direct", path=path, pred_col="prediction")]
        )

        self.assertEqual(metrics["method"].tolist(), ["Direct", "Direct", "Direct"])
        self.assertEqual(metrics["split"].tolist(), ["overall", "train", "test"])
        self.assertEqual(metrics["n_eval"].tolist(), [3, 2, 1])
        self.assertEqual(metrics["accuracy"].round(6).tolist(), [0.666667, 0.5, 1.0])

    def test_validate_final_chain_detects_aligned_predictions(self):
        from src.final_results import validate_final_chain

        rows = [
            {
                "id": 1,
                "source_sentence_id": "s1",
                "sentence": "A",
                "target": "x",
                "from": 0,
                "to": 1,
                "polarity": "positive",
                "domain": "laptop",
                "split": "test",
            }
        ]

        direct_path = csv_buffer(pd.DataFrame([{**rows[0], "prediction": "positive"}]))
        thor_path = csv_buffer(pd.DataFrame([{**rows[0], "prediction": "neutral"}]))
        etc_path = csv_buffer(
            pd.DataFrame(
                [
                    {
                        **rows[0],
                        "direct_prediction": "positive",
                        "thor_prediction": "neutral",
                        "diagnostic_label": "positive",
                        "diagnostic_confidence": "high",
                        "error_type": "missed_implicit_positive",
                        "controller_prediction": "positive",
                    }
                ]
            )
        )
        selected_path = csv_buffer(pd.DataFrame([{**rows[0], "selected_prediction": "positive"}]))

        summary = validate_final_chain(
            direct_path=direct_path,
            thor_path=thor_path,
            etc_path=etc_path,
            selected_path=selected_path,
        )

        self.assertEqual(summary["row_count"], 1)
        self.assertEqual(summary["diagnostic_columns_present"], True)
        self.assertEqual(summary["direct_alignment_mismatches"], 0)
        self.assertEqual(summary["thor_alignment_mismatches"], 0)


class QualitativeExamplesTests(unittest.TestCase):
    def test_add_direct_comparison_group_marks_gain_and_loss(self):
        from src.qualitative_examples import add_direct_comparison_group

        df = pd.DataFrame(
            [
                {
                    "polarity": "neutral",
                    "direct_prediction": "positive",
                    "selected_prediction": "neutral",
                },
                {
                    "polarity": "negative",
                    "direct_prediction": "negative",
                    "selected_prediction": "neutral",
                },
                {
                    "polarity": "positive",
                    "direct_prediction": "positive",
                    "selected_prediction": "positive",
                },
                {
                    "polarity": "negative",
                    "direct_prediction": "positive",
                    "selected_prediction": "neutral",
                },
            ]
        )

        labeled = add_direct_comparison_group(df)

        self.assertEqual(
            labeled["direct_comparison_group"].tolist(),
            ["gain_vs_direct", "loss_vs_direct", "both_correct", "both_wrong"],
        )

    def test_select_qualitative_examples_is_test_only_and_limited_per_group(self):
        from src.qualitative_examples import select_qualitative_examples

        df = pd.DataFrame(
            [
                {
                    "id": 3,
                    "domain": "restaurant",
                    "split": "test",
                    "polarity": "neutral",
                    "direct_prediction": "positive",
                    "selected_prediction": "neutral",
                    "sentence": "A",
                    "target": "x",
                },
                {
                    "id": 1,
                    "domain": "laptop",
                    "split": "test",
                    "polarity": "negative",
                    "direct_prediction": "negative",
                    "selected_prediction": "neutral",
                    "sentence": "B",
                    "target": "y",
                },
                {
                    "id": 2,
                    "domain": "laptop",
                    "split": "test",
                    "polarity": "neutral",
                    "direct_prediction": "positive",
                    "selected_prediction": "neutral",
                    "sentence": "C",
                    "target": "z",
                },
                {
                    "id": 4,
                    "domain": "laptop",
                    "split": "train",
                    "polarity": "neutral",
                    "direct_prediction": "positive",
                    "selected_prediction": "neutral",
                    "sentence": "D",
                    "target": "w",
                },
            ]
        )

        examples = select_qualitative_examples(df, per_group=1)

        self.assertEqual(examples["split"].unique().tolist(), ["test"])
        self.assertEqual(examples["direct_comparison_group"].tolist(), ["gain_vs_direct", "loss_vs_direct"])
        self.assertEqual(examples["id"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
