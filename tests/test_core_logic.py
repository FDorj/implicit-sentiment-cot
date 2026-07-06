import unittest
from io import StringIO

import pandas as pd

from experiments.apply_etc_policy import apply_guarded_train_calibrated_policy, apply_train_calibrated_policy
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

    def test_guarded_policy_falls_back_when_profile_support_is_too_small(self):
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

        predictions, decisions, sources, _, learned_policy, policy_metadata = apply_guarded_train_calibrated_policy(
            df,
            key_columns=[
                "direct_prediction",
                "thor_prediction",
                "error_type",
                "diagnostic_confidence",
                "domain",
            ],
            min_support=3,
            min_margin_default=1,
            min_margin_second=1,
            min_relative_gain=0.0,
        )

        key = ("positive", "neutral", "no_error", "high", "laptop")
        self.assertEqual(sources, ["direct", "direct", "direct"])
        self.assertEqual(predictions, ["positive", "positive", "positive"])
        self.assertEqual(decisions, ["guarded_train_calibrated_use_direct_low_support"] * 3)
        self.assertEqual(learned_policy[key], "direct")
        self.assertEqual(policy_metadata[key]["fallback_reason"], "low_support")

    def test_guarded_policy_falls_back_when_best_source_margin_is_too_small(self):
        rows = []
        for polarity in ["neutral"] * 6 + ["positive"] * 5:
            rows.append(
                {
                    "split": "train",
                    "domain": "restaurant",
                    "polarity": polarity,
                    "direct_prediction": "positive",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "negative",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                }
            )
        rows.append(
            {
                "split": "test",
                "domain": "restaurant",
                "polarity": "neutral",
                "direct_prediction": "positive",
                "thor_prediction": "neutral",
                "diagnostic_label": "negative",
                "error_type": "no_error",
                "diagnostic_confidence": "high",
            }
        )
        df = pd.DataFrame(rows)

        predictions, decisions, sources, _, learned_policy, policy_metadata = apply_guarded_train_calibrated_policy(
            df,
            key_columns=[
                "direct_prediction",
                "thor_prediction",
                "error_type",
                "diagnostic_confidence",
                "domain",
            ],
            min_support=5,
            min_margin_default=2,
            min_margin_second=1,
            min_relative_gain=0.0,
        )

        key = ("positive", "neutral", "no_error", "high", "restaurant")
        self.assertEqual(sources[-1], "direct")
        self.assertEqual(predictions[-1], "positive")
        self.assertEqual(decisions[-1], "guarded_train_calibrated_use_direct_low_default_margin")
        self.assertEqual(learned_policy[key], "direct")
        self.assertEqual(policy_metadata[key]["fallback_reason"], "low_default_margin")

    def test_guarded_policy_uses_best_source_when_support_and_margins_are_strong(self):
        rows = []
        for polarity in ["neutral"] * 8 + ["positive"] * 2:
            rows.append(
                {
                    "split": "train",
                    "domain": "laptop",
                    "polarity": polarity,
                    "direct_prediction": "positive",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "positive",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                }
            )
        rows.append(
            {
                "split": "test",
                "domain": "laptop",
                "polarity": "neutral",
                "direct_prediction": "positive",
                "thor_prediction": "neutral",
                "diagnostic_label": "positive",
                "error_type": "no_error",
                "diagnostic_confidence": "high",
            }
        )
        df = pd.DataFrame(rows)

        predictions, decisions, sources, _, learned_policy, policy_metadata = apply_guarded_train_calibrated_policy(
            df,
            key_columns=[
                "direct_prediction",
                "thor_prediction",
                "error_type",
                "diagnostic_confidence",
                "domain",
            ],
            min_support=5,
            min_margin_default=2,
            min_margin_second=1,
            min_relative_gain=0.05,
        )

        key = ("positive", "neutral", "no_error", "high", "laptop")
        self.assertEqual(sources[-1], "thor")
        self.assertEqual(predictions[-1], "neutral")
        self.assertEqual(decisions[-1], "guarded_train_calibrated_use_thor")
        self.assertEqual(learned_policy[key], "thor")
        self.assertEqual(policy_metadata[key]["fallback_reason"], "")


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


class PolicyAblationHelperTests(unittest.TestCase):
    def test_guarded_ablation_grid_includes_requested_threshold_candidates(self):
        from experiments.ablate_guarded_etc_policy import build_policy_configs

        configs = build_policy_configs()
        names = {config["name"] for config in configs}

        self.assertEqual(len(configs), 110)
        self.assertIn("current_key_unguarded", names)
        self.assertIn("richer_key_unguarded", names)
        self.assertIn("guarded_richer_s10_md2_ms1_rg0p05", names)
        self.assertIn("guarded_current_s20_md3_ms2_rg0p03", names)


class MetaSelectorHelperTests(unittest.TestCase):
    def test_oracle_selects_first_correct_source_and_falls_back_to_direct(self):
        from experiments.run_meta_selector import oracle_select_sources

        df = pd.DataFrame(
            [
                {
                    "polarity": "negative",
                    "direct_prediction": "neutral",
                    "thor_prediction": "negative",
                    "diagnostic_label": "positive",
                },
                {
                    "polarity": "positive",
                    "direct_prediction": "positive",
                    "thor_prediction": "positive",
                    "diagnostic_label": "neutral",
                },
                {
                    "polarity": "neutral",
                    "direct_prediction": "positive",
                    "thor_prediction": "negative",
                    "diagnostic_label": "negative",
                },
            ]
        )

        predictions, sources = oracle_select_sources(df)

        self.assertEqual(predictions, ["negative", "positive", "positive"])
        self.assertEqual(sources, ["thor", "direct", "direct"])

    def test_build_meta_features_adds_agreement_and_vote_margin_features(self):
        from experiments.run_meta_selector import build_meta_features

        df = pd.DataFrame(
            [
                {
                    "direct_prediction": "positive",
                    "thor_prediction": "negative",
                    "diagnostic_label": "negative",
                    "controller_prediction": "negative",
                    "error_type": "missed_implicit_negative",
                    "diagnostic_confidence": "high",
                    "diagnostic_triggered": True,
                    "domain": "laptop",
                    "sc_vote_counts": "negative:2;positive:1",
                }
            ]
        )

        features = build_meta_features(df)

        self.assertEqual(features.loc[0, "direct_thor_agreement"], "no")
        self.assertEqual(features.loc[0, "direct_diagnostic_agreement"], "no")
        self.assertEqual(features.loc[0, "thor_diagnostic_agreement"], "yes")
        self.assertEqual(features.loc[0, "all_sources_agree"], "no")
        self.assertEqual(features.loc[0, "sc_top_vote_count"], 2)
        self.assertEqual(features.loc[0, "sc_vote_margin"], 1)
        self.assertAlmostEqual(features.loc[0, "sc_top_vote_share"], 2 / 3)

    def test_profile_calibration_features_capture_train_source_margins(self):
        from experiments.run_meta_selector import build_profile_calibration_features

        df = pd.DataFrame(
            [
                {
                    "polarity": "neutral",
                    "domain": "laptop",
                    "direct_prediction": "positive",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "positive",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                },
                {
                    "polarity": "positive",
                    "domain": "laptop",
                    "direct_prediction": "positive",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "positive",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                },
                {
                    "polarity": "neutral",
                    "domain": "laptop",
                    "direct_prediction": "positive",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "negative",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                },
                {
                    "polarity": "neutral",
                    "domain": "laptop",
                    "direct_prediction": "positive",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "positive",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                },
                {
                    "polarity": "negative",
                    "domain": "restaurant",
                    "direct_prediction": "negative",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "negative",
                    "error_type": "no_error",
                    "diagnostic_confidence": "high",
                },
            ]
        )

        features = build_profile_calibration_features(
            df,
            train_indices=[0, 1, 2],
            key_columns=[
                "direct_prediction",
                "thor_prediction",
                "error_type",
                "diagnostic_confidence",
                "domain",
            ],
            prefix="rich_profile",
        )

        self.assertEqual(features.loc[3, "rich_profile_support"], 3)
        self.assertEqual(features.loc[3, "rich_profile_best_source"], "thor")
        self.assertEqual(features.loc[3, "rich_profile_direct_correct_count"], 1)
        self.assertEqual(features.loc[3, "rich_profile_thor_correct_count"], 2)
        self.assertAlmostEqual(features.loc[3, "rich_profile_thor_correct_rate"], 2 / 3)
        self.assertEqual(features.loc[3, "rich_profile_margin_vs_direct"], 1)
        self.assertEqual(features.loc[4, "rich_profile_support"], 0)
        self.assertEqual(features.loc[4, "rich_profile_best_source"], "unknown")


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
