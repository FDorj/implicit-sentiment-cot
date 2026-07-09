import unittest

import pandas as pd

from experiments.run_logistic_source_ranker import (
    build_candidate_rows,
    select_sources_from_scores,
)


class LogisticSourceRankerTests(unittest.TestCase):
    def test_build_candidate_rows_expands_each_sample_source_and_correctness_label(self):
        df = pd.DataFrame(
            [
                {
                    "polarity": "negative",
                    "direct_prediction": "neutral",
                    "thor_prediction": "negative",
                    "diagnostic_label": "positive",
                    "controller_prediction": "negative",
                    "error_type": "missed_implicit_negative",
                    "diagnostic_confidence": "high",
                    "domain": "laptop",
                    "diagnostic_triggered": True,
                }
            ]
        )

        candidates = build_candidate_rows(df)

        self.assertEqual(candidates["candidate_source"].tolist(), ["direct", "thor", "diagnostic"])
        self.assertEqual(candidates["candidate_prediction"].tolist(), ["neutral", "negative", "positive"])
        self.assertEqual(candidates["candidate_correct"].tolist(), [False, True, False])
        self.assertEqual(candidates["candidate_agrees_thor"].tolist(), ["no", "yes", "no"])
        self.assertEqual(candidates["direct_thor_agreement"].tolist(), ["no", "no", "no"])

    def test_select_sources_from_scores_picks_highest_scoring_candidate_per_sample(self):
        df = pd.DataFrame(
            [
                {
                    "direct_prediction": "positive",
                    "thor_prediction": "negative",
                    "diagnostic_label": "neutral",
                },
                {
                    "direct_prediction": "neutral",
                    "thor_prediction": "neutral",
                    "diagnostic_label": "positive",
                },
            ]
        )
        candidate_scores = pd.DataFrame(
            [
                {"row_index": 0, "candidate_source": "direct", "candidate_prediction": "positive", "score": 0.20},
                {"row_index": 0, "candidate_source": "thor", "candidate_prediction": "negative", "score": 0.80},
                {"row_index": 0, "candidate_source": "diagnostic", "candidate_prediction": "neutral", "score": 0.10},
                {"row_index": 1, "candidate_source": "direct", "candidate_prediction": "neutral", "score": 0.40},
                {"row_index": 1, "candidate_source": "thor", "candidate_prediction": "neutral", "score": 0.30},
                {"row_index": 1, "candidate_source": "diagnostic", "candidate_prediction": "positive", "score": 0.60},
            ]
        )

        predictions, sources = select_sources_from_scores(df, candidate_scores)

        self.assertEqual(predictions.tolist(), ["negative", "positive"])
        self.assertEqual(sources.tolist(), ["thor", "diagnostic"])


if __name__ == "__main__":
    unittest.main()
