import unittest

import pandas as pd

from src.evaluator import evaluate_predictions


class EvaluatorTests(unittest.TestCase):
    def test_invalid_predictions_are_excluded_from_main_metrics_but_reported(self):
        df = pd.DataFrame(
            [
                {"polarity": "positive", "prediction": "positive"},
                {"polarity": "negative", "prediction": "unknown"},
                {"polarity": "neutral", "prediction": ""},
                {"polarity": "negative", "prediction": "negative"},
            ]
        )

        metrics = evaluate_predictions(df)

        self.assertEqual(metrics["n_total"], 4)
        self.assertEqual(metrics["n_eval"], 2)
        self.assertEqual(metrics["n_invalid"], 2)
        self.assertEqual(metrics["valid_prediction_rate"], 0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
