import unittest

from experiments.run_final_pipeline import METHODS


class FinalPipelineMethodTests(unittest.TestCase):
    def test_classical_baseline_is_the_first_compared_method(self):
        baseline = METHODS[0]

        self.assertEqual(baseline.name, "TF-IDF + Logistic Regression")
        self.assertEqual(baseline.path.name, "tfidf_logreg_predictions.csv")
        self.assertEqual(baseline.pred_col, "prediction")


if __name__ == "__main__":
    unittest.main()
