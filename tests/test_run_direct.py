import unittest

import pandas as pd

from experiments.run_direct import (
    OUTPUT_COLUMNS,
    attach_previous_outputs,
    completed_prediction_mask,
    select_experiment_rows,
)


class RunDirectTests(unittest.TestCase):
    def test_select_experiment_rows_filters_split_before_debug_limit(self):
        df = pd.DataFrame(
            [
                {"id": 1, "split": "train"},
                {"id": 2, "split": "test"},
                {"id": 3, "split": "train"},
                {"id": 4, "split": "test"},
                {"id": 5, "split": "test"},
            ]
        )

        selected = select_experiment_rows(df, data_split="test", debug_n=2)

        self.assertEqual(selected["id"].tolist(), [2, 4])
        self.assertEqual(selected["split"].tolist(), ["test", "test"])

    def test_select_experiment_rows_rejects_missing_split(self):
        df = pd.DataFrame([{"id": 1, "split": "train"}])

        with self.assertRaisesRegex(ValueError, "No rows found"):
            select_experiment_rows(df, data_split="validation", debug_n=None)

    def test_attach_previous_outputs_restores_completed_predictions_by_key(self):
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "source_sentence_id": "s1",
                    "sentence": "A",
                    "target": "x",
                    "from": 0,
                    "to": 1,
                    "polarity": "positive",
                },
                {
                    "id": 2,
                    "source_sentence_id": "s2",
                    "sentence": "B",
                    "target": "y",
                    "from": 0,
                    "to": 1,
                    "polarity": "negative",
                },
            ]
        )
        previous_df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "source_sentence_id": "s1",
                    "sentence": "A",
                    "target": "x",
                    "from": 0,
                    "to": 1,
                    "polarity": "positive",
                    "raw_output": "positive",
                    "prediction": "positive",
                }
            ]
        )

        restored = attach_previous_outputs(df, previous_df)

        self.assertEqual(restored.loc[0, "raw_output"], "positive")
        self.assertEqual(restored.loc[0, "prediction"], "positive")
        self.assertTrue(pd.isna(restored.loc[1, "raw_output"]))
        self.assertTrue(pd.isna(restored.loc[1, "prediction"]))
        self.assertEqual(OUTPUT_COLUMNS, ["raw_output", "prediction"])

    def test_completed_prediction_mask_treats_unknown_as_pending(self):
        df = pd.DataFrame(
            {
                "prediction": [
                    "positive",
                    "unknown",
                    pd.NA,
                ]
            }
        )

        completed = completed_prediction_mask(df)

        self.assertEqual(completed.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
