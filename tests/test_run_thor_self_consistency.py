import unittest

import pandas as pd

from experiments.run_thor_self_consistency import select_experiment_rows


class RunThorSelfConsistencyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
