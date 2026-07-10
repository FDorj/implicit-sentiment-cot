import unittest

import pandas as pd

from experiments.run_error_type_controller import pending_error_type_indices


class RunErrorTypeControllerTests(unittest.TestCase):
    def test_pending_indices_can_rerun_invalid_triggered_diagnostics(self):
        df = pd.DataFrame(
            [
                {
                    "controller_prediction": "negative",
                    "diagnostic_triggered": True,
                    "diagnostic_label": "unknown",
                },
                {
                    "controller_prediction": "neutral",
                    "diagnostic_triggered": True,
                    "diagnostic_label": "negative",
                },
                {
                    "controller_prediction": "positive",
                    "diagnostic_triggered": False,
                    "diagnostic_label": "neutral",
                },
            ],
            index=[10, 20, 30],
        )

        self.assertEqual(pending_error_type_indices(df, rerun_invalid_diagnostics=False), [])
        self.assertEqual(pending_error_type_indices(df, rerun_invalid_diagnostics=True), [10])


if __name__ == "__main__":
    unittest.main()
