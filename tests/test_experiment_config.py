import os
import unittest
from unittest.mock import patch

from src import experiment_config


class ExperimentConfigTests(unittest.TestCase):
    def test_data_path_uses_default_when_env_is_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            actual = getattr(experiment_config, "data_path", lambda: "__missing__")()

        self.assertEqual(actual, "data/processed/semeval14_scapt_isa_only_clean.csv")

    def test_data_path_uses_env_override(self):
        with patch.dict(os.environ, {"DATA_PATH": "data/processed/pilot.csv"}):
            actual = getattr(experiment_config, "data_path", lambda: "__missing__")()

        self.assertEqual(actual, "data/processed/pilot.csv")


if __name__ == "__main__":
    unittest.main()
