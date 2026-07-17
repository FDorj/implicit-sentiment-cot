import unittest

import pandas as pd

from src.data_loader import clean_scapt_rows, validate_required_fields


def complete_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "source_sentence_id": "s-1",
                "domain": "laptop",
                "split": "train",
                "sentence": "The battery lasts all day.",
                "target": "battery",
                "from": 4,
                "to": 11,
                "polarity": "positive",
                "is_implicit": 1,
            }
        ]
    )


class DataLoaderValidationTests(unittest.TestCase):
    def test_complete_required_fields_are_accepted(self):
        validate_required_fields(complete_sample_frame())

    def test_missing_required_column_is_rejected(self):
        frame = complete_sample_frame().drop(columns=["target"])

        with self.assertRaisesRegex(ValueError, "target"):
            validate_required_fields(frame)

    def test_null_required_value_reports_column_and_count(self):
        frame = complete_sample_frame()
        frame.loc[0, "sentence"] = pd.NA

        with self.assertRaisesRegex(ValueError, "sentence=1"):
            validate_required_fields(frame)

    def test_cleaning_filters_conflict_row_before_required_field_validation(self):
        valid = complete_sample_frame()
        conflict = complete_sample_frame()
        conflict.loc[0, "id"] = 2
        conflict.loc[0, "polarity"] = "conflict"
        conflict.loc[0, "is_implicit"] = pd.NA

        cleaned = clean_scapt_rows(pd.concat([valid, conflict], ignore_index=True))

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["polarity"], "positive")
        self.assertEqual(cleaned.iloc[0]["is_implicit"], 1)


if __name__ == "__main__":
    unittest.main()
