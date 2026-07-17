# TF-IDF + Logistic Regression Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, and run a leakage-free TF-IDF plus logistic-regression baseline on the official ISA train/test split without changing the thesis.

**Architecture:** A single experiment module owns dataset validation, text construction, sparse feature/model construction, train-only cross-validation, final evaluation, and artifact serialization. Small pure functions make selection and formatting independently testable; the command-line entry point wires them together and evaluates the test split once.

**Tech Stack:** Python 3, pandas, scikit-learn (`FeatureUnion`, `TfidfVectorizer`, `LogisticRegression`, `StratifiedKFold`), standard-library `unittest`.

## Global Constraints

- Input defaults to `data/processed/semeval14_scapt_isa_only_clean.csv` with 1,746 train rows and 442 test rows.
- Input text is `target + " [SEP] " + sentence` after whitespace normalization.
- Supported labels and fixed report order are `positive`, `negative`, `neutral`.
- Candidate values are exactly `C = 0.1, 1.0, 10.0`.
- Cross-validation is exactly five-fold stratified, shuffled with `random_state=42`, on train only.
- Selection maximizes mean validation Macro-F1, then mean validation Accuracy, then prefers smaller `C`.
- Logistic regression uses `class_weight="balanced"`, `max_iter=2000`, and `random_state=42`.
- Test is evaluated once after refitting the selected configuration on all training data.
- Save all run artifacts regardless of result direction.
- Do not modify any thesis source, figure, table, or PDF.

## File Structure

- Create `experiments/run_tfidf_logreg_baseline.py`: complete reusable experiment implementation and CLI.
- Create `tests/test_tfidf_logreg_baseline.py`: unit and small integration tests for validation, text construction, deterministic model selection, CV shape, and artifact consistency.
- Create by execution `results/tfidf_logreg_predictions.csv`: official test predictions.
- Create by execution `results/tfidf_logreg_cv_results.csv`: train-only candidate summaries.
- Create by execution `results/tfidf_logreg_metrics.txt`: final configuration and test metrics.

---

### Task 1: Dataset contract, feature pipeline, and deterministic selection

**Files:**
- Create: `experiments/run_tfidf_logreg_baseline.py`
- Create: `tests/test_tfidf_logreg_baseline.py`

**Interfaces:**
- Produces: `validate_dataset(df: pd.DataFrame) -> None`
- Produces: `build_model_text(df: pd.DataFrame) -> pd.Series`
- Produces: `make_pipeline(c_value: float) -> Pipeline`
- Produces: `select_best_candidate(results: pd.DataFrame) -> float`

- [ ] **Step 1: Write failing tests for validation, text construction, and tie-breaking**

```python
import unittest

import pandas as pd

from experiments.run_tfidf_logreg_baseline import (
    build_model_text,
    select_best_candidate,
    validate_dataset,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"split": "train", "sentence": "  Service   was slow. ", "target": " staff ", "polarity": "negative"},
            {"split": "train", "sentence": "It works.", "target": "battery", "polarity": "positive"},
            {"split": "train", "sentence": "It exists.", "target": "screen", "polarity": "neutral"},
            {"split": "test", "sentence": "Average item.", "target": "item", "polarity": "neutral"},
        ]
    )


class TfidfLogregBaselineTests(unittest.TestCase):
    def test_build_model_text_normalizes_target_and_sentence(self):
        df = sample_frame()
        self.assertEqual(build_model_text(df).iloc[0], "staff [SEP] Service was slow.")

    def test_validate_dataset_rejects_missing_required_values(self):
        df = sample_frame()
        df.loc[0, "sentence"] = None
        with self.assertRaisesRegex(ValueError, "Missing values"):
            validate_dataset(df)

    def test_validate_dataset_rejects_unknown_labels(self):
        df = sample_frame()
        df.loc[0, "polarity"] = "mixed"
        with self.assertRaisesRegex(ValueError, "Unsupported labels"):
            validate_dataset(df)

    def test_select_best_candidate_uses_smaller_c_after_metric_tie(self):
        results = pd.DataFrame(
            [
                {"c": 1.0, "mean_macro_f1": 0.5, "mean_accuracy": 0.6},
                {"c": 0.1, "mean_macro_f1": 0.5, "mean_accuracy": 0.6},
                {"c": 10.0, "mean_macro_f1": 0.4, "mean_accuracy": 0.9},
            ]
        )
        self.assertEqual(select_best_candidate(results), 0.1)
```

- [ ] **Step 2: Run tests to verify missing-module failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_tfidf_logreg_baseline -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'experiments.run_tfidf_logreg_baseline'`.

- [ ] **Step 3: Implement constants, validation, text construction, pipeline, and candidate selection**

Create `experiments/run_tfidf_logreg_baseline.py` with project-root import setup and these behaviors:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import VALID_LABELS
from src.experiment_config import DEFAULT_DATA_PATH

SEED = 42
CANDIDATE_C_VALUES = (0.1, 1.0, 10.0)
REQUIRED_COLUMNS = ("split", "sentence", "target", "polarity")


def validate_dataset(df: pd.DataFrame) -> None:
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    missing_counts = df.loc[:, REQUIRED_COLUMNS].isna().sum()
    missing_counts = {name: int(count) for name, count in missing_counts.items() if count}
    if missing_counts:
        raise ValueError(f"Missing values in required columns: {missing_counts}")
    splits = set(df["split"].astype(str).str.strip().str.lower())
    if splits != {"train", "test"}:
        raise ValueError(f"Expected exactly train/test splits, found: {sorted(splits)}")
    labels = set(df["polarity"].astype(str).str.strip().str.lower())
    unsupported = sorted(labels - set(VALID_LABELS))
    if unsupported:
        raise ValueError(f"Unsupported labels: {unsupported}")


def normalize_text(value: object) -> str:
    return " ".join(str(value).split())


def build_model_text(df: pd.DataFrame) -> pd.Series:
    targets = df["target"].map(normalize_text)
    sentences = df["sentence"].map(normalize_text)
    return targets + " [SEP] " + sentences


def make_pipeline(c_value: float) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ]
    )
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def select_best_candidate(results: pd.DataFrame) -> float:
    ranked = results.sort_values(
        ["mean_macro_f1", "mean_accuracy", "c"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["c"])
```

- [ ] **Step 4: Run targeted tests**

Run: `.venv\Scripts\python.exe -m unittest tests.test_tfidf_logreg_baseline -v`

Expected: four tests pass.

---

### Task 2: Train-only cross-validation and final result serialization

**Files:**
- Modify: `experiments/run_tfidf_logreg_baseline.py`
- Modify: `tests/test_tfidf_logreg_baseline.py`

**Interfaces:**
- Consumes: `build_model_text`, `make_pipeline`, `select_best_candidate`
- Produces: `cross_validate_candidates(texts: pd.Series, labels: pd.Series, c_values: tuple[float, ...] = CANDIDATE_C_VALUES) -> pd.DataFrame`
- Produces: `run_experiment(data_path: Path, predictions_path: Path, cv_path: Path, metrics_path: Path) -> dict[str, object]`
- Produces: `format_metrics(...) -> str`

- [ ] **Step 1: Add failing tests for CV output and end-to-end artifact agreement**

Add synthetic balanced examples for all three labels in both split values, then assert:

```python
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from experiments.run_tfidf_logreg_baseline import (
    VALID_LABELS,
    build_model_text,
    cross_validate_candidates,
    run_experiment,
)


def balanced_training_data(repeats: int = 5) -> pd.DataFrame:
    rows = []
    phrases = {
        "positive": "excellent reliable pleasant",
        "negative": "awful broken unpleasant",
        "neutral": "ordinary available standard",
    }
    for label, phrase in phrases.items():
        for index in range(repeats):
            rows.append({"split": "train", "sentence": f"{phrase} {index}", "target": "item", "polarity": label})
        rows.append({"split": "test", "sentence": phrase, "target": "item", "polarity": label})
    return pd.DataFrame(rows)


def test_cross_validate_candidates_returns_one_deterministic_row_per_c(self):
    df = balanced_training_data()
    train = df[df["split"] == "train"]
    results = cross_validate_candidates(build_model_text(train), train["polarity"], c_values=(0.1, 1.0))
    self.assertEqual(results["c"].tolist(), [0.1, 1.0])
    self.assertEqual(results["fold_count"].tolist(), [5, 5])
    self.assertTrue(results["mean_macro_f1"].between(0.0, 1.0).all())
    self.assertTrue(results["mean_accuracy"].between(0.0, 1.0).all())


def test_run_experiment_writes_predictions_cv_and_reproducible_metrics(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data_path = root / "data.csv"
        predictions_path = root / "predictions.csv"
        cv_path = root / "cv.csv"
        metrics_path = root / "metrics.txt"
        balanced_training_data().to_csv(data_path, index=False)

        summary = run_experiment(data_path, predictions_path, cv_path, metrics_path)
        predictions = pd.read_csv(predictions_path)
        cv_results = pd.read_csv(cv_path)

        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(cv_results), 3)
        self.assertAlmostEqual(summary["test_accuracy"], accuracy_score(predictions["polarity"], predictions["prediction"]))
        self.assertAlmostEqual(summary["test_macro_f1"], f1_score(predictions["polarity"], predictions["prediction"], labels=VALID_LABELS, average="macro", zero_division=0))
        text = metrics_path.read_text(encoding="utf-8")
        self.assertIn(f"selected_c: {summary['selected_c']}", text)
        self.assertIn("test_rows: 3", text)
```

- [ ] **Step 2: Run the new tests and verify attribute/function failures**

Run: `.venv\Scripts\python.exe -m unittest tests.test_tfidf_logreg_baseline -v`

Expected: new tests fail because `cross_validate_candidates` and `run_experiment` do not exist.

- [ ] **Step 3: Implement manual stratified CV, refit, metrics, and outputs**

Add imports for `numpy`, `confusion_matrix`, `classification_report`, `accuracy_score`, `f1_score`, and `StratifiedKFold`. For every candidate and fold, build a fresh pipeline, fit only the fold-training rows, and record validation Accuracy and Macro-F1 with explicit `labels=VALID_LABELS` and `zero_division=0`. Serialize fold-score lists as JSON strings and return columns:

```python
[
    "c", "fold_count", "fold_macro_f1", "mean_macro_f1", "std_macro_f1",
    "fold_accuracy", "mean_accuracy", "std_accuracy",
]
```

`run_experiment` must:

```python
df = pd.read_csv(data_path)
validate_dataset(df)
normalized_split = df["split"].astype(str).str.strip().str.lower()
train_df = df[normalized_split == "train"].copy()
test_df = df[normalized_split == "test"].copy()
cv_results = cross_validate_candidates(build_model_text(train_df), train_df["polarity"])
selected_c = select_best_candidate(cv_results)
model = make_pipeline(selected_c)
model.fit(build_model_text(train_df), train_df["polarity"])
predictions = model.predict(build_model_text(test_df))
```

The prediction artifact preserves all test columns and appends `model_text` and `prediction`. The summary dictionary contains train/test counts, seed, candidates, selected C, Accuracy, and Macro-F1. The metrics text contains the same values plus exact vectorizer settings, confusion matrix in `VALID_LABELS` order, and `classification_report(..., labels=VALID_LABELS, digits=6, zero_division=0)`. Create each output parent directory before writing.

- [ ] **Step 4: Add CLI argument parsing and main entry point**

Implement flags:

```text
--data-path        default data/processed/semeval14_scapt_isa_only_clean.csv
--predictions-path default results/tfidf_logreg_predictions.csv
--cv-path          default results/tfidf_logreg_cv_results.csv
--metrics-path     default results/tfidf_logreg_metrics.txt
```

`main()` calls `run_experiment`, prints the selected C and final test metrics, and is protected by `if __name__ == "__main__":`.

- [ ] **Step 5: Run targeted tests**

Run: `.venv\Scripts\python.exe -m unittest tests.test_tfidf_logreg_baseline -v`

Expected: all baseline tests pass.

---

### Task 3: Full verification and official experiment run

**Files:**
- Create by execution: `results/tfidf_logreg_predictions.csv`
- Create by execution: `results/tfidf_logreg_cv_results.csv`
- Create by execution: `results/tfidf_logreg_metrics.txt`
- Do not modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/**`

**Interfaces:**
- Consumes: CLI in `experiments/run_tfidf_logreg_baseline.py`
- Produces: three mutually consistent official experiment artifacts and a user-facing result summary

- [ ] **Step 1: Run the complete existing test suite before the experiment**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`

Expected: all tests pass; any unrelated pre-existing failure is investigated and separated from baseline changes.

- [ ] **Step 2: Snapshot thesis paths and run the official baseline**

Record `git status --short` entries under the thesis directory, then run:

```powershell
python experiments/run_tfidf_logreg_baseline.py
```

Expected: process exits zero and prints selected C, test Accuracy, and test Macro-F1.

- [ ] **Step 3: Independently verify artifact consistency**

Run a read-only Python check that loads predictions and CV results, verifies 442 prediction rows, verifies three candidate rows and five folds per candidate, checks every prediction is in `VALID_LABELS`, recomputes Accuracy and Macro-F1 from the CSV, and confirms those numbers match `tfidf_logreg_metrics.txt` to printed precision.

- [ ] **Step 4: Verify thesis files did not change during this task**

Compare the post-run thesis-directory status with the snapshot from Step 2. Expected: identical; the baseline task added no thesis changes.

- [ ] **Step 5: Run targeted and complete tests once more**

Run: `.venv\Scripts\python.exe -m unittest tests.test_tfidf_logreg_baseline -v`

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`

Expected: both commands pass.

- [ ] **Step 6: Report without editing the thesis**

Report selected C, all CV candidate means/standard deviations, official test Accuracy and Macro-F1, per-class scores, confusion matrix, artifact links, test evidence, and the unchanged-thesis check. Explicitly state that thesis inclusion has not yet occurred and requires a separate review decision.
