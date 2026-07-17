# TF-IDF + Logistic Regression Baseline Design

## Goal

Add and run a reproducible classical baseline for three-class implicit sentiment classification. The baseline is an experimental comparator only. This task does not modify the thesis; thesis inclusion and wording will be reviewed separately after the result is available.

## Scientific reporting rule

The run artifacts and metrics are saved regardless of whether the result is favorable to the proposed system. The test result must not be used to tune the baseline or decide which completed baseline runs to disclose. After the run, the user and researcher may decide whether the baseline belongs in the main comparison table, an auxiliary analysis, or a limitations section, but any scientific claim based on this experiment must describe the observed result faithfully.

## Data contract

- Input file: `data/processed/semeval14_scapt_isa_only_clean.csv`.
- Official split: 1,746 `train` rows and 442 `test` rows.
- Labels: `positive`, `negative`, and `neutral` from `polarity`.
- Model text: normalized `target + " [SEP] " + sentence`.
- Required columns are validated before training. Missing text values or labels, unknown split names, and labels outside the three supported classes stop the run with a clear error.
- The test split is never used for feature fitting, hyperparameter selection, thresholding, or model choice.

## Model and hyperparameter selection

The classifier is a scikit-learn pipeline whose feature space combines:

1. word TF-IDF features with unigram and bigram ranges; and
2. character-within-word TF-IDF features with 3-to-5-character ranges.

The combined sparse features are passed to multinomial-capable logistic regression with `class_weight="balanced"`, `max_iter=2000`, and `random_state=42`.

Regularization strength is selected from `C in {0.1, 1.0, 10.0}` using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training split only. The primary selection score is mean validation Macro-F1. Ties are resolved by higher mean validation accuracy and then by the smaller `C`, producing a deterministic selection.

After selection, a fresh pipeline with the selected `C` is fitted on all training rows and evaluated once on all 442 test rows.

## Outputs

The experiment writes the following files under `results/`:

- `tfidf_logreg_predictions.csv`: original identifying fields, gold polarity, and predicted polarity for every test row;
- `tfidf_logreg_cv_results.csv`: one row per candidate `C`, including fold scores, mean, and standard deviation for Macro-F1 and accuracy;
- `tfidf_logreg_metrics.txt`: data counts, fixed seed, vectorizer configuration, candidate values, selected `C`, test Accuracy, test Macro-F1, confusion matrix, and per-class classification report.

All output paths can be overridden by command-line options without changing the default reproducible run.

## Code structure

- `experiments/run_tfidf_logreg_baseline.py` contains small functions for input construction, validation, pipeline construction, train-only cross-validation, deterministic candidate selection, final fitting, evaluation, and artifact writing.
- `tests/test_tfidf_logreg_baseline.py` tests these units on synthetic data so tests do not depend on the full experiment runtime.
- Existing evaluator conventions and label order are reused where practical; the experiment does not alter the LLM pipeline or its stored predictions.

## Error handling

The script fails before model fitting when the dataset contract is violated. It creates parent output directories when needed. Convergence warnings are not suppressed; failure to converge is visible in the run output and recorded configuration makes reruns diagnosable.

## Verification and acceptance criteria

The implementation is accepted when:

1. targeted tests are written first and shown to fail for the missing behavior;
2. the targeted tests pass after implementation;
3. the complete existing test suite passes;
4. the real experiment finishes successfully on the official split;
5. the three output files exist and agree on the selected `C`, counts, and test predictions;
6. recomputing Accuracy and Macro-F1 from the prediction CSV reproduces the metrics file; and
7. no thesis source or generated thesis PDF is modified by this task.

## Explicit non-goals

- No neural encoder baseline is added in this task.
- No LLM inference is rerun.
- No tuning is performed on the test split.
- No thesis table, prose, figure, or PDF is changed until the user reviews the experimental result.
