# Thesis Integration of the TF-IDF Logistic Baseline

## Goal

Integrate the completed TF-IDF plus logistic-regression experiment into the thesis as a reproducible classical baseline, while preserving the Amirkabir template and keeping the baseline separate from the proposed LLM system.

The thesis narrative presents this comparator as the first experimental step: the study first establishes a low-cost classical reference point, then evaluates direct LLM prediction, and only afterward develops the reasoning, review, controller, and source-selection variants. The prose describes this research order naturally without claiming that the classical baseline is a component of the final architecture.

## Chapter 4 placement and content

Add a subsection titled «خط پایه کلاسیک مبتنی بر TF-IDF و رگرسیون لجستیک» after data preparation and before the LLM prediction-source sections. The subsection begins by explaining that the first experimental step was to establish a classical reference point before using language models. It states that this model is an independent comparator rather than a component of the proposed pipeline.

It documents the exact evidence-backed configuration:

- input text is normalized `target + [SEP] + sentence`;
- word TF-IDF uses unigram and bigram features;
- character-within-word TF-IDF uses 3-to-5-character features;
- logistic regression uses balanced class weights, maximum 2,000 iterations, and seed 42;
- `C` is selected from 0.1, 1, and 10 by five-fold stratified cross-validation on the 1,746 training rows only;
- selection prioritizes mean validation Macro-F1, then validation Accuracy, then smaller `C`;
- `C=1` is selected, refitted on all training rows, and evaluated once on 442 test rows;
- no test label is used for feature fitting or hyperparameter selection.

The implementation-module table gains the baseline script and its responsibility.

## Chapter 5 placement and content

The compared-method count changes from seven to eight. The classical baseline becomes the first item, explicitly labeled as non-LLM. The Qwen direct predictor remains the first LLM method. The method list and analysis follow the same chronological research narrative: classical baseline, direct LLM prediction, reasoning variants, review/controller variants, and final source selection.

The main full-test table becomes a comparison of all principal methods rather than only Qwen-based methods. It gains the row:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.547511 | 0.514082 |

The surrounding analysis reports that:

- direct Qwen exceeds the classical baseline by about 13.12 Accuracy points and 16.00 Macro-F1 points;
- the final system exceeds it by about 17.65 Accuracy points and 20.50 Macro-F1 points;
- the baseline's weakest class is negative, with F1 equal to 0.374332;
- this baseline establishes a classical reference point but is not presented as a state-of-the-art fine-tuned encoder.

The existing direct-versus-final analysis remains unchanged. No claim attributes the observed gap solely to the proposed reasoning architecture, because the classical baseline and Qwen methods differ in model family and capacity.

## Reproducible results and figure integration

Add `TF-IDF + Logistic Regression` to `experiments/run_final_pipeline.py` using the saved test prediction file and prediction column. Regenerate `results/final_results_table.csv` and `.md` so the table has an evidence-backed test row.

Add the baseline to the first position of the main figure's method order in `experiments/generate_thesis_result_figures.py`. Increase the plot height enough to keep eight method labels readable and update the figure caption/paragraph to describe eight principal methods rather than seven Qwen-only variants. Other Chapter 5 figures remain unchanged.

## Tests and verification

Update figure-data tests to require eight test methods, with the baseline first and the final system last. Update rendering tests to require the baseline label in the symbolic y-axis. Add or update final-pipeline coverage so the baseline `MethodSpec` points to the correct file and prediction column.

Verification consists of:

1. running the baseline-specific tests;
2. running the complete Python test suite;
3. regenerating the final results table and all Chapter 5 figures;
4. rebuilding the thesis PDF;
5. checking the LaTeX log for undefined references/citations, fatal errors, and overfull boxes;
6. confirming the PDF contains the new method name and reported metrics; and
7. confirming all thesis numbers match the saved CSV and metrics files.

## Explicit non-goals

- Do not rerun any LLM.
- Do not retune the baseline after seeing test results.
- Do not add an encoder baseline in this change.
- Do not describe TF-IDF plus logistic regression as part of the proposed system.
- Do not change the reported LLM or final-system metrics.
