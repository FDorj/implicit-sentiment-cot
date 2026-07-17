# Thesis TF-IDF Baseline Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the completed TF-IDF plus logistic-regression baseline to the reproducible result pipeline, Chapter 4 methodology, Chapter 5 comparisons, main result figure, and compiled thesis PDF.

**Architecture:** The saved prediction CSV remains the source of truth. The final-result pipeline evaluates it like every other method, the figure generator consumes the regenerated result table, and the Persian thesis reports the same values in a baseline-first research narrative.

**Tech Stack:** Python 3, pandas, scikit-learn, standard-library `unittest`, LaTeX/XePersian, PGFPlots, existing PowerShell build scripts.

## Global Constraints

- Present the classical baseline as the first experiment and an independent comparator, not as a component of the proposed system.
- Use only the existing train-tuned result: Accuracy `0.547511312217`, Macro-F1 `0.514082021498`, selected `C=1.0`.
- Do not rerun an LLM or retune the baseline.
- Preserve all existing Qwen, THOR, Gemini, controller, and final-system numbers.
- Keep Persian terminology and LaTeX conventions consistent with the current Amirkabir template.

---

### Task 1: Reproducible result-table and figure integration

**Files:**
- Modify: `tests/test_generate_thesis_result_figures.py`
- Modify: `experiments/run_final_pipeline.py`
- Modify: `experiments/generate_thesis_result_figures.py`
- Regenerate: `results/final_results_table.csv`
- Regenerate: `results/final_results_table.md`

**Interfaces:**
- Consumes: `results/tfidf_logreg_predictions.csv`, column `prediction`
- Produces: `MethodSpec(name="TF-IDF + Logistic Regression", ...)`
- Produces: an eight-row test comparison from `load_main_test_results`

- [ ] **Step 1: Change figure tests first**

Update the main-result expectations to require eight rows, baseline first, final pipeline last, and the exact baseline metrics:

```python
def test_main_results_are_test_only_and_include_eight_methods(self):
    rows = load_main_test_results(REPO_ROOT)
    self.assertEqual(len(rows), 8)
    self.assertEqual(rows[0]["method"], "TF-IDF + Logistic Regression")
    self.assertAlmostEqual(rows[0]["accuracy"], 0.5475113122171946)
    self.assertAlmostEqual(rows[0]["macro_f1"], 0.5140820214979989)
    self.assertEqual(rows[-1]["method"], "Final selected pipeline")
```

Update the order and rendered y-axis expectations so `TF-IDF + Logistic Regression` precedes `Direct Qwen3 8B`.

- [ ] **Step 2: Run the targeted tests and observe failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_generate_thesis_result_figures -v`

Expected: failure because the result table and `MAIN_METHOD_ORDER` still contain seven methods.

- [ ] **Step 3: Add the baseline MethodSpec**

In `experiments/run_final_pipeline.py`, define:

```python
TFIDF_LOGREG_PATH = RESULTS_DIR / "tfidf_logreg_predictions.csv"
```

Insert first in `METHODS`:

```python
MethodSpec(
    name="TF-IDF + Logistic Regression",
    path=TFIDF_LOGREG_PATH,
    prediction_column="prediction",
    note="Classical word/character TF-IDF baseline tuned on train only.",
),
```

- [ ] **Step 4: Regenerate final result tables**

Run: `.venv\Scripts\python.exe experiments\run_final_pipeline.py`

Expected: `results/final_results_table.csv` contains one test row for the baseline with 442 evaluated predictions and the exact saved metrics.

- [ ] **Step 5: Extend the main figure**

Add `TF-IDF + Logistic Regression` first in `MAIN_METHOD_ORDER`, its display label in `_format_coordinates`, and first in symbolic y coordinates. Increase the main plot height from `8.4cm` to `9.2cm`. Preserve special highlighting only for the final system.

- [ ] **Step 6: Run targeted tests**

Run: `.venv\Scripts\python.exe -m unittest tests.test_generate_thesis_result_figures -v`

Expected: all figure-data, rendering, and build tests pass.

---

### Task 2: Baseline-first Persian thesis narrative

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`

**Interfaces:**
- Consumes: baseline configuration and verified result artifacts
- Produces: methodology subsection, eighth compared method, table row, and evidence-bounded analysis

- [ ] **Step 1: Add the Chapter 4 methodology subsection**

After data preparation and before LLM prediction sources, add `\section{خط پایۀ کلاسیک}` and `\subsection{بازنمایی TF-IDF و رگرسیون لجستیک}`. State in Persian that the study first tested a low-cost classical reference. Include the combined target/sentence input, word and character ranges, balanced logistic regression, five-fold train-only tuning over `C={0.1,1,10}`, seed 42, selected `C=1`, full-train refit, and one-time test evaluation.

Add `experiments/run_tfidf_logreg_baseline.py` to the implementation-module table as the reproducible classical baseline runner.

- [ ] **Step 2: Add the Chapter 5 method and main table row**

Change «هفت پیکربندی» to «هشت پیکربندی». Add the classical baseline as item one, followed by the existing seven methods. Rename the main result section/caption language from Qwen-only to principal-method comparison and add:

```latex
خط پایۀ کلاسیک \lr{TF-IDF + Logistic Regression} &
\lr{\setpersianfont ۰/۵۴۷۵۱۱} &
\lr{\setpersianfont ۰/۵۱۴۰۸۲} \\
\hline
```

- [ ] **Step 3: Add evidence-bounded analysis**

Explain that direct Qwen exceeds the classical baseline by 13.12 Accuracy points and 16.00 Macro-F1 points, while the final system exceeds it by 17.65 and 20.50 points. Report negative-class F1 `0.374332` as the baseline weakness. Explicitly state that the baseline is classical, not a fine-tuned encoder, and that the gap cannot be attributed solely to the proposed reasoning because model families differ.

Update the figure paragraph and caption to describe eight principal methods, while leaving direct-versus-final confusion analysis unchanged.

- [ ] **Step 4: Run textual consistency scans**

Run `rg` checks for `هفت پیکربندی`, Qwen-only caption wording, the baseline name, `۰/۵۴۷۵۱۱`, `۰/۵۱۴۰۸۲`, `۱۳/۱۲`, `۱۶/۰۰`, `۱۷/۶۵`, and `۲۰/۵۰`.

Expected: no stale seven-method wording in the main comparison and all required values appear exactly where intended.

---

### Task 3: Regenerate figures, build PDF, and verify evidence

**Files:**
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_qwen_full_test_methods.pdf`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_qwen_full_test_methods.png`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: updated result table, figure generator, Chapter 4/5 LaTeX
- Produces: verified thesis PDF containing the baseline-first narrative and eight-method comparison

- [ ] **Step 1: Regenerate Chapter 5 figures**

Run: `.venv\Scripts\python.exe experiments\generate_thesis_result_figures.py`

Expected: five PDF and five PNG figures are rebuilt; the main figure contains eight labels including the classical baseline.

- [ ] **Step 2: Run all Python tests**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`

Expected: all tests pass with zero failures.

- [ ] **Step 3: Build the thesis**

Run: `powershell -ExecutionPolicy Bypass -File scripts\build_thesis.ps1`

Expected: XePersian build exits zero and produces a nonempty `AUTthesis.pdf`.

- [ ] **Step 4: Inspect LaTeX diagnostics**

Scan `AUTthesis.log` for `Undefined`, `Citation`, `LaTeX Error`, `Emergency stop`, `Fatal error`, and `Overfull`.

Expected: no undefined references/citations, fatal errors, or overfull boxes introduced by this change.

- [ ] **Step 5: Verify artifact-to-thesis consistency**

Independently recompute baseline Accuracy and Macro-F1 from `results/tfidf_logreg_predictions.csv`, compare them with `results/final_results_table.csv`, and verify the Persian rounded values in Chapter 5. Confirm the other seven test-method metrics are unchanged.

- [ ] **Step 6: Inspect the built PDF text and main figure**

Extract PDF text to confirm the baseline method and key numbers are present. Visually inspect the regenerated main comparison figure to ensure all eight labels are legible and no bars or annotations are clipped.
