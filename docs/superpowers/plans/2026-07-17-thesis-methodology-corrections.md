# Thesis Methodology Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the guarded validation-tuned selector the reproducible primary pipeline and align parser behavior, data validation, figures, and Persian thesis prose with the implementation.

**Architecture:** Keep existing public interfaces and result files. Tighten two shared utilities through test-first changes, point the final-results pipeline at the already-generated guarded output, regenerate derived tables/figures, and edit only the relevant thesis paragraphs.

**Tech Stack:** Python 3, unittest, pandas, scikit-learn, XeLaTeX, Persian LaTeX template.

## Global Constraints

- Preserve all unrelated uncommitted thesis edits.
- Do not call an LLM or regenerate stochastic predictions.
- Use `results/etc_thor_originalish_sc3_guarded_tuned_selected_isa_predictions.csv` as the primary final prediction artifact.
- Keep the unguarded selector visible only as an ablation.
- Modify production Python only after observing the corresponding new test fail.

---

### Task 1: Strict label normalization

**Files:**
- Modify: `tests/test_core_logic.py`
- Modify: `src/utils.py`

**Interfaces:**
- Consumes: arbitrary model output text.
- Produces: `normalize_label(text: str) -> str`, returning one of `positive`, `negative`, `neutral`, or `unknown`.

- [ ] Add assertions that `normalize_label("positive or negative")`, `normalize_label("nonpositive")`, and `normalize_label("negative, then neutral")` return `unknown`, while ordinary single-label prose still works.
- [ ] Run `python -m unittest tests.test_core_logic.CoreLogicTests.test_normalize_label_handles_extra_text -v`; expect failure on ambiguous substring inputs.
- [ ] Replace substring priority logic with:

```python
import re

VALID_OUTPUT_LABELS = ("positive", "negative", "neutral")

def normalize_label(text: str) -> str:
    if not text:
        return "unknown"
    normalized = " ".join(str(text).strip().lower().split())
    matches = {
        label
        for label in VALID_OUTPUT_LABELS
        if re.search(rf"\b{re.escape(label)}\b", normalized)
    }
    return next(iter(matches)) if len(matches) == 1 else "unknown"
```

- [ ] Re-run the focused test; expect PASS.

### Task 2: Fail-fast required-field validation

**Files:**
- Create: `tests/test_data_loader.py`
- Modify: `src/data_loader.py`

**Interfaces:**
- Produces: `validate_required_fields(df: pd.DataFrame, required_columns: tuple[str, ...] = REQUIRED_SAMPLE_COLUMNS) -> None`.
- Required fields: `id`, `source_sentence_id`, `domain`, `split`, `sentence`, `target`, `from`, `to`, `polarity`, `is_implicit`.

- [ ] Add tests proving a complete frame is accepted, a missing required column raises `ValueError`, and a null required value raises `ValueError` containing the column and count.
- [ ] Run `python -m unittest tests.test_data_loader -v`; expect import failure because the validator does not exist.
- [ ] Add the required-column constant and validator; call it on `df_all` immediately after concatenating parsed XML frames and before saving/filtering.
- [ ] Re-run `python -m unittest tests.test_data_loader -v`; expect all tests PASS.

### Task 3: Promote guarded output in reproducible results and figures

**Files:**
- Modify: `tests/test_generate_thesis_result_figures.py`
- Modify: `experiments/run_final_pipeline.py`
- Modify: `experiments/generate_thesis_result_figures.py`
- Regenerate: `results/final_results_table.csv`
- Regenerate: `results/final_results_table.md`
- Regenerate: `results/final_pipeline_validation.txt`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/*`

**Interfaces:**
- Primary final file: `etc_thor_originalish_sc3_guarded_tuned_selected_isa_predictions.csv`.
- Expected test metrics: Accuracy `0.7239819004524887`, Macro-F1 `0.719119` (full precision computed from CSV).
- Expected confusion matrix: `[[70, 9, 3], [51, 160, 37], [3, 19, 90]]`.
- Expected selected-source counts: direct `409`, THOR `33`, diagnostic `0`.

- [ ] Change figure tests to the guarded expectations above.
- [ ] Run the focused figure-data tests; expect failures because loaders still point at the unguarded result.
- [ ] Change `FINAL_SELECTED_PATH` and `_load_aligned_direct_final()` to the guarded-tuned artifact and replace the final-method note with validation-tuned guarded wording.
- [ ] Run `python experiments/run_final_pipeline.py` to regenerate the final summary tables.
- [ ] Run the focused figure tests; expect PASS.
- [ ] Run `python experiments/generate_thesis_result_figures.py` to regenerate the five Chapter 5 figure pairs.

### Task 4: Align thesis method, claims, and figures

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter4/ch4_proposed_pipeline.tex`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter4/ch4_proposed_pipeline.pdf`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Required prose outcomes:**
- Parser prose says whole-word extraction accepts exactly one unique label; ambiguous or absent labels become `unknown`.
- Data prose says required fields were checked and no null required rows were found; it describes the composite key rather than unique IDs.
- THOR prose states reasoning stages use 0.7 and label conversion uses 0.0.
- Polarity equation uses `P_R(s,t,o)` and states aspect contributes indirectly through opinion.
- The primary selector profile is `[direct, THOR, error_type, confidence, domain]` with support 10, direct margin 2, second-source margin 0, and relative gain 0.05 selected over validation seeds 0--9.
- Test gold is used only after configuration freeze to compute metrics.
- Add a learned-selector subsection covering oracle target priority, features, 75/25 stratified internal split with seed 42, candidate models, validation Macro-F1 selection, refit, fallback, and multiple-correct-source limitation.
- Qwen/Gemini text refers to model/backend choice rather than capacity.
- Official-THOR wording calls whole-chain SC3 a project-specific adaptation, not a near reproduction.
- Primary test text uses Accuracy `0.723982`, Macro-F1 `0.719119`, and source counts 409/33/0; unguarded `0.719204` remains only in the ablation discussion.

- [ ] Apply the precise prose replacements and update the Chapter 4 diagram training box to mention internal validation and guarded fallback.
- [ ] Build the Chapter 4 diagram with XeLaTeX.
- [ ] Search all thesis `.tex` files for superseded calibration/capacity wording and old primary metrics; classify any remaining occurrence as intentional ablation or fix it.
- [ ] Run the full Python suite: `python -m unittest discover -s tests -v`.
- [ ] Audit stored direct and THOR raw outputs with the strict parser and require zero prediction changes.
- [ ] Build the thesis with `powershell -ExecutionPolicy Bypass -File .\scripts\build_thesis.ps1`.
- [ ] Check the LaTeX log for undefined references/citations and review `git diff --check` plus the final diff summary.
