# Thesis Chapter 5 Result Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate four reproducible, scientifically accurate Chapter 5 figures as vector PDFs and PNG previews from the repository's saved result files.

**Architecture:** A single focused Python script reads and validates CSV results, computes plot-ready summaries, emits four standalone PGFPlots/TikZ documents, compiles them with XeLaTeX, and converts PDFs to PNG using `pdftocairo`. Pure data functions are separated from rendering and command execution so their behavior can be tested without invoking LaTeX.

**Tech Stack:** Python 3.13 standard library, pandas, scikit-learn, PGFPlots/TikZ, XeLaTeX, pdftocairo, unittest.

## Global Constraints

- Generate exactly four evidence-bearing Chapter 5 figures.
- Keep full-test Qwen results separate from the shared Qwen/Gemini subset.
- Output vector PDF plus PNG preview for every figure.
- Do not modify `AUTthesis.cls`, style files, fonts, or global LaTeX configuration.
- Do not insert figures into `chapter5.tex` before visual review.
- Use the existing saved result files; do not rerun language-model inference.

---

### Task 1: Validated result summaries

**Files:**
- Create: `experiments/generate_thesis_result_figures.py`
- Create: `tests/test_generate_thesis_result_figures.py`

**Interfaces:**
- Produces: `load_main_test_results(repo_root) -> list[dict]`
- Produces: `load_confusion_data(repo_root) -> dict`
- Produces: `load_selector_behavior(repo_root) -> dict`
- Produces: `load_shared_subset_comparison(repo_root) -> dict`

- [ ] **Step 1: Write failing tests for exact saved-result summaries**

Tests assert seven full-test methods with `n=442`, 442 aligned Direct/final rows, selector counts `{direct: 300, thor: 142, diagnostic: 0}`, correctness transitions `{both_correct: 294, gain: 26, loss: 6, both_wrong: 116}`, and the six shared-subset Macro-F1 values documented in the design.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.test_generate_thesis_result_figures -v`

Expected: collection/import failure because `experiments.generate_thesis_result_figures` does not exist.

- [ ] **Step 3: Implement minimal loading, alignment, and validation functions**

Use stable sample keys `(id, source_sentence_id, sentence, target, from, to, domain, split)`, reject duplicates or mismatched gold labels, filter every comparison to the intended split, and calculate metrics from prediction CSVs where possible.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m unittest tests.test_generate_thesis_result_figures -v`

Expected: all focused tests pass.

### Task 2: Standalone vector figure generation

**Files:**
- Modify: `experiments/generate_thesis_result_figures.py`
- Modify: `tests/test_generate_thesis_result_figures.py`

**Interfaces:**
- Produces: `render_figure_tex(repo_root, output_dir) -> list[pathlib.Path]`
- Produces: four standalone `.tex` sources named after the four final assets.

- [ ] **Step 1: Add failing tests for the four render documents**

Tests require four files, `standalone` and `pgfplots` declarations, exact plot identifiers, fully resolved numeric values, and fixed axis ranges that do not truncate annotations.

- [ ] **Step 2: Run the rendering tests and verify RED**

Run: `python -m unittest tests.test_generate_thesis_result_figures -v`

Expected: failure because `render_figure_tex` is not implemented.

- [ ] **Step 3: Implement the four renderers**

Render:

1. Horizontal grouped bars for Accuracy and Macro-F1 across seven full-test Qwen methods.
2. Two row-normalized confusion matrices with count and percentage annotations.
3. Test source-selection bars plus a 2x2 Direct/final correctness-transition matrix.
4. Grouped Macro-F1 bars for Qwen3 8B and Gemini 2.5 Flash on the shared 90-example test subset.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m unittest tests.test_generate_thesis_result_figures -v`

Expected: all focused tests pass.

### Task 3: PDF and PNG build pipeline

**Files:**
- Modify: `experiments/generate_thesis_result_figures.py`
- Modify: `tests/test_generate_thesis_result_figures.py`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_qwen_full_test_methods.pdf`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_qwen_full_test_methods.png`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_direct_vs_final_confusion.pdf`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_direct_vs_final_confusion.png`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_selector_behavior.pdf`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_selector_behavior.png`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_qwen_gemini_shared_subset.pdf`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_qwen_gemini_shared_subset.png`

**Interfaces:**
- Produces: `build_figures(repo_root, output_dir) -> list[pathlib.Path]`

- [ ] **Step 1: Add a failing integration test for expected build products**

The test invokes the build in a temporary directory and requires four non-empty PDFs and four non-empty PNGs.

- [ ] **Step 2: Run the integration test and verify RED**

Run: `python -m unittest tests.test_generate_thesis_result_figures -v`

Expected: failure because the build command is absent.

- [ ] **Step 3: Implement compilation and conversion**

Run XeLaTeX in nonstop/halt-on-error mode for each standalone source and convert each PDF with `pdftocairo -png -singlefile -r 300`. Surface command output on failure and remove only temporary LaTeX auxiliary files created by this generator.

- [ ] **Step 4: Run focused and full regression tests**

Run: `python -m unittest tests.test_generate_thesis_result_figures -v`

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

### Task 4: Visual and artifact verification

**Files:**
- Inspect: all four generated PNG previews.
- Inspect: all four generated PDF assets.

- [ ] **Step 1: Generate the final assets**

Run: `python experiments/generate_thesis_result_figures.py`

Expected: eight final assets reported, with four PDFs and four PNGs.

- [ ] **Step 2: Validate artifact signatures and dimensions**

Confirm every PDF begins with `%PDF`, every PNG begins with the PNG signature, no file is empty, and all preview dimensions are large enough for thesis inspection.

- [ ] **Step 3: Visually inspect every preview**

Check clipping, overlaps, font substitution, axis integrity, annotation readability, consistent color meaning, and whether the final pipeline is visually emphasized without distorting scale.

- [ ] **Step 4: Re-run the complete focused verification**

Run: `python -m unittest tests.test_generate_thesis_result_figures -v`

Run: `python experiments/generate_thesis_result_figures.py`

Expected: zero failures and deterministic regeneration of the same eight final assets.

### Task 5: Gemini confusion-matrix companion figure

**Files:**
- Modify: `experiments/generate_thesis_result_figures.py`
- Modify: `tests/test_generate_thesis_result_figures.py`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_gemini_direct_vs_selected_confusion.pdf`
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_gemini_direct_vs_selected_confusion.png`

**Interfaces:**
- Produces: `load_gemini_confusion_data(repo_root) -> dict`
- Extends: `render_figure_tex(repo_root, output_dir)` and `build_figures(repo_root, output_dir)` to five figures and ten assets.

- [ ] **Step 1: Write failing data and rendering tests**

Assert 90 aligned test rows, 30 gold examples per class, exact Direct and selected-profile confusion matrices, the fifth standalone source name, and the two required panel titles.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m unittest tests.test_generate_thesis_result_figures -v`

Expected: import or assertion failure because the Gemini confusion loader and fifth figure are absent.

- [ ] **Step 3: Implement aligned loading and reuse the existing matrix renderer**

Read the two Gemini prediction files, align stable sample keys, verify gold labels and saved Direct predictions, compute matrices in Negative/Neutral/Positive order, and render them with the same shared normalization and colors as the Qwen figure.

- [ ] **Step 4: Build and visually inspect PDF and PNG**

Run: `python experiments/generate_thesis_result_figures.py`

Expected: ten assets reported; the new preview has two unclipped matrices and explicitly states the shared balanced test subset size `n=90`.

- [ ] **Step 5: Run final regression verification**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass, including the new Gemini matrix tests.
