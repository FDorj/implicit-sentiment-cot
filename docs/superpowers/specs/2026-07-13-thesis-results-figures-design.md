# Design: Thesis Chapter 5 Result Figures

## Goal

Create exactly four evidence-bearing figures for Chapter 5 of the Persian undergraduate thesis. Every figure must be generated from saved project results, use a formal print-friendly style, and add information that is harder to understand from prose alone.

The figures must not mix results evaluated on different sample sets. Qwen full-test results and the shared Qwen/Gemini subset comparison are therefore separate figures.

## Shared visual rules

- Output location: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/`
- Produce vector PDF for LaTeX and PNG previews for visual inspection.
- Use a color-blind-safe palette that remains distinguishable in grayscale.
- Keep plot-internal labels in English where they are model/method identifiers; Persian explanations and captions remain in the thesis source.
- Use Latin digits inside scientific axes and metric annotations.
- Do not place decorative titles inside the plots; LaTeX captions provide titles.
- Export with embedded fonts, tight bounding boxes, and resolution of at least 300 DPI for PNG previews.
- Do not alter the thesis class, style files, fonts, or global LaTeX configuration.

## Figure 1: Full-test Qwen method comparison

**Purpose:** Compare the principal experimental paths on the official test split and make the final selected pipeline's advantage visible.

**Scope:** Qwen3 8B, official test split, `n=442`.

**Source:** `results/final_results_table.csv`.

**Methods:**

1. Direct Qwen3 8B
2. THOR simplified
3. Simple reflection
4. ETC standard
5. THOR original-ish SC3
6. ETC over original-ish SC3
7. Final selected pipeline

**Encoding:** Horizontal grouped bars with Accuracy and Macro-F1 for each method. Use muted colors for intermediate methods and one strong accent for the final pipeline. Annotate every bar with its exact value to three decimals.

**Files:**

- `ch5_qwen_full_test_methods.pdf`
- `ch5_qwen_full_test_methods.png`

## Figure 2: Direct versus final confusion matrices

**Purpose:** Show which sentiment classes changed, not merely whether the aggregate score increased.

**Scope:** Official Qwen test split, `n=442`.

**Sources:**

- `results/direct_isa_predictions.csv`
- `results/etc_thor_originalish_sc3_selected_isa_predictions.csv`

**Encoding:** Two side-by-side row-normalized confusion matrices, ordered Negative, Neutral, Positive. Each cell displays both the raw count and row percentage. Both panels share one color scale so visual intensity is comparable.

**Validation:** Join or compare rows by stable sample keys; do not assume physical row order. Confirm identical gold labels and exactly 442 test rows before plotting.

**Files:**

- `ch5_direct_vs_final_confusion.pdf`
- `ch5_direct_vs_final_confusion.png`

## Figure 3: Selector behavior and effect relative to Direct

**Purpose:** Explain what the selector did and whether its changes were beneficial.

**Scope:** Official Qwen test split, `n=442` for both panels.

**Source:** `results/etc_thor_originalish_sc3_selected_isa_predictions.csv`.

**Encoding:** A two-panel figure.

- Left panel: selected-source counts on test: Direct `300`, THOR `142`, Diagnostic `0`. Use ordinary linear bars and annotate zero explicitly.
- Right panel: a 2x2 correctness-transition matrix comparing Direct and Final: both correct `294`, Direct wrong/Final correct `26`, Direct correct/Final wrong `6`, both wrong `116`. Highlight gain and loss cells with restrained green and red accents.

This design keeps both panels on the same test population and avoids mixing overall source counts with test-only gain/loss counts.

**Files:**

- `ch5_selector_behavior.pdf`
- `ch5_selector_behavior.png`

## Figure 4: Qwen versus Gemini on the shared subset

**Purpose:** Compare model backends fairly while separating model strength from the effect of reasoning and source selection.

**Scope:** Shared balanced test subset only, `n=90`; 15 examples for each domain/polarity combination.

**Sources:**

- `results/qwen_modelcmp_subset_metrics.txt`
- `results/qwen_modelcmp_validation_tuned_selected_metrics.txt`
- `results/gemini_qwen_modelcmp_subset_summary.csv`
- `results/gemini_modelcmp_validation_tuned_selected_metrics.txt`

**Methods and test Macro-F1:**

| Method | Qwen3 8B | Gemini 2.5 Flash |
| --- | ---: | ---: |
| Direct | 0.721903 | 0.804886 |
| THOR SC3 | 0.606450 | 0.716109 |
| Validation-tuned selected | 0.721903 | 0.808340 |

**Encoding:** Grouped vertical bars, one pair per method. Show only test Macro-F1 because the subset is balanced and this figure's purpose is model comparison, not duplication of both aggregate metrics. Annotate exact values to three decimals and state `Shared balanced test subset (n=90)` inside a small subtitle.

**Files:**

- `ch5_qwen_gemini_shared_subset.pdf`
- `ch5_qwen_gemini_shared_subset.png`

## Figure 5: Gemini Direct versus selected-profile confusion matrices

**Purpose:** Show how validation-tuned source selection changes class-level Gemini behavior even when aggregate test accuracy remains unchanged.

**Scope:** Shared balanced Gemini test subset only, `n=90`; 30 gold examples per class.

**Sources:**

- `results/gemini_modelcmp_direct_subset_predictions.csv`
- `results/gemini_modelcmp_validation_tuned_selected_predictions.csv`

**Encoding:** Reuse the exact visual grammar of the Qwen confusion figure: two side-by-side row-normalized matrices, ordered Negative, Neutral, Positive, with raw count and row percentage in every cell and a shared color scale. The left panel is `Gemini 2.5 Flash Direct`; the right panel is `Gemini selected profile`.

**Expected matrices:**

- Direct: `[[27, 2, 1], [3, 18, 9], [1, 1, 28]]`
- Selected profile: `[[27, 2, 1], [3, 20, 7], [1, 3, 26]]`

**Files:**

- `ch5_gemini_direct_vs_selected_confusion.pdf`
- `ch5_gemini_direct_vs_selected_confusion.png`

## Integration boundaries

- Create the image assets and a reproducible generation script first.
- Do not insert them into `chapter5.tex` until all five previews have been visually reviewed.
- When integrated, each figure gets a Persian caption and a unique label; surrounding prose must interpret the result rather than repeat every number.
- Existing tables can remain for exact numeric lookup, but captions and prose must avoid claiming that full-test Qwen and subset Gemini results are directly comparable.

## Verification

1. Recompute plotted values from source files and fail on row-count or alignment mismatches.
2. Confirm all five PDFs and PNGs are non-empty.
3. Inspect every PNG for clipped labels, overlapping annotations, unreadable fonts, and misleading axes.
4. Compile the thesis only after image review and integration approval.
5. Confirm no new LaTeX errors, undefined references, or overfull boxes are introduced.
