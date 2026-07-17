# Thesis Methodology Corrections Design

**Date:** 2026-07-17

**Goal:** Align the Persian thesis, the implemented selection policy, and the supporting Python behavior without rerunning any language model.

## Scope

1. Promote the validation-tuned guarded policy to the primary final selection method.
2. Describe the five-field guarded profile, its validation-only tuning procedure, and the selected thresholds.
3. Update the primary Chapter 5 results and figures to the guarded output; retain the unguarded policy as an ablation.
4. Add a method subsection for the decision-tree meta-selector and logistic source ranker, including targets, features, internal split, hyperparameters, selection metric, refit, fallback, and multiple-correct-source limitation.
5. Replace causal claims about Qwen/Gemini model capacity with claims about model/backend choice in the observed experimental setting.
6. Make label parsing strict: use whole-word matching, accept exactly one unique valid label, and map absent or ambiguous labels to `unknown`.
7. Add regression tests for strict parsing and audit stored raw outputs to confirm reparsing does not change saved predictions.
8. State that THOR reasoning stages use temperature 0.7 while the final label-conversion stage uses temperature 0.0; do not change generation behavior.
9. Change the polarity-reasoning formula to `P_R(s,t,o)` and explain that aspect affects the final decision indirectly through opinion extraction rather than as an explicit polarity-prompt input.
10. Replace the claim that incomplete rows were removed with the verified claim that required fields were checked and no missing values were found.
11. Add fail-fast required-field validation to data loading, with tests for missing values.
12. Replace the claim that row IDs are unique with an explanation of the composite alignment key `[id, source_sentence_id, sentence, target, from, to, polarity]`; do not renumber IDs.
13. Replace calibration terminology in scientific prose with training-tuned policy and performance profile terminology.
14. Replace wording that implies near reproduction of official THOR with a concise statement that whole-chain SC3 is a project-specific THOR-inspired adaptation.
15. Preserve the claim that test gold labels are excluded from policy learning and threshold selection; clarify that test labels are read only after configuration is fixed to compute final metrics.

## Primary Guarded Policy

The primary selector is the configuration chosen by mean validation Macro-F1 and then validation accuracy over ten internal training-data splits with seeds 0 through 9. It uses the richer five-field profile:

```text
[direct_prediction, thor_prediction, error_type, diagnostic_confidence, domain]
```

The selected guarded thresholds are:

- minimum profile support: 10
- minimum advantage over the direct default: 2 correct training examples
- minimum advantage over the second-best source: 0
- minimum relative gain over the direct default: 0.05

If the profile fails a guard, the selector falls back to the direct prediction. The primary test results are Accuracy `0.723982` and Macro-F1 `0.719119`, with 409 direct selections and 33 THOR selections. The previous unguarded policy remains visible only as an ablation.

## Code Design

### Strict label parsing

`src.utils.normalize_label` remains the single public normalization function. It will normalize case and whitespace, find valid labels using whole-word regular expressions, and return the label only when the set of matched valid labels has size one. Otherwise it returns `unknown`.

### Required-field validation

`src.data_loader` will expose a small validation function that checks the required columns and rejects rows containing missing values in those columns. Existing dataset construction will call this function before returning the prepared dataset. The validation will report the affected column names and row count.

## Thesis Design

- Chapter 4 defines the guarded selector as the final method and places learned alternative selectors before results.
- Chapter 5 uses guarded metrics as the primary result and labels the unguarded selector as an ablation.
- Chapter 6 uses guarded numbers and avoids causal capacity claims.
- Existing university-template structure and unrelated user edits remain intact.

## Verification

1. New parser tests must fail against the old substring implementation and pass after the strict parser is implemented.
2. New missing-field tests must fail before loader validation and pass after implementation.
3. The full Python test suite must pass.
4. Stored raw direct and THOR outputs must be reparsed and compared with stored predictions; any changed row must be reported rather than silently accepted.
5. Thesis source must be searched for superseded terms and old primary metrics.
6. The thesis PDF must build successfully and the build log must contain no undefined references caused by these changes.

## Explicit Exclusions

- No new LLM calls or stochastic THOR repetitions.
- No statistical-significance implementation in this change.
- No new TF-IDF, encoder, or external baseline experiment.
- No broad literature expansion.
- No new reproducibility inventory.
- No large THOR-versus-official-THOR comparison section.
