# Thesis Chapter 4 Design

## Goal

Write the complete Persian chapter «کار پیشنهادی» so it connects the research gap in Chapter 3 to the experiments reserved for Chapter 5 and describes the implemented system accurately.

## Narrative

The chapter follows the data flow rather than the source-code file order. It starts from the sentence-target input, produces Direct and THOR-SC3 sources, diagnoses disagreements, records the controller's intermediate decision, and ends with the train-calibrated selector's true final output. Definitions already established in Chapter 2 and literature already reviewed in Chapter 3 are referenced briefly instead of repeated.

## Required content

- Formal task definition and notation for sentence, target, domain, split, and three-class output.
- One compact architecture figure built with existing LaTeX facilities; no class or style change.
- SCAPT/SemEval 2014 extraction, cleaning, original split preservation, and stable row alignment.
- Direct zero-shot prediction with normalized labels.
- Original-ish THOR aspect, opinion, polarity reasoning, and label stages.
- Three complete THOR runs, majority vote, and deterministic tie handling.
- Error-type-aware diagnostic output with type, proposed label, and confidence; repair of malformed output.
- Conservative controller rules, explicitly labeled as an intermediate decision.
- Final profile key `direct_prediction,error_type,diagnostic_confidence,domain`, train-only source scoring, candidate sources Direct/THOR/Diagnostic, Direct tie priority, and Direct fallback for unseen profiles.
- Modular implementation, resumability, saved intermediate outputs, and chain validation.
- Main Qwen3 8B/Ollama execution and supplementary Gemini 2.5 Flash/OpenAI-compatible execution without reporting Chapter 5 metrics.

## Writing constraints

- Use Persian digits in prose; mathematical expressions may use conventional formula digits.
- Keep model and technical identifiers in Latin script.
- Do not mention unused model families or unvalidated execution paths.
- Avoid first person, long sentences, heterogeneous lists, redundant definitions, and premature result claims.
- Introduce every figure and table in the text, provide a descriptive caption, and analyze its role.
- Keep paragraphs focused, connected by transition sentences, and end each major section with a bridge to the next section.

## Verification

- Compare the written rules against the current Python implementation and saved final-pipeline validation.
- Scan the chapter for English digits outside math and Latin technical spans.
- Compile the complete thesis with XeLaTeX/BibTeX passes.
- Require zero LaTeX errors, zero unresolved citations/references, and zero overfull horizontal boxes.
