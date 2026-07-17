# Thesis Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove template placeholders and unrelated material, correct thesis terminology and claims, localize result figures, and produce a verified delivery PDF without changing experimental results.

**Architecture:** Keep the Amirkabir LaTeX scaffold intact and replace only its content-bearing files. Add repository-level textual regression tests for thesis invariants, update the existing deterministic figure generator under TDD, then rebuild all figures and the thesis and inspect both machine-readable PDF properties and rendered pages.

**Tech Stack:** XeLaTeX/XePersian, BibTeX, Python 3, `unittest`, pandas/scikit-learn (existing), TikZ/PGFPlots, Poppler tools.

## Global Constraints

- Do not modify `taid.tex` in this iteration.
- Do not rerun Qwen, Gemini, Ollama, or any LLM inference.
- Do not alter reported metrics, result CSVs, sample identifiers, or train/test membership.
- Preserve the existing dirty worktree and commit only explicitly scoped files.
- Preserve official names such as `THOR`, `SCAPT`, `SAoT`, `SC3`, `ETC`, `Qwen3 8B`, `Gemini 2.5 Flash`, and `Macro-F1` while expanding acronyms at first use.

---

### Task 1: Add thesis-content regression checks

**Files:**
- Create: `tests/test_thesis_finalization.py`

**Interfaces:**
- Consumes: UTF-8 LaTeX and bibliography files in the thesis directory.
- Produces: `ThesisFinalizationTests`, a set of content invariants used by later tasks.

- [ ] **Step 1: Write failing tests for every reported placeholder class**

Create tests that assert: personal pages contain their approved text; English title/abstract no longer contain `Title of Thesis`, `Department of ...`, `Write a 3 to 5 KeyWords`, or the sample translation; symbols contain `s`, `t`, `a`, `o`, `y`, `P`, `R`, `F_1`, `\pi`, `\mathcal{D}_{\mathrm{train}}`, and `\mathcal{D}_{\mathrm{test}}`; appendix contains `qwen3:8b`, `Ollama`, `۰/۷`, and the non-inference regeneration commands; glossaries contain project terms and no algebra/graph placeholders; `references.bib` excludes the six keys `bidabad2007classification`, `aa`, `najafi2008finsler`, `najafi`, `zakeri`, and `obradovic2023decentralized`; thesis prose excludes `original-ish`, `Macro-F۱`, and `F۱-score`; `taid.tex` matches its pre-change hash.

- [ ] **Step 2: Run the new tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_thesis_finalization -v`

Expected: failures identify the current placeholders and unrelated template content, while the `taid.tex` integrity test passes.

- [ ] **Step 3: Keep the failing suite as the acceptance contract**

Do not weaken assertions to match current content. Later tasks must make these tests pass.

### Task 2: Replace front matter, symbols, appendix, glossaries, and references

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Chant.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/acknowledgement.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/fa_title.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/en-abstract.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/en_title.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/list-of-symbols.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/appendix1.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/dicfa2en.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/dicen2fa.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/references.bib`

**Interfaces:**
- Consumes: approved Persian personal-page text and existing project-backed configuration.
- Produces: complete front/back matter with no template-domain contamination.

- [ ] **Step 1: Replace the dedication and acknowledgement**

Use the exact approved dedication and acknowledgement from `docs/superpowers/specs/2026-07-17-thesis-finalization-design.md`; do not add the removed sentence beginning with «سپاس خدای را».

- [ ] **Step 2: Complete title and abstracts**

Set `\department{مهندسی کامپیوتر}`. Translate the Persian title and abstract faithfully, use `Fatemeh Darj`, `Dr. Mostafa Haghir Chehreghani`, `July 2026`, and the five English keywords in the design. Remove the unused advisor placeholder rather than inventing an advisor.

- [ ] **Step 3: Replace symbols and dictionaries**

Define only symbols used by the thesis and add paired Persian/English entries for implicit sentiment analysis, aspect-based sentiment analysis, large language model, chain of thought, self-consistency, prompt, direct prediction, diagnostic review, source selection, confusion matrix, accuracy, precision, recall, and Macro-F1.

- [ ] **Step 4: Replace the appendix**

Document data sizes, models/backends, temperatures, SC3 count, guarded validation-tuned selection, key output files, and exact commands for `run_final_pipeline.py`, `generate_thesis_result_figures.py`, the unit suite, and `scripts/build_thesis.ps1`. State explicitly which commands reuse stored predictions and do not call an LLM.

- [ ] **Step 5: Remove unrelated bibliography records**

Delete the six complete BibTeX entries beginning at the keys listed in Task 1, preserving all project citations.

- [ ] **Step 6: Run the content suite**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_thesis_finalization -v`

Expected: front/back matter, symbol, glossary, appendix, bibliography, and `taid.tex` invariants pass; prose/figure terminology checks may remain red until later tasks.

### Task 3: Correct terminology, scope, and float placement

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter1.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter2.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter3.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`

**Interfaces:**
- Consumes: existing citations `fei2023thor`, `li2021scapt`, `duan2024isa`, and per-class F1 table `tab:ch5-class-f1`.
- Produces: academically scoped prose and a compact Qwen/Gemini results layout.

- [ ] **Step 1: Expand acronyms at first use**

Introduce THOR as the method from “Reasoning Implicit Sentiment with Chain-of-Thought Prompting”, SCAPT as “Supervised Contrastive Pre-Training”, SAoT with the paper-backed full name used by Duan and Wang, SC3 as three-run self-consistency, and ETC as error-type-aware correction/control before later shorthand uses.

- [ ] **Step 2: Fix typography and informal naming**

Replace `Macro-F۱` with `Macro-F1`, `F۱-score` with `F1-score`, and all thesis-facing `original-ish` strings with «سازگارشدۀ سه‌اجرایی» or `THOR SC3` as appropriate.

- [ ] **Step 3: Scope scientific claims**

End the Persian and English abstracts with a claim limited to the model, data, and experimental setting. Replace the inference from close Accuracy/Macro-F1 values with a direct reference to `tab:ch5-class-f1`, which shows all three class F1 values increased.

- [ ] **Step 4: Reflow the model-comparison figures**

Place the shared-subset comparison and Gemini confusion figure as a compact consecutive block with `[!htbp]`, widths chosen from rendered evidence, and a page barrier only if needed; avoid forcing either figure onto a mostly empty page.

- [ ] **Step 5: Run content tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_thesis_finalization -v`

Expected: prose-related assertions pass.

### Task 4: Hide link colors and add PDF metadata

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/commands.tex`
- Test: `tests/test_thesis_finalization.py`

**Interfaces:**
- Consumes: `hyperref` already required by `cleveref` and internal references.
- Produces: visually print-safe links and populated PDF document properties.

- [ ] **Step 1: Add failing metadata/link-style assertions**

Assert `commands.tex` uses `hidelinks`, contains `pdftitle`, `pdfauthor`, `pdfsubject`, and `pdfkeywords`, and no longer contains `linkcolor=blue` or `citecolor=red`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_thesis_finalization.ThesisFinalizationTests.test_pdf_links_and_metadata -v`

Expected: failure reports visible link colors and missing metadata.

- [ ] **Step 3: Apply the minimal hyperref configuration**

Load `hyperref` once with `pagebackref=false,hidelinks` and add Persian title/author/subject/keyword metadata through `\hypersetup`. Do not claim PDF/UA tagging.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the same focused command; expected: PASS.

### Task 5: Localize generated figures under TDD

**Files:**
- Modify: `tests/test_generate_thesis_result_figures.py`
- Modify: `experiments/generate_thesis_result_figures.py`

**Interfaces:**
- Consumes: unchanged result CSVs and existing `render_figure_tex`/`build_figures` interfaces.
- Produces: the same five PDF/PNG assets with Persian or bilingual display labels.

- [ ] **Step 1: Change rendering expectations to the new academic labels**

Require the generated TeX to include Persian/bilingual axis and panel labels, `THOR SC3` instead of `original-ish`, and defined first-use captions in chapter text. Keep official model names Latin.

- [ ] **Step 2: Run rendering tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_generate_thesis_result_figures.ThesisResultFigureRenderingTests -v`

Expected: failures show the old English-only labels.

- [ ] **Step 3: Add Persian support to standalone figure documents**

Load `xepersian` after PGFPlots, set the bundled B Nazanin and Times fonts through paths rooted at the repository, and replace human-readable labels while leaving internal method keys unchanged.

- [ ] **Step 4: Run rendering tests and verify GREEN**

Run the focused rendering suite; expected: PASS.

- [ ] **Step 5: Build all assets**

Run: `.\.venv\Scripts\python.exe -B experiments\generate_thesis_result_figures.py`

Expected: five nonempty PDFs and five PNG previews under `Images/Chapter5`.

### Task 6: Full verification and visual review

**Files:**
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`
- Inspect: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.log`

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: final verified PDF and an evidence-backed handoff.

- [ ] **Step 1: Run the full unit suite**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Build the thesis**

Run: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_thesis.ps1`

Expected: exit code 0 and a nonempty `AUTthesis.pdf`; the build script rejects fatal LaTeX errors, unresolved citations/references, and overfull hboxes.

- [ ] **Step 3: Inspect PDF properties and extracted text**

Run `pdfinfo` and `pdftotext`; verify Title/Author are populated and search the extracted text for every removed placeholder and unrelated term. Confirm `taid.tex` remains unchanged.

- [ ] **Step 4: Inspect rendered pages**

Render the dedication, acknowledgement, symbol list, model comparison, appendix, dictionaries, English abstract, and English title pages to PNG. Confirm no clipping, unreadable bidi text, colored links, excessive blank pages, or isolated comparison float.

- [ ] **Step 5: Review the final diff**

Run `git diff --check` and a scoped `git diff --stat`. Report all modified files, verification counts, PDF page count, metadata, and the two explicit follow-ups: signed defense form and administrative confirmation of the department field.
