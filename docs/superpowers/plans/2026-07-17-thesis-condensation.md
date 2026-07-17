# Thesis Condensation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the complete thesis PDF from 100 pages to 60–65 pages while preserving every required template section, the scientific argument, verified results, and the unchanged defense-approval source.

**Architecture:** Condense the thesis in three content passes: foundations, method, and evaluation/conclusion. Each pass removes duplication and merges overlapping sections without changing the university typography. A PDF page-count acceptance test and source-anchor tests protect the target range and required scientific content; each pass ends with a thesis build and a measured page budget.

**Tech Stack:** XeLaTeX/XePersian, BibTeX, PowerShell build script, Python `unittest`, Poppler `pdfinfo`/`pdftoppm`, Git.

## Global Constraints

- The complete PDF, including front matter and back matter, must contain 60–65 pages.
- Preserve all front matter, dedication, acknowledgement, defense-approval source, references, appendix, both glossaries, and English abstract.
- Do not change the university font size, margins, or standard line spacing.
- Do not change experimental results, dataset counts, method names, formulas required to define the method, or evidence-backed claims.
- Preserve the main results table, class-level F1 evidence, domain analysis, selector analysis, qualitative errors, and the Gemini comparison.
- `taid.tex` must retain SHA-256 `8212555f994bea6aae5976199921b1cc55c925063440dd862e5cc3d4ac9adab8`.
- Do not run the LLM or generate new experimental outputs.
- Build output must have no LaTeX errors, undefined references/citations, or overfull boxes.

---

## File Structure

- Create `tests/test_thesis_condensation.py`: page-count and required-content acceptance checks.
- Modify `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter1.tex`: compact research framing.
- Modify `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter2.tex`: compact definitions and metrics.
- Modify `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter3.tex`: focused related work.
- Modify `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`: compact method while preserving equations and architecture.
- Modify `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`: compact result interpretation and qualitative analysis.
- Modify `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex`: compact conclusions, limitations, and future work.
- Modify `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/appendix1.tex` only if the total remains above 65 pages after chapter condensation; preserve every reproducibility category.
- Modify chapter figure widths/float options only when a measured half-empty page remains; do not regenerate experimental data.
- Regenerate `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf` after each content pass.

---

### Task 1: Add the page-budget acceptance test

**Files:**
- Create: `tests/test_thesis_condensation.py`
- Reference: `tests/test_thesis_finalization.py`
- Test artifact: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: the built thesis PDF and UTF-8 LaTeX chapter sources.
- Produces: `ThesisCondensationTests`, enforcing total pages, required section anchors, result anchors, and the unchanged defense page.

- [ ] **Step 1: Write the failing total-page test**

Create a `unittest.TestCase` that runs `pdfinfo` on `AUTthesis.pdf`, extracts `Pages:`, and asserts `60 <= pages <= 65`. Include an explicit failure message containing the observed count.

```python
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir"


class ThesisCondensationTests(unittest.TestCase):
    def test_complete_pdf_has_60_to_65_pages(self):
        completed = subprocess.run(
            ["pdfinfo", str(THESIS / "AUTthesis.pdf")],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        match = re.search(r"^Pages:\s+(\d+)$", completed.stdout, re.MULTILINE)
        self.assertIsNotNone(match, completed.stdout)
        pages = int(match.group(1))
        self.assertGreaterEqual(pages, 60, f"Thesis is too short: {pages} pages")
        self.assertLessEqual(pages, 65, f"Thesis is too long: {pages} pages")
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_thesis_condensation.ThesisCondensationTests.test_complete_pdf_has_60_to_65_pages -v
```

Expected: FAIL with `Thesis is too long: 100 pages`.

- [ ] **Step 3: Add required-content and integrity tests**

Add tests that read the six chapter files and assert the presence of these anchors:

```python
required = {
    "chapter1.tex": ["پرسش‌های پژوهش", "نوآوری‌ها"],
    "chapter2.tex": ["تحلیل احساسات ضمنی", "Macro-F1"],
    "chapter3.tex": ["SCAPT", "THOR", "SAoT", "شکاف تحقیقاتی"],
    "chapter4.tex": ["P_R", "اعتبارسنجی داخلی", "پیش‌بینی نهایی"],
    "chapter5.tex": ["۰/۷۲۳۹۸۲", "۰/۷۱۹۱۱۹", "Gemini 2.5 Flash"],
    "chapter6.tex": ["محدودیت‌های پژوهش", "کارهای آتی"],
}
```

Also copy the SHA-256 integrity assertion for `taid.tex` from `tests/test_thesis_finalization.py`. These tests should pass before prose editing.

- [ ] **Step 4: Run integrity tests**

Run:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_thesis_condensation -v
```

Expected: only the page-count test fails; every content and integrity check passes.

- [ ] **Step 5: Commit the RED acceptance test**

```powershell
git add tests/test_thesis_condensation.py
git commit -m "test: define condensed thesis page budget"
```

---

### Task 2: Condense the introduction and foundations to 11 pages

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter1.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter2.tex`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: research framing and terminology used by Chapters 3–6.
- Produces: Chapter 1 at approximately 5 pages and Chapter 2 at approximately 6 pages, with all later terminology introduced once.

- [ ] **Step 1: Condense Chapter 1 into six sections**

Retain this exact section flow:

1. `بیان مسئله و اهمیت پژوهش`
2. `اهداف پژوهش`
3. `پرسش‌های پژوهش`
4. `نوآوری‌ها و دستاوردها`
5. `روش کلی پژوهش`
6. `ساختار پایان‌نامه`

Merge the current opening, problem statement, and motivation. Limit the objectives to one primary objective plus four concise secondary objectives. Keep all research questions verbatim. Describe the pipeline in one paragraph and refer readers to Chapter 4 instead of repeating component details. Limit the thesis structure to one sentence per chapter.

- [ ] **Step 2: Condense Chapter 2 into six conceptual sections**

Retain this exact section flow:

1. `تحلیل احساسات، جنبه و قطبیت`
2. `تحلیل احساسات ضمنی و چالش‌های آن`
3. `مدل‌های زبانی و مهندسی پرامپت`
4. `استدلال چندمرحله‌ای، بازبینی و خودسازگاری`
5. `کنترل‌گر و انتخاب چندمنبعی`
6. `معیارهای ارزیابی`

Remove generic textbook examples and repeated transitions. Keep one explicit-versus-implicit example. Combine Accuracy, Precision, Recall, F1, and Macro-F1 into one metrics section, retaining only the formulas used later. Keep the class-imbalance rationale for Macro-F1.

- [ ] **Step 3: Build and measure the first pass**

Run:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_thesis.ps1
pdfinfo .\قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir\AUTthesis.pdf
```

Expected: build succeeds; Chapter 2 begins at internal page 6 or 7 and Chapter 3 begins at internal page 12 or 13. Record the observed page boundaries in the plan checklist.

- [ ] **Step 4: Run thesis integrity tests**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_thesis_finalization tests.test_thesis_condensation -v
```

Expected: all tests except the total-page target pass.

- [ ] **Step 5: Commit the foundations pass**

```powershell
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter1.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter2.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf
git commit -m "docs: condense thesis introduction and foundations"
```

---

### Task 3: Condense related work to seven pages

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter3.tex`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: definitions established once in Chapter 2.
- Produces: a seven-page evidence-focused literature review leading directly to the research gap.

- [ ] **Step 1: Replace the broad history with a focused opening**

Compress lexical, classical ML, neural, and language-model history into two paragraphs. Do not redefine sentiment analysis, prompting, chain of thought, or self-consistency. Use citations already present in the chapter.

- [ ] **Step 2: Organize the evidence into four sections**

Use this structure:

1. `تحلیل احساسات ضمنی و مجموعه‌داده‌ها`
2. `استدلال و بازبینی در روش‌های مولد`
3. `THOR، SAoT و خودسازگاری`
4. `انتخاب منبع و شکاف تحقیقاتی`

Keep the comparison table. Remove prose that repeats every table cell; follow the table with one paragraph contrasting inference structure, correction mechanism, and source selection. End with the exact gap addressed by the guarded training-tuned selector.

- [ ] **Step 3: Build and measure Chapter 3**

Run the thesis build and inspect `AUTthesis.toc`.

Expected: Chapter 4 begins no later than internal page 20. The total PDF will still exceed 65 pages at this checkpoint.

- [ ] **Step 4: Run integrity tests and commit**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_thesis_finalization tests.test_thesis_condensation -v
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter3.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf
git commit -m "docs: focus thesis related work"
```

Expected: only the total-page test may remain failing.

---

### Task 4: Condense the proposed method to nine pages

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`
- Reference: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/appendix1.tex`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: the compact terminology and research gap.
- Produces: a self-contained method definition preserving the architecture figure, decision equations, train-only tuning, fallback behavior, and reproducibility boundary.

- [ ] **Step 1: Keep the method spine and merge supporting sections**

Use this structure:

1. `صورت‌بندی مسئله و نمای کلی سامانه`
2. `تولید منابع پیش‌بینی`
3. `بازبینی آگاه از نوع خطا و کنترل‌گر میانی`
4. `انتخاب منبع محافظت‌شده و تنظیم‌شده با دادهٔ آموزش`
5. `انتخاب‌گرهای جایگزین`
6. `پیاده‌سازی و جمع‌بندی`

Keep the architecture figure. Merge data preparation and TF-IDF baseline into the opening section. Preserve `P_R(s,t,o)`, the SC3 majority equation, profile definition, train source-selection equation, guarded thresholds, final-selection equation, and fallback rule. Keep the error taxonomy table but reduce the accompanying prose to the correctable/non-correctable distinction.

- [ ] **Step 2: Move implementation inventory out of Chapter 4**

Remove the software file inventory and command-level execution detail from Chapter 4 because Appendix 1 already contains them. Retain one paragraph stating that stages store aligned CSV outputs, support resume, and validate the composite sample key.

- [ ] **Step 3: Build and verify the method page budget**

Expected after build: Chapter 5 begins no later than internal page 29. Check that the architecture figure and both method tables remain readable.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_thesis_finalization tests.test_thesis_condensation -v
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf
git commit -m "docs: condense proposed thesis method"
```

Expected: required equation and train/test integrity anchors pass.

---

### Task 5: Condense results and conclusion to fourteen pages

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: stored result tables and figures without recomputation.
- Produces: a ten-page results chapter and four-page conclusion preserving the complete evidence chain.

- [ ] **Step 1: Condense Chapter 5 around evidence rather than repeated numbers**

Use this structure:

1. `تنظیمات و معیارهای آزمایش`
2. `نتایج اصلی و تحلیل کلاسی و دامنه‌ای`
3. `تحلیل انتخاب‌گر و مؤلفه‌ها`
4. `تحلیل کیفی و خطا`
5. `مقایسهٔ تکمیلی با Gemini 2.5 Flash`
6. `جمع‌بندی`

Keep all four existing thesis figures and the six evidence tables unless two tables report the same values. Remove prose that enumerates every table row. For each table or figure, retain one paragraph answering: what changed, which class/domain/source explains it, and what limitation remains. Keep two corrected examples and two weakened/error examples representing distinct mechanisms. Move the consolidated limitations discussion to Chapter 6.

- [ ] **Step 2: Condense Chapter 6 into five sections**

Use this structure:

1. `جمع‌بندی پژوهش`
2. `پاسخ به پرسش‌های پژوهش`
3. `دستاوردهای اصلی`
4. `محدودیت‌ها و کارهای آتی`
5. `نتیجه‌گیری نهایی`

Answer each research question in one paragraph. Merge scientific and engineering contributions into one list. State each limitation once. Replace the six future-work subsections with one prioritized list covering repeated inference runs, independent datasets/models, selector improvement, targeted review, human evaluation, and cost reduction.

- [ ] **Step 3: Build and assess the complete page count**

Run the full thesis build and the page-count test.

Expected: PDF contains 60–68 pages. If it is 60–65, proceed directly to Task 7. If it is 66–68, execute Task 6. If it is below 60, restore the most important removed explanation in Chapter 4 or Chapter 5 rather than adding spacing.

- [ ] **Step 4: Run the full test suite and commit**

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf
git commit -m "docs: condense thesis results and conclusion"
```

Expected: all tests pass if the page target is met; otherwise only the page-count test fails with the measured count.

---

### Task 6: Remove the final one to three excess pages, only if measured

**Files:**
- Modify if needed: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/appendix1.tex`
- Modify if needed: figure placement options in `chapter4.tex` or `chapter5.tex`
- Regenerate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: a scientifically complete 66–68-page draft.
- Produces: a 60–65-page draft without typography changes or content loss.

- [ ] **Step 1: Identify physical half-empty pages**

Render the PDF at low resolution:

```powershell
pdftoppm -png -r 80 -f 1 -l 68 .\قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir\AUTthesis.pdf .\.test_tmp\condensed
```

Inspect chapter boundaries, float-only pages, Appendix 1, and glossary transitions. Do not tighten pages already visually dense.

- [ ] **Step 2: Apply only measured layout/content corrections**

In order:

1. Change `[!htbp]` float placement or reduce a figure from at most `0.94\textwidth` to no less than `0.84\textwidth` when this removes a float-only page and labels remain readable.
2. Convert Appendix 1 command lines into one compact `LTR` block while preserving data, model, policy, output-file, and command categories.
3. Remove a repeated chapter summary sentence already stated in the next chapter introduction.

Do not change global font, margins, geometry, line spacing, or glossary entries.

- [ ] **Step 3: Rebuild after each correction**

Stop as soon as the total reaches 60–65 pages. Run `tests.test_thesis_condensation` after every build.

- [ ] **Step 4: Commit measured final compression**

```powershell
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/appendix1.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf
git commit -m "docs: meet final thesis page budget"
```

Only add files actually changed.

---

### Task 7: Verify coherence, rendering, and repository state

**Files:**
- Verify: all six chapter sources, appendix, glossaries, PDF, tests.
- Modify only if verification exposes a concrete defect.

**Interfaces:**
- Consumes: the 60–65-page candidate PDF.
- Produces: verified final thesis and a clean Git history.

- [ ] **Step 1: Run the full automated suite**

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
```

Expected: all tests pass, including page count and `taid.tex` integrity.

- [ ] **Step 2: Run a fresh three-pass thesis build**

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_thesis.ps1
```

Expected: exit code 0; no LaTeX errors, undefined references/citations, or overfull boxes.

- [ ] **Step 3: Verify PDF metadata and exact total**

```powershell
pdfinfo .\قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir\AUTthesis.pdf
```

Expected: `Pages` is 60–65 and Title/Author metadata remains populated.

- [ ] **Step 4: Perform visual review**

Render and inspect:

- dedication and acknowledgement;
- table of contents and symbols;
- first and last page of every chapter;
- every figure/table page in Chapters 4 and 5;
- references, appendix, both glossaries, and English abstract.

Confirm readable labels, no clipped text, no isolated heading, and no unintended blank or half-empty page.

- [ ] **Step 5: Perform coherence review**

Read the compact chapters in order and verify:

- every acronym is expanded at first use;
- no section refers to a removed section, table, equation, or example;
- Chapter 3 gap leads to Chapter 4 method;
- Chapter 4 method names match Chapter 5 result labels;
- Chapter 6 answers the Chapter 1 research questions in the same order;
- numerical claims still match stored results.

- [ ] **Step 6: Check the final diff and commit any verification fixes**

```powershell
git diff --check
git status --short
```

If verification required fixes, rerun Steps 1–5 and commit only after fresh evidence. Otherwise report the final page count and commit sequence without creating an empty commit.
