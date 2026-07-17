# Chapter 4 Pipeline Figure Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the plain Chapter 4 architecture diagram with a formal vector figure that accurately separates the intermediate controller from the final selector.

**Architecture:** A standalone TikZ source produces a PDF asset under `Images/Chapter4`. Chapter 4 includes that PDF through the template's existing `graphicx` support, so the thesis class, styles, and font configuration remain unchanged.

**Tech Stack:** XeLaTeX, standalone, TikZ, XePersian, existing AUT thesis template

## Global Constraints

- Do not modify `AUTthesis.cls`, `commands.tex`, style files, or thesis fonts.
- Use a formal navy, pale-blue, and gray palette that remains readable in grayscale.
- Show Direct and THOR as parallel sources, diagnostic review only on disagreement, and training data as a separate selector input.
- Keep the controller visibly intermediate and do not connect it as an input to the selector.
- Produce a vector PDF and preserve the existing figure label `fig:proposed-pipeline`.

---

### Task 1: Build the standalone vector figure

**Files:**
- Create: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter4/ch4_proposed_pipeline.tex`
- Generate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter4/ch4_proposed_pipeline.pdf`

**Interfaces:**
- Consumes: The validated pipeline order described in Chapter 4.
- Produces: A cropped vector PDF suitable for `\\includegraphics`.

- [x] Create a standalone RTL TikZ document with local font paths and formal color definitions.
- [x] Define consistent node styles for input, model sources, comparison, diagnostic, intermediate, selector, training signal, and final output.
- [x] Draw the primary top-to-bottom flow and separate side branches with labeled agreement and disagreement arrows.
- [x] Compile with `xelatex -interaction=nonstopmode -halt-on-error ch4_proposed_pipeline.tex`.
- [x] Confirm that `ch4_proposed_pipeline.pdf` exists and the standalone log has no error or overfull box.

### Task 2: Integrate the figure into Chapter 4

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`

**Interfaces:**
- Consumes: `Images/Chapter4/ch4_proposed_pipeline.pdf`.
- Produces: The existing `fig:proposed-pipeline` reference in a cleaner rendered form.

- [x] Replace the in-document `fbox` and `tabular` diagram with `\\includegraphics[width=0.92\\textwidth]{Images/Chapter4/ch4_proposed_pipeline.pdf}`.
- [x] Preserve the caption and label, and adjust the preceding paragraph to mention the explicit agreement branch and training-data input.
- [x] Keep the explanatory paragraph after the figure aligned with the new visual hierarchy.

### Task 3: Verify the thesis output

**Files:**
- Verify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`
- Verify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.log`

**Interfaces:**
- Consumes: The integrated Chapter 4 source and generated vector asset.
- Produces: A verified thesis PDF with a readable formal diagram.

- [x] Run `scripts/build_thesis.ps1` and expect exit code zero.
- [x] Search the final log for LaTeX errors, unresolved references or citations, and overfull horizontal boxes; expect none.
- [x] Render the figure page to a raster preview and inspect Persian direction, arrow routing, text size, color contrast, and page fit.
- [x] Recheck that no template, class, style, or font file changed.

### Task 4: Correct bidirectional text and arrow visibility

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter4/ch4_proposed_pipeline.tex`
- Generate: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter4/ch4_proposed_pipeline.pdf`
- Verify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: User feedback about mixed RTL/LTR text and indistinct arrowheads in the integrated figure.
- Produces: A Persian-first figure with isolated `THOR` text and clearly visible branch direction.

- [x] Replace inline `THOR` in Persian sentences with Persian role names; retain `THOR` only as a standalone source heading.
- [x] Replace `no_error` with «وضعیت بدون خطا» and use «استدلالی» instead of inline Latin source names in signal and selector descriptions.
- [x] Compile the standalone figure and confirm that only one `\\lr{...}` occurrence remains in the source.
- [x] Increase the primary flow line to at least `1.05pt` and the arrowhead to at least `3.2mm` by `2.3mm`.
- [x] Move «توافق» and «اختلاف» into separately positioned badges that do not intersect any path.
- [x] Compile the standalone figure again and inspect its raster preview at thesis-scale dimensions.
- [x] Run `scripts/build_thesis.ps1`, inspect the final figure page, and confirm no LaTeX error, unresolved reference, or overfull horizontal box.
