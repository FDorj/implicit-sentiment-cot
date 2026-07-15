# Chapter 4 Pipeline Figure Redesign

## Goal

Replace the plain box diagram in Chapter 4 with a formal, publication-quality vector figure that communicates the implemented decision flow without changing the university template, fonts, class, or style files.

## Visual Direction

- Use a restrained academic palette: dark navy for primary outlines and headings, pale blue for model stages, and light gray for auxiliary or intermediate stages.
- Preserve legibility when printed in grayscale through contrast, labels, border weight, and line style rather than color alone.
- Use consistent rounded rectangles, compact Persian labels, aligned spacing, and directional arrows.
- Keep the figure within a portrait thesis page at normal reading size.
- Keep Persian sentences free of inline Latin identifiers. The literal name `THOR` may appear only as a standalone heading on its own line.
- Replace implementation identifiers such as `no_error` with clear Persian conceptual labels inside the figure.
- Use heavier flow lines and substantially larger arrowheads so direction remains visible after the figure is scaled in the thesis.
- Place the agreement and disagreement labels beside their branches in separate badges rather than over the arrow paths.

## Information Architecture

The figure follows a top-to-bottom primary path:

1. Sentence and specified target.
2. Parallel Direct and THOR SC3 branches.
3. Explicit comparison node.
4. Agreement and disagreement outcomes.
5. Diagnostic review only on disagreement.
6. Unified decision signals.
7. Two visibly separate consumers:
   - rule-based controller as an intermediate analysis output;
   - train-calibrated selector as the only path to the final label.
8. A separate training-data input enters the selector to show that source profiles are learned from the training split.
9. The selected final label is emphasized as the terminal output.

The controller is not connected as an input to the selector. Direct, THOR, and diagnostic labels are named as the selector candidates.

## Implementation

- Add a standalone TikZ source under `Images/Chapter4` and compile it to PDF.
- Include the generated PDF with the template's existing `graphicx` support.
- Do not load TikZ in the thesis document and do not modify `AUTthesis.cls`, `commands.tex`, style files, or fonts.
- Replace only the old in-document figure block and adjust its surrounding explanation where needed.

## Verification

- Compile the standalone figure successfully.
- Compile the full thesis with the existing build script.
- Confirm no unresolved references, LaTeX errors, or overfull horizontal boxes.
- Inspect the rendered page at normal size for Persian direction, arrow routing, text size, and grayscale readability.
