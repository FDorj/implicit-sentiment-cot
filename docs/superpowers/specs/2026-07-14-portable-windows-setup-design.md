# Portable Windows Setup Design

## Goal

Make a fresh clone usable on another Windows laptop for three independent tasks: running the Python project, validating the saved experimental pipeline, and compiling the committed LaTeX thesis into `AUTthesis.pdf`.

## Design

- Keep system-level installations explicit: Git, Python 3.10 or newer, Ollama, and MiKTeX with XeLaTeX.
- Add a PowerShell setup script that creates `.venv`, installs only the core Python requirements, and optionally pulls `qwen3:8b`.
- Add a PowerShell thesis build script that runs XeLaTeX, BibTeX, and the two final XeLaTeX passes from the thesis directory, then checks that the PDF exists and that the log has no fatal LaTeX error or unresolved citation/reference warning.
- Add a verification script that runs the unit tests, validates/regenerates the saved final result summaries without calling a language model, and builds the thesis.
- Document a short clone-to-run path in `README.md`, including the MiKTeX missing-package setting and the optional Poppler dependency used only when regenerating figure previews.

## Constraints

- Do not store credentials or machine-specific absolute paths.
- Do not require rerunning the expensive language-model experiments to build the thesis.
- Keep the existing LaTeX class, style, fonts, and thesis structure unchanged.
- Keep generated experiment outputs and the final thesis PDF versioned; keep temporary build files ignored.

## Verification

- Run the portability asset tests.
- Run the complete Python unit-test suite.
- Regenerate and validate final result summaries.
- Build the full thesis and inspect the final LaTeX log for errors, unresolved citations/references, and overfull boxes.
