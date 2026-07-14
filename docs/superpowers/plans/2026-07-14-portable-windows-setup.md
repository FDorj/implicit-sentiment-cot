# Portable Windows Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh Windows clone directly set up, testable, runnable with Ollama, and able to compile the thesis PDF.

**Architecture:** Small PowerShell entry points wrap the existing Python and LaTeX commands. The README remains the source of truth for system prerequisites and separates core execution from optional figure regeneration.

**Tech Stack:** PowerShell 5.1+, Python 3.10+, Ollama, MiKTeX/XeLaTeX/BibTeX

## Global Constraints

- Do not change the LaTeX class, style, fonts, or thesis structure.
- Do not commit credentials, virtual environments, or LaTeX intermediate files.
- Do not rerun language-model inference during portability verification.

---

### Task 1: Define the portable repository contract

**Files:**
- Create: `tests/test_portability_assets.py`
- Modify: `.gitignore`
- Modify: `requirements.txt`

- [x] Add failing tests for the expected setup, build, verification, and README entry points.
- [x] Run the focused test and confirm it fails because the scripts do not exist.
- [x] Add bounded core dependency ranges; the existing ignore rules already cover local setup artifacts.

### Task 2: Add Windows entry points

**Files:**
- Create: `scripts/setup_windows.ps1`
- Create: `scripts/build_thesis.ps1`
- Create: `scripts/verify_project.ps1`

- [x] Implement a virtual-environment setup using `python -m pip`.
- [x] Implement the four-pass XeLaTeX/BibTeX build with post-build log checks.
- [x] Implement the complete local verification sequence.
- [x] Run the focused portability test and confirm it passes.

### Task 3: Document and verify a clean handoff

**Files:**
- Modify: `README.md`

- [x] Add exact clone, setup, Ollama smoke-test, saved-result validation, and thesis-build commands.
- [x] Explain MiKTeX on-the-fly package installation and optional figure-preview requirements.
- [x] Run all unit tests and the final-pipeline validator.
- [x] Build the thesis and inspect the log.
- [x] Prepare the verified change set for commit and push to `main`.
