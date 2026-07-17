# Thesis Chapter 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a coherent, implementation-faithful Persian Chapter 4 with one useful architecture diagram and no overlap with the definitions, literature review, or experimental results chapters.

**Architecture:** The chapter is organized as a forward data-flow narrative: data preparation, candidate generation, diagnostic reasoning, intermediate control, final train-calibrated source selection, and reproducible implementation. Formal notation anchors the rules, while one in-document diagram and two compact tables reduce explanatory repetition.

**Tech Stack:** Persian LaTeX, XePersian, existing AUT thesis template, Python implementation files as factual sources

## Global Constraints

- Modify only `chapter4.tex` among thesis source files.
- Do not change `AUTthesis.cls`, `commands.tex`, fonts, or template settings.
- Use Persian digits in prose and conventional digits only in formulas or Latin identifiers.
- Do not report Chapter 5 performance metrics.
- Do not mention unused model families or unvalidated execution paths.

---

### Task 1: Establish the method narrative and formal interfaces

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`

**Interfaces:**
- Consumes: Chapter 3 research gap and the implemented sentence-target classification contract.
- Produces: notation for Direct, THOR, diagnostic, controller, and selected outputs used throughout the chapter.

- [x] Replace the empty skeleton with the introduction, task formulation, design principles, and architecture overview.
- [x] Add one figure using only packages already loaded by the template.
- [x] Verify that the controller is labeled intermediate and the selected prediction is labeled final.

### Task 2: Describe candidate generation and diagnostic control

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`

**Interfaces:**
- Consumes: Direct output and three-run THOR output.
- Produces: structured diagnosis and conservative controller output.

- [x] Explain SCAPT extraction and preservation of the original train/test split.
- [x] Define Direct prompting and output normalization.
- [x] Define THOR's aspect, opinion, polarity reasoning, final label, self-consistency vote, and tie rule.
- [x] Add a homogeneous table of diagnostic error types and define malformed-output repair.
- [x] State the controller's agreement, correction, and fallback rules formally.

### Task 3: Describe final selection and reproducibility

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`

**Interfaces:**
- Consumes: Direct, THOR, diagnostic signals, domain, and train labels.
- Produces: final selected prediction without test-label access.

- [x] Define the four-field profile and train-only source-scoring equation.
- [x] State source tie priority and unseen-profile fallback exactly.
- [x] Add a compact module-responsibility table and explain saved intermediate outputs, resume behavior, and validation.
- [x] Close with a concise summary that leads into Chapter 5.

### Task 4: Review and compile

**Files:**
- Verify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`
- Verify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

- [x] Scan structure, digits, forbidden terms, first-person phrasing, citations, labels, and repeated content.
- [x] Run `scripts/build_thesis.ps1`.
- [x] Inspect `AUTthesis.log` for errors, unresolved citations/references, and overfull horizontal boxes.
- [x] Review the rendered Chapter 4 pages for figure/table layout and Persian text direction.
