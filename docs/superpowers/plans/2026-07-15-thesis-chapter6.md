# Thesis Chapter 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a coherent five-to-seven-page Persian conclusion and future-work chapter that answers all five research questions using only the verified project evidence.

**Architecture:** Replace the existing chapter skeleton with a question-driven narrative. Keep empirical details in Chapter 5, use Chapter 6 to interpret them, separate scientific and engineering contributions, and derive every future-work item from an observed limitation.

**Tech Stack:** XeLaTeX, XePersian, the existing AUT thesis class, PowerShell, MiKTeX XeLaTeX.

## Global Constraints

- Modify only `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex` as thesis source content.
- Do not modify `cls/sty`, fonts, or template settings.
- Add no new figure or table.
- Keep official model and method names in Latin script.
- Use Persian digits in Persian prose except inside formulas and official model names.
- Do not mention `Flan-T5`, `HuggingFace`, or any unevaluated execution capability.
- Do not state that multi-step reasoning alone improved accuracy.
- Treat the Gemini comparison as supplementary evidence from only 90 shared test samples.

---

### Task 1: Write the question-driven conclusion chapter

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex`
- Reference: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter1.tex`
- Reference: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`

**Interfaces:**
- Consumes: the five research questions in Chapter 1 and verified metrics and limitations in Chapter 5.
- Produces: a standalone Chapter 6 with no unresolved citations or references.

- [ ] **Step 1: Replace the placeholder skeleton with the approved structure**

Use these exact LaTeX headings and no additional top-level sections:

```tex
\chapter{نتیجه‌گیری و کارهای آتی}
\section{جمع‌بندی پژوهش}
\section{پاسخ به پرسش‌های پژوهش}
\section{دستاوردهای اصلی}
\subsection{دستاوردهای علمی}
\subsection{دستاوردهای فنی و مهندسی}
\section{محدودیت‌های پژوهش}
\section{کارهای آتی}
\subsection{بهبود انتخاب‌گر منبع}
\subsection{تقویت استدلال و بازبینی هدف‌محور}
\subsection{گسترش منابع و دامنۀ ارزیابی}
\subsection{کاهش هزینۀ اجرا}
\subsection{ارزیابی انسانی و ریزتنظیم}
\section{نتیجه‌گیری نهایی}
```

- [ ] **Step 2: Answer all five research questions explicitly**

Use an `enumerate` environment with five items. Each item must begin with a short bold answer label and contain these evidence anchors:

```tex
\begin{enumerate}
  \item \textbf{تفاوت پیش‌بینی مستقیم و استدلال چندمرحله‌ای.} ...
  \item \textbf{نقش بازبینی و کنترل‌گر.} ...
  \item \textbf{اثر انتخاب منبع تنظیم‌شده.} ...
  \item \textbf{الگوهای اصلی خطا.} ...
  \item \textbf{اثر ظرفیت مدل پایه.} ...
\end{enumerate}
```

The third answer must report accuracy `۰/۶۷۸۷۳۳` to `۰/۷۲۳۹۸۲`, Macro-F1 `۰/۶۷۴۰۷۵` to `۰/۷۱۹۲۰۴`, 26 gains, 6 losses, and 20 net additional correct predictions. The fifth answer must state that Gemini was evaluated on 90 shared test samples and that the observation is not a general model ranking.

- [ ] **Step 3: Write contributions, limitations, and future work without duplicating Chapter 5**

Scientific contributions must explain source complementarity, intermediate-versus-final decisions, and train-only calibration. Engineering contributions must cover traceable intermediate outputs, validation, Qwen/Ollama main execution, and the controlled Gemini comparison. Limitations must include two English domains, Gemini sample size, stochastic reasoning, unmeasured runtime cost, the unused diagnostic label as a final test source, the oracle gap, and the project-adapted rather than exact official THOR reproduction.

Future work must be written as proposals, not completed results. It must cover source-selector validation and sparse profiles, target-aware reasoning, the 116 shared errors, broader languages/domains/models, conditional execution, human reasoning evaluation, and parameter-efficient fine-tuning only if larger trustworthy supervision becomes available.

- [ ] **Step 4: Review the prose for integration and non-repetition**

Confirm that Chapter 6 refers back to established findings rather than reproducing formulas, result tables, confusion matrices, or qualitative examples. Remove repeated procedural explanations already present in Chapter 4 and detailed metric breakdowns already present in Chapter 5.

### Task 2: Run textual and factual checks

**Files:**
- Check: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex`

**Interfaces:**
- Consumes: completed Chapter 6.
- Produces: an audit showing required claims are present and forbidden content is absent.

- [ ] **Step 1: Check required structure and forbidden terms**

Run:

```powershell
Select-String -Path chapter6.tex -Pattern '^\\chapter|^\\section|^\\subsection'
Select-String -Path chapter6.tex -Pattern 'Flan|HuggingFace|ch5_selector_behavior|\\includegraphics|\\begin\{table\}'
```

Expected: all approved headings appear; the second command returns no matches.

- [ ] **Step 2: Check evidence anchors**

Run:

```powershell
Select-String -Path chapter6.tex -Pattern '۰/۶۷۸۷۳۳|۰/۷۲۳۹۸۲|۰/۶۷۴۰۷۵|۰/۷۱۹۲۰۴|۲۶|۶|۲۰|۱۱۶|۹۰'
```

Expected: every required evidence anchor appears in an appropriate answer or limitation.

- [ ] **Step 3: Inspect ASCII digits**

Run:

```powershell
rg -n '[0-9]' chapter6.tex
```

Expected: matches are limited to official Latin names such as `Qwen3 8B`, `Gemini 2.5 Flash`, labels, or LaTeX source syntax; Persian prose contains no English digits.

### Task 3: Compile and inspect the complete thesis

**Files:**
- Build: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.tex`
- Inspect: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.log`
- Output: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: completed Chapter 6 and existing thesis sources.
- Produces: the final compiled PDF and evidence that Chapter 6 introduces no blocking LaTeX issue.

- [ ] **Step 1: Compile twice with XeLaTeX**

Run twice:

```powershell
& 'C:\Users\2021\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe' -interaction=nonstopmode -halt-on-error AUTthesis.tex
```

Expected: both runs exit with code 0 and produce `AUTthesis.pdf`.

- [ ] **Step 2: Inspect the log for blocking issues**

Run:

```powershell
Select-String -Path AUTthesis.log -Pattern 'LaTeX Error|Undefined control sequence|Reference .* undefined|Citation .* undefined|There were undefined references|Overfull \\hbox|Overfull \\vbox'
```

Expected: no matches introduced by Chapter 6. Existing template-level warnings may remain and must not be fixed by changing `cls/sty`.

- [ ] **Step 3: Inspect the Chapter 6 PDF pages visually**

Render or open the pages from the Chapter 6 title through its final paragraph. Confirm correct Persian shaping, decimal order, Latin model-name direction, headings, page breaks, and absence of clipped text.

- [ ] **Step 4: Review the final diff**

Run:

```powershell
git diff --check -- قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex
git status --short
```

Expected: no whitespace errors; Chapter 6 and the generated PDF are modified while unrelated pre-existing user changes remain untouched.
