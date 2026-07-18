# Thesis Acronym Footnotes Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** حذف عنوان کامل مقالۀ THOR از نثر فصل اول و حذف اختصارهای تکراری یا استفاده‌نشده از پانویس‌های فارسی، بدون تغییر محتوای علمی.

**Architecture:** تغییر فقط در متن LaTeX فصل‌های ۱، ۲، ۳ و ۵ انجام می‌شود. یک آزمون منبع‌محور از بازگشت عنوان بلند و الگوهای پانویس تکراری جلوگیری می‌کند و سپس PDF دوباره ساخته و کنترل می‌شود.

**Tech Stack:** XeLaTeX، Python `unittest`، LaTeX/xepersian

## Global Constraints

- عنوان کامل مقالۀ THOR از متن حذف شود و `THOR` همراه `\cite{fei2023thor}` باقی بماند.
- معادل انگلیسی مفید پانویس‌ها حفظ شود، اما اختصار داخل پرانتز در پانویس تکرار نشود.
- پانویس‌های ترجمه‌ای مانند `Target`، `Aspect`، `Accuracy`، `Precision` و `Recall` حفظ شوند.
- `ETC` در متن اصلی معرفی شود و پانویس آن فقط نام کامل انگلیسی را نگه دارد.
- هیچ عدد، ارجاع، نتیجه یا ادعای علمی تغییر نکند.

---

### Task 1: افزودن کنترل رگرسیون نگارشی

**Files:**
- Modify: `tests/test_thesis_finalization.py`
- Test: `tests/test_thesis_finalization.py`

**Interfaces:**
- Consumes: متن UTF-8 فایل‌های فصل با تابع موجود `read_thesis_file(name)`
- Produces: آزمونی که عنوان بلند و اختصارهای تکراری پانویس را ممنوع می‌کند

- [ ] **Step 1: آزمون شکست‌خورنده را اضافه کنید**

```python
def test_acronym_footnotes_do_not_repeat_abbreviations(self):
    chapters = "\n".join(
        read_thesis_file(name)
        for name in ["chapter1.tex", "chapter2.tex", "chapter3.tex", "chapter5.tex"]
    )
    self.assertNotIn(
        "Reasoning Implicit Sentiment with Chain-of-Thought Prompting",
        chapters,
    )
    for redundant in [
        "Implicit Sentiment Analysis (ISA)",
        "Supervised Contrastive Pre-Training (SCAPT)",
        "Aspect-Based Sentiment Analysis (ABSA)",
        "Large Language Model (LLM)",
        "Chain-of-Thought (CoT)",
        "Sentiment Analysis of Thinking (SAoT)",
        "Error-Type-Aware Reflection; ETC",
    ]:
        self.assertNotIn(redundant, chapters)
```

- [ ] **Step 2: شکست آزمون را تأیید کنید**

Run: `python -m unittest tests.test_thesis_finalization.ThesisFinalizationTests.test_acronym_footnotes_do_not_repeat_abbreviations`

Expected: `FAIL` چون عنوان مقاله و الگوهای تکراری هنوز در فصل‌ها وجود دارند.

---

### Task 2: پاک‌سازی متن و پانویس‌ها

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter1.tex:13`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter2.tex:14`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter3.tex:25`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex:57`
- Test: `tests/test_thesis_finalization.py`

**Interfaces:**
- Consumes: قواعد آزمون Task 1
- Produces: نثر کوتاه‌تر و پانویس‌های بدون اختصار تکراری

- [ ] **Step 1: عنوان مقالۀ THOR را از فصل اول حذف کنید**

عبارت آغازین باید به این صورت بماند:

```tex
روش استدلال سه‌مرحله‌ای \lr{THOR}\LTRfootnote{Three-hop Reasoning} مسئله را به استخراج جنبه، استنتاج نظر ضمنی و تعیین قطبیت تقسیم می‌کند \cite{fei2023thor}.
```

- [ ] **Step 2: اختصارهای تکراری یا استفاده‌نشده را از پانویس‌ها حذف کنید**

جایگزینی‌های دقیق:

```text
Implicit Sentiment Analysis (ISA) -> Implicit Sentiment Analysis
Supervised Contrastive Pre-Training (SCAPT) -> Supervised Contrastive Pre-Training
Aspect-Based Sentiment Analysis (ABSA) -> Aspect-Based Sentiment Analysis
Large Language Model (LLM) -> Large Language Model
Chain-of-Thought (CoT) -> Chain-of-Thought
Sentiment Analysis of Thinking (SAoT) -> Sentiment Analysis of Thinking
Error-Type-Aware Reflection; ETC -> Error-Type-Aware Reflection
```

در فصل پنجم `\lr{ETC}` باید کنار اصطلاح فارسی در متن اصلی قرار گیرد تا اختصار واقعاً معرفی شود.

- [ ] **Step 3: آزمون متمرکز را اجرا کنید**

Run: `python -m unittest tests.test_thesis_finalization.ThesisFinalizationTests.test_acronym_footnotes_do_not_repeat_abbreviations`

Expected: `OK`.

---

### Task 3: بازسازی و کنترل پایان‌نامه

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`
- Test: `tests/`

**Interfaces:**
- Consumes: فایل‌های LaTeX پاک‌سازی‌شده
- Produces: PDF نهایی و شواهد آزمون

- [ ] **Step 1: مجموعۀ آزمون کامل را اجرا کنید**

Run: `python -m unittest discover -s tests -p "test_*.py"`

Expected: همۀ آزمون‌ها `OK`.

- [ ] **Step 2: PDF را دو بار بسازید**

Working directory: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir`

Run twice: `xelatex --enable-installer -interaction=nonstopmode AUTthesis.tex`

Expected: exit code `0` و تولید `AUTthesis.pdf`.

- [ ] **Step 3: مشخصات PDF و پاکیزگی diff را کنترل کنید**

Run: `pdfinfo AUTthesis.pdf`

Expected: اندازۀ A4 و تعداد صفحات بین ۶۰ تا ۶۵.

Run: `git diff --check`

Expected: بدون خطا.

- [ ] **Step 4: تغییرات اجرایی را کامیت کنید**

```powershell
git add tests/test_thesis_finalization.py قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter1.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter2.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter3.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf
git commit -m "docs: streamline thesis acronym footnotes"
```
