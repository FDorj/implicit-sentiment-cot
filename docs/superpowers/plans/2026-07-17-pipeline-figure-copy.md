# Pipeline Figure Copy Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** بازنویسی متن کادرهای شکل معماری فصل چهارم به‌صورت کوتاه، منسجم و فنی، بدون تغییر منطق جریان سامانه.

**Architecture:** فایل مستقل TikZ منبع حقیقت شکل باقی می‌ماند. یک آزمون متنی عبارات مصوب و حذف عبارت‌های مبهم را کنترل می‌کند؛ سپس شکل مستقل و PDF کامل پایان‌نامه بازسازی و صفحۀ شکل به‌صورت تصویری بازبینی می‌شود.

**Tech Stack:** XeLaTeX، TikZ، XePersian، Python `unittest`، Poppler (`pdfinfo` و `pdftoppm`)

## Global Constraints

- ساختار گره‌ها، رنگ‌ها، شاخه‌های تصمیم و جهت پیکان‌ها حفظ شود.
- تمایز کنترل‌گر قاعده‌محور با انتخاب‌گر منبع نهایی صریح بماند.
- عنوان هر کادر پررنگ و توضیح آن کوتاه و معمولی باشد.
- شکل در عرض `0.90\textwidth` فصل چهارم بدون هم‌پوشانی یا شکست نامناسب خوانا باشد.
- PDF کامل در بازۀ ۶۰ تا ۶۵ صفحه باقی بماند.

---

### Task 1: تثبیت متن مصوب و بازنویسی شکل TikZ

**Files:**
- Modify: `tests/test_thesis_finalization.py`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter4/ch4_proposed_pipeline.tex`

**Interfaces:**
- Consumes: متن مصوب در `docs/superpowers/specs/2026-07-17-pipeline-figure-copy-design.md`
- Produces: منبع TikZ دارای متن نهایی و آزمون جلوگیری از بازگشت عبارت‌های مبهم

- [ ] **Step 1: نوشتن آزمون شکست‌خورنده برای متن شکل**

به کلاس `ThesisFinalizationTests` در `tests/test_thesis_finalization.py` متد زیر افزوده شود:

```python
def test_pipeline_figure_uses_clear_approved_copy(self):
    figure = read_thesis_file("Images/Chapter4/ch4_proposed_pipeline.tex")

    for expected in [
        "جمله و هدفی که باید احساس نسبت به آن تعیین شود",
        "پیش‌بینی مستقیم",
        "مسیر استدلالی \\lr{THOR}",
        "تعیین برچسب با رأی اکثریت",
        "تحلیل اختلاف",
        "نوع خطا، برچسب پیشنهادی و سطح اطمینان تعیین می‌شود",
        "ساخت پروفایل انتخاب",
        "دو برچسب، نوع خطا، اطمینان و دامنه",
        "تنظیم با دادۀ آموزش",
        "انتخاب منبع نهایی",
        "در غیر این صورت، پاسخ مستقیم حفظ می‌شود",
        "فقط یک تصمیم میانی برای مقایسه می‌سازد",
        "پیش‌بینی نهایی",
    ]:
        self.assertIn(expected, figure)

    for unclear in [
        "عبارتی که احساس نسبت به آن سنجیده می‌شود",
        "برچسب‌های تولیدشده و نتیجۀ بررسی اختلاف",
        "پروفایل عملکردی پنج‌جزئی و شروط محافظ",
        "تولید تصمیم کمکی برای تحلیل",
    ]:
        self.assertNotIn(unclear, figure)
```

- [ ] **Step 2: اجرای آزمون و تأیید وضعیت RED**

Run:

```powershell
& 'D:\implicit-sentiment-cot\.venv\Scripts\python.exe' -B -m unittest tests.test_thesis_finalization.ThesisFinalizationTests.test_pipeline_figure_uses_clear_approved_copy -v
```

Expected: `FAIL` به‌دلیل نبود عبارت `جمله و هدفی که باید احساس نسبت به آن تعیین شود`.

- [ ] **Step 3: جایگزینی متن گره‌های TikZ**

متن گره‌ها در `ch4_proposed_pipeline.tex` با محتوای زیر جایگزین شود؛ مختصات و پیکان‌ها حفظ شوند:

```latex
\node[input] (input) at (0,0)
  {\textbf{ورودی سامانه}\\[-1pt]
   جمله و هدفی که باید احساس نسبت به آن تعیین شود};

\node[source] (thor) at (-3.35,-2.05)
  {\textbf{مسیر استدلالی \lr{THOR}}\\[-1pt]
   سه اجرای مستقل؛ تعیین برچسب با رأی اکثریت};
\node[source] (direct) at (3.35,-2.05)
  {\textbf{پیش‌بینی مستقیم}\\[-1pt]
   مدل بدون تولید استدلال، یک برچسب احساس می‌دهد};

\node[compare] (compare) at (0,-4.05)
  {مقایسۀ دو مسیر\\آیا برچسب‌ها یکسان‌اند؟};

\node[diagnostic] (diagnostic) at (-3.35,-6.05)
  {\textbf{تحلیل اختلاف}\\[-1pt]
   نوع خطا، برچسب پیشنهادی و سطح اطمینان تعیین می‌شود};
\node[agreement] (agreement) at (3.35,-6.05)
  {\textbf{توافق دو مسیر}\\[-1pt]
   برچسب مشترک بدون تغییر حفظ می‌شود};

\node[signals] (signals) at (0,-8.15)
  {\textbf{ساخت پروفایل انتخاب}\\[-1pt]
   دو برچسب، نوع خطا، اطمینان و دامنه کنار هم قرار می‌گیرند};

\node[training] (training) at (-6.2,-10.35)
  {\textbf{تنظیم با دادۀ آموزش}\\[-1pt]
   تعیین آستانه‌های محافظ با اعتبارسنجی داخلی};
\node[selector] (selector) at (-1.45,-10.35)
  {\textbf{انتخاب منبع نهایی}\\[-1pt]
   اگر شواهد کافی باشد، بهترین منبع انتخاب می‌شود؛\\[-1pt]
   در غیر این صورت، پاسخ مستقیم حفظ می‌شود};
\node[controller] (controller) at (4.25,-10.35)
  {\textbf{کنترل‌گر قاعده‌محور}\\[-1pt]
   فقط یک تصمیم میانی برای مقایسه می‌سازد\\[-1pt]
   و در پاسخ نهایی دخالت ندارد};

\node[final] (final) at (-1.45,-12.4)
  {\textbf{پیش‌بینی نهایی}\\[-1pt]
   برچسب منبع منتخب، خروجی سامانه است};
```

سبک‌های زیر نیز برای جاگیری متن تنظیم شوند:

```latex
signals: text width=8.4cm, minimum height=1.25cm
training: text width=3.0cm, minimum height=1.5cm
selector: text width=5.5cm, minimum height=1.9cm, font=\footnotesize
controller: text width=4.4cm, minimum height=1.8cm, font=\footnotesize
final: text width=5.0cm, minimum height=1.05cm, font=\small
```

- [ ] **Step 4: اجرای آزمون و تأیید وضعیت GREEN**

Run:

```powershell
& 'D:\implicit-sentiment-cot\.venv\Scripts\python.exe' -B -m unittest tests.test_thesis_finalization.ThesisFinalizationTests.test_pipeline_figure_uses_clear_approved_copy -v
```

Expected: `Ran 1 test ... OK`.

- [ ] **Step 5: ساخت مستقل شکل**

Run از پوشۀ `Images/Chapter4`:

```powershell
& xelatex -interaction=nonstopmode -halt-on-error ch4_proposed_pipeline.tex
```

Expected: خروجی `ch4_proposed_pipeline.pdf` با exit code صفر و بدون `Overfull \hbox`.

- [ ] **Step 6: کامیت منبع شکل و آزمون**

```powershell
git add tests/test_thesis_finalization.py قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter4/ch4_proposed_pipeline.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter4/ch4_proposed_pipeline.pdf
git commit -m "docs: clarify thesis pipeline figure"
```

### Task 2: بازسازی پایان‌نامه و کنترل بصری

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`
- Verify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`

**Interfaces:**
- Consumes: `Images/Chapter4/ch4_proposed_pipeline.pdf` از Task 1
- Produces: PDF کامل پایان‌نامه با شکل خوانا و آزمون‌های سبز

- [ ] **Step 1: ساخت کامل پایان‌نامه**

Run از ریشۀ مخزن:

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\build_thesis.ps1'
```

Expected: `Thesis PDF created successfully` و PDF برابر ۶۰ تا ۶۵ صفحه.

- [ ] **Step 2: رندر صفحۀ شکل برای بازبینی**

```powershell
$pdf='قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir\AUTthesis.pdf'
pdftoppm -f 29 -l 29 -png -r 180 -singlefile $pdf pipeline-page
```

Expected: تصویر دارای متن کامل در تمام کادرها، بدون هم‌پوشانی، قطع‌شدن کلمه یا برخورد پیکان با متن.

- [ ] **Step 3: اجرای آزمون کامل**

```powershell
& 'D:\implicit-sentiment-cot\.venv\Scripts\python.exe' -B -m unittest discover -s tests -q
```

Expected: تمام آزمون‌ها با `OK` خاتمه یابند.

- [ ] **Step 4: کنترل نهایی PDF و diff**

```powershell
pdfinfo قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir\AUTthesis.pdf
git diff --check
git status --short
```

Expected: `Pages: 60..65`، نبود خطای whitespace و فقط PDF اصلی به‌عنوان تغییر باقی‌مانده.

- [ ] **Step 5: کامیت PDF کامل**

```powershell
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf
git commit -m "build: refresh thesis with clarified pipeline figure"
```
