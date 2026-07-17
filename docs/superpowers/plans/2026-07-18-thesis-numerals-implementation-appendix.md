# Thesis Numerals, Implementation Documentation, and Appendix Revision Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** نمایش تمام اعداد علمی پایان‌نامه با رقم و ممیز فارسی، توضیح ساختار واقعی پیاده‌سازی و خط پایۀ `TF-IDF` در فصل چهارم، تعریف روشن انتخاب‌گرهای مقایسه‌ای و بازآرایی پیوست بازتولید.

**Architecture:** قالب‌بندی عدد در دو مرز حل می‌شود: متن LaTeX از دستور واحد `\fanum` استفاده می‌کند و تولیدکنندۀ شکل‌ها برچسب نمایشی فارسی را جدا از مختصات عددی لاتین `pgfplots` می‌سازد. مستندسازی کد بر اساس لایه‌های داده، خط پایۀ کلاسیک، اجرای مدل، تولید منابع، انتخاب منبع و ارزیابی نوشته می‌شود؛ پیوست فقط اطلاعات عملی بازتولید را نگه می‌دارد.

**Tech Stack:** Python 3، unittest، pandas، scikit-learn، XeLaTeX/xepersian، TikZ/pgfplots، PowerShell، Poppler.

## Global Constraints

- اعداد آماری، تعداد نمونه‌ها، درصدها، دماها، محور نمودارها و مقدارهای روی شکل‌ها باید فارسی باشند.
- ممیز نمایش فارسی `٫` است؛ `/` نباید به‌عنوان ممیز باقی بماند.
- رقم فارسی نباید داخل `\lr` قرار گیرد.
- نام‌های رسمی مانند `Qwen3 8B` و `Gemini 2.5 Flash` و نام فایل‌ها/فرمان‌ها لاتین باقی می‌مانند.
- مقایسۀ تکمیلی واقعی `Qwen3 8B` و `Gemini 2.5 Flash` روی ۹۰ نمونۀ مشترک است، نه ۲۴۰ نمونه.
- فصل چهارم باید `TF-IDF + Logistic Regression` و تمام لایه‌های اصلی کد را توضیح دهد، بدون درج فهرست خط‌به‌خط توابع یا قطعه‌کد خام.
- پیوست در فهرست مطالب حفظ و با عنوان «پیوست: راهنمای بازتولید آزمایش‌ها» تمیز و فشرده می‌شود.
- هیچ اجرای جدید مدل زبانی و هیچ تغییر در نتایج آزمایش‌ها انجام نمی‌شود.
- PDF نهایی باید ۶۵ تا ۷۵ صفحه بماند.

---

### Task 1: قالب‌بندی فارسی اعداد در شکل‌های فصل پنجم

**Files:**
- Modify: `experiments/generate_thesis_result_figures.py`
- Modify: `tests/test_generate_thesis_result_figures.py`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_*.pdf`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/ch5_*.png`

**Interfaces:**
- Produces: `_format_persian_number(value: int | float, precision: int | None = None) -> str`
- Produces: مختصات `pgfplots` با مقدار محاسباتی لاتین و `point meta=explicit symbolic` فارسی.
- Consumes: خروجی‌های ذخیره‌شدۀ موجود در `results/`؛ مدل زبانی اجرا نمی‌شود.

- [ ] **Step 1: افزودن آزمون شکست‌خور برای تبدیل اعداد**

در `tests/test_generate_thesis_result_figures.py` تابع قالب‌بندی را وارد و آزمون زیر را اضافه کن:

```python
from experiments.generate_thesis_result_figures import _format_persian_number

def test_persian_number_formatter_uses_persian_digits_and_decimal_separator(self):
    self.assertEqual(_format_persian_number(0.7, 1), "۰٫۷")
    self.assertEqual(_format_persian_number(90), "۹۰")
    self.assertEqual(_format_persian_number(80.5, 1), "۸۰٫۵")
    self.assertNotIn("/", _format_persian_number(0.7, 1))
```

- [ ] **Step 2: اجرای آزمون و تأیید RED**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest tests.test_generate_thesis_result_figures.ThesisResultFigureRenderingTests.test_persian_number_formatter_uses_persian_digits_and_decimal_separator -v
```

Expected: `ImportError` یا شکست به‌علت نبود `_format_persian_number`.

- [ ] **Step 3: پیاده‌سازی قالب‌بند مشترک**

در ابتدای `generate_thesis_result_figures.py` اضافه کن:

```python
PERSIAN_DIGIT_TRANSLATION = str.maketrans("0123456789.-", "۰۱۲۳۴۵۶۷۸۹٫−")


def _format_persian_number(value: int | float, precision: int | None = None) -> str:
    if precision is None:
        rendered = str(int(value)) if isinstance(value, int) or float(value).is_integer() else str(value)
    else:
        rendered = f"{float(value):.{precision}f}"
    return rendered.translate(PERSIAN_DIGIT_TRANSLATION)
```

در پیش‌درآمد شکل‌ها، قلم رقم پایان‌نامه را نیز تنظیم کن:

```latex
\setdigitfont[Path={...},BoldFont={Yas Bd.ttf}]{Yas.ttf}
```

- [ ] **Step 4: افزودن آزمون شکست‌خور برای metadata و برچسب‌های فارسی**

در آزمون رندر شکل‌ها این کنترل‌ها را اضافه کن:

```python
self.assertIn("point meta=explicit symbolic", comparison)
self.assertIn("[۰٫۷۲۲]", comparison)
self.assertIn("yticklabels={۰,۰٫۱,۰٫۲,۰٫۳,۰٫۴,۰٫۵,۰٫۶,۰٫۷,۰٫۸,۰٫۹}", comparison)
self.assertNotIn(r"\pgfmathprintnumber", comparison)
self.assertIn(r"۶۶\\۸۰٫۵\%", confusion)
self.assertIn("تعداد نمونه‌ها: ۹۰", gemini_confusion)
```

Expected before implementation: FAIL because current render uses Latin `pgfmathprintnumber`, Latin ticks, and Latin cell values.

- [ ] **Step 5: جداسازی مختصات محاسباتی از برچسب نمایشی**

برای نمودارهای میله‌ای مختصات را به‌شکل زیر تولید کن:

```python
label = _format_persian_number(row[field], 3)
coordinate = f"({row[field]:.6f},{{{method_label}}}) [{label}]"
```

و در `axis` از تنظیم زیر استفاده کن:

```latex
point meta=explicit symbolic,
nodes near coords*={\pgfplotspointmeta},
```

برای محورهای عددی `xticklabels` یا `yticklabels` فارسی صریح بنویس؛ مختصات و `xtick`/`ytick` لاتین باقی بمانند تا محاسبۀ `pgfplots` آسیب نبیند. در ماتریس اغتشاش، شمار و درصد را پیش از ورود به node با `_format_persian_number` بساز. عبارت‌های `\lr{$n=90$}` و `\lr{$n=442$}` را به «تعداد نمونه‌ها: ۹۰» و «تعداد نمونه‌ها: ۴۴۲» تبدیل کن.

- [ ] **Step 6: اجرای آزمون‌های تولید شکل**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest tests.test_generate_thesis_result_figures -v
```

Expected: تمام آزمون‌های این ماژول `OK`.

- [ ] **Step 7: بازسازی و بازبینی پنج شکل**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B experiments\generate_thesis_result_figures.py
```

Expected: `Generated 10 assets` و پنج PDF/PNG غیرخالی. هر پنج PNG را بررسی کن: رقم فارسی، ممیز `٫`، ترتیب درست اعداد، نبود هم‌پوشانی و حفظ نام لاتین مدل‌ها.

- [ ] **Step 8: commit**

```powershell
git add experiments/generate_thesis_result_figures.py tests/test_generate_thesis_result_figures.py قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5
git commit -m "fix: render thesis result figures with Persian numerals"
```

---

### Task 2: اصلاح اعداد متن، جدول‌ها و تعداد نمونۀ Gemini

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/commands.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/fa_title.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/en-abstract.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter1.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex`
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/appendix1.tex`
- Modify: `tests/test_thesis_finalization.py`

**Interfaces:**
- Produces: `\fanum{<Persian digits with ٫>}` برای عدد فارسی نشکن و هم‌جهت با متن.
- Preserves: نام‌های رسمی مدل و متن انگلیسی چکیده با رقم لاتین.

- [ ] **Step 1: نوشتن آزمون‌های رگرسیون متن پایان‌نامه**

در `tests/test_thesis_finalization.py` آزمونی اضافه کن که فایل‌های فارسی را بخواند و این شرایط را اعمال کند:

```python
def test_persian_thesis_numbers_are_direction_safe(self):
    thesis_files = [
        "fa_title.tex", "chapter1.tex", "chapter4.tex",
        "chapter5.tex", "chapter6.tex", "appendix1.tex",
    ]
    combined = "\n".join((THESIS_DIR / name).read_text(encoding="utf-8") for name in thesis_files)
    self.assertNotIn(r"\lr{\setpersianfont", combined)
    self.assertNotRegex(combined, r"[۰-۹]+/[۰-۹]+")
    self.assertIn(r"\fanum{۰٫۷}", combined)
    self.assertIn("۹۰ نمون", combined)
```

آزمون دیگری مقدار منسوخ را رد کند:

```python
def test_shared_gemini_subset_is_consistently_ninety(self):
    for name in ["fa_title.tex", "en-abstract.tex", "appendix1.tex"]:
        text = (THESIS_DIR / name).read_text(encoding="utf-8")
        self.assertNotIn("۲۴۰", text)
        self.assertNotIn("240 examples", text)
```

- [ ] **Step 2: اجرای آزمون و تأیید RED**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest tests.test_thesis_finalization -v
```

Expected: شکست روی الگوهای `\lr{\setpersianfont ...}`، ممیز `/` و مقدار ۲۴۰.

- [ ] **Step 3: تعریف دستور عدد فارسی**

پس از تنظیم قلم رقم در `commands.tex` اضافه کن:

```latex
% عدد فارسیِ نشکن؛ ورودی باید رقم فارسی و ممیز فارسی «٫» داشته باشد.
\newcommand{\fanum}[1]{\mbox{#1}}
```

- [ ] **Step 4: مهاجرت تمام اعداد فارسی اعشاری**

تمام نمونه‌های زیر را در فایل‌های فارسی جایگزین کن:

```latex
\lr{\setpersianfont ۰/۶۷۸۷۳۳}  ->  \fanum{۰٫۶۷۸۷۳۳}
\lr{\setpersianfont ۱۳/۱۲}      ->  \fanum{۱۳٫۱۲}
\lr{0.7}                         ->  \fanum{۰٫۷}
\lr{0.05}                        ->  \fanum{۰٫۰۵}
```

همۀ اعداد جدول‌ها و نثر را پوشش بده و سپس با `rg` ثابت کن هیچ رقم فارسی داخل `\lr` و هیچ ممیز `/` باقی نمانده است.

- [ ] **Step 5: اصلاح تعداد نمونۀ مقایسۀ تکمیلی**

در چکیدۀ فارسی و پیوست `۲۴۰` را به `۹۰` و در چکیدۀ انگلیسی `240 examples` را به `90 examples` تغییر بده. اعداد چکیدۀ انگلیسی لاتین باقی بمانند.

- [ ] **Step 6: اجرای آزمون‌های فصل و diff hygiene**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest tests.test_thesis_finalization -v
git diff --check
```

Expected: آزمون‌ها `OK` و `git diff --check` بدون خروجی.

- [ ] **Step 7: commit**

```powershell
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/commands.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/fa_title.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/en-abstract.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter1.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter6.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/appendix1.tex tests/test_thesis_finalization.py
git commit -m "fix: normalize Persian numerals across thesis text"
```

---

### Task 3: افزودن ساختار واقعی پیاده‌سازی به فصل چهارم

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex`
- Modify: `tests/test_thesis_finalization.py`

**Interfaces:**
- Consumes: نام‌های واقعی کلاس‌ها و اسکریپت‌های موجود در `src/` و `experiments/`.
- Produces: زیربخش «ساختار پیاده‌سازی سامانه» با توضیح لایه‌ای و جدول نقش/ورودی/خروجی.

- [ ] **Step 1: نوشتن آزمون پوشش مؤلفه‌های واقعی**

```python
def test_chapter4_documents_real_implementation_components(self):
    text = (THESIS_DIR / "chapter4.tex").read_text(encoding="utf-8")
    required = [
        "ساختار پیاده‌سازی سامانه",
        "run_tfidf_logreg_baseline.py",
        "PromptRunner", "OllamaPromptRunner", "OpenAICompatiblePromptRunner",
        "THORPipeline", "SimpleReflectionPipeline", "ErrorTypeReflectionPipeline",
        "data_loader.py", "experiment_config.py", "controller.py",
        "apply_etc_policy.py", "run_meta_selector.py",
        "run_logistic_source_ranker.py", "final_results.py",
        "run_final_pipeline.py", "generate_thesis_result_figures.py",
        "اعتبارسنجی پنج‌لایه", "بازبرازش",
    ]
    for token in required:
        self.assertIn(token, text)
```

- [ ] **Step 2: اجرای آزمون و تأیید RED**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest tests.test_thesis_finalization.ThesisFinalizationTests.test_chapter4_documents_real_implementation_components -v
```

Expected: FAIL روی عنوان زیربخش و نام مؤلفه‌ها.

- [ ] **Step 3: افزودن مقدمه و جدول لایه‌های نرم‌افزار**

پیش از «پیاده‌سازی و جمع‌بندی» زیربخش جدید را اضافه کن. متن باید روشن کند که کلاس‌های هسته‌ای در `src` هستند و انتخاب‌گرها/اجراهای دسته‌ای عمدتاً به‌صورت توابع و اسکریپت‌های `experiments` پیاده شده‌اند. جدول چهار ستون داشته باشد: «لایه»، «کلاس یا فایل»، «ورودی/خروجی» و «مسئولیت».

ردیف خط پایۀ کلاسیک باید صریحاً بگوید:

```latex
اسکریپت \lr{run\_tfidf\_logreg\_baseline.py} ویژگی‌های واژه‌ای و نویسه‌ای \lr{TF-IDF} را می‌سازد، رگرسیون لجستیک متوازن را آموزش می‌دهد، مقدار \lr{C} را فقط با اعتبارسنجی پنج‌لایۀ آموزش انتخاب می‌کند و مدل منتخب را روی کل آموزش بازبرازش می‌دهد.
```

- [ ] **Step 4: توضیح جریان اجرا و خطاها**

پس از جدول، دو پاراگراف اضافه کن:

1. مسیر داده از بارگذاری تا پیش‌بینی‌های مستقیم/استدلالی/تشخیصی و سپس انتخاب نهایی؛
2. کنترل‌های پیاده‌سازی شامل ستون‌های ضروری، `unknown` برای خروجی مبهم، بازگشت مستقیم، کلید مرکب، ذخیرۀ CSV و اجرای ادامه‌پذیر.

- [ ] **Step 5: اجرای آزمون و بررسی نگارشی**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest tests.test_thesis_finalization -v
git diff --check
```

Expected: `OK` و بدون خطای whitespace.

- [ ] **Step 6: commit**

```powershell
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter4.tex tests/test_thesis_finalization.py
git commit -m "docs: explain thesis software implementation"
```

---

### Task 4: تعریف انتخاب‌گرها و بازآرایی پیوست بازتولید

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`
- Rewrite: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/appendix1.tex`
- Modify: `tests/test_thesis_finalization.py`

**Interfaces:**
- Produces: تعریف محلی سه روش پیش از جدول نتایج.
- Produces: پیوست فشرده با جدول تنظیمات، مراحل اجرا، ورودی/خروجی و فرمان‌های حداقلی.

- [ ] **Step 1: نوشتن آزمون تعریف روش‌ها و ساختار پیوست**

```python
def test_selector_comparison_methods_are_explained_before_table(self):
    text = (THESIS_DIR / "chapter5.tex").read_text(encoding="utf-8")
    table_pos = text.index(r"\label{tab:ch5-selector-comparison}")
    for phrase in [
        "اوراکل انتخاب منبع", "کران بالای تحلیلی",
        "هر زوج نمونه و منبع", "بیشترین امتیاز",
        "انتخاب‌گر فراسطح مبتنی بر درخت", "تقسیم داخلی آموزش",
    ]:
        self.assertGreaterEqual(text[:table_pos].find(phrase), 0)

def test_appendix_is_clean_reproduction_guide(self):
    text = (THESIS_DIR / "appendix1.tex").read_text(encoding="utf-8")
    for phrase in [
        "پیوست: راهنمای بازتولید آزمایش‌ها",
        "داده و کلید نمونه", "مدل‌ها و تنظیمات تولید",
        "مراحل بازتولید", "نیاز به اجرای مدل",
        "run_final_pipeline.py", "generate_thesis_result_figures.py",
        r"\fanum{۰٫۷}", "۹۰ نمونه",
    ]:
        self.assertIn(phrase, text)
    self.assertNotIn("۲۴۰", text)
```

- [ ] **Step 2: اجرای آزمون و تأیید RED**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest tests.test_thesis_finalization -v
```

Expected: شکست روی عنوان/ساختار جدید پیوست و تعریف‌های دقیق روش‌ها.

- [ ] **Step 3: تعریف سه روش پیش از جدول فصل پنجم**

در زیربخش مقایسۀ انتخاب‌گرها، پیش از جدول، سه بند کوتاه با عنوان پررنگ اضافه کن:

```latex
\textbf{رتبه‌بند لجستیک منبع.} هر زوج نمونه و منبع یک ردیف آموزشی است؛ مدل احتمال درست‌بودن هر منبع را برآورد می‌کند و منبع با بیشترین امتیاز انتخاب می‌شود.

\textbf{انتخاب‌گر فراسطح مبتنی بر درخت.} این مدل با ویژگی‌های خروجی منابع، توافق، نوع خطا، اطمینان، دامنه و رأی‌ها مستقیماً نام منبع مناسب را پیش‌بینی می‌کند؛ انتخاب مدل فقط با تقسیم داخلی آموزش انجام می‌شود.

\textbf{اوراکل انتخاب منبع.} اوراکل با دیدن برچسب طلایی برای هر نمونه یک منبع درست را برمی‌گزیند؛ بنابراین فقط کران بالای تحلیلی است و هنگام استنتاج قابل استفاده نیست.
```

- [ ] **Step 4: بازنویسی پیوست با ساختار فشرده**

`appendix1.tex` را با این ترتیب بازنویسی کن:

1. عنوان و یک پاراگراف هدف؛
2. «داده و کلید نمونه» با شمار ۲۱۸۸/۱۷۴۶/۴۴۲ و کلید مرکب؛
3. جدول «مدل‌ها و تنظیمات تولید» شامل `qwen3:8b`، دمای صفر مستقیم، دمای `\fanum{۰٫۷}` استدلال، دمای صفر برچسب نهایی، سه اجرای SC3 و مقایسۀ ۹۰نمونه‌ای Gemini؛
4. جدول «مراحل بازتولید» با ستون‌های مرحله، اسکریپت، خروجی و نیاز به اجرای مدل؛
5. فهرست فایل‌های اصلی؛
6. بلوک فرمان حداقلی برای `run_final_pipeline.py`، `generate_thesis_result_figures.py`، unittest و `build_thesis.ps1`؛
7. یک یادداشت پایانی که بازسازی جدول/شکل به اجرای LLM نیاز ندارد، ولی تولید دوبارۀ خروجی خام نیازمند پشتانۀ مدل است.

- [ ] **Step 5: اجرای آزمون و کنترل حجم**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest tests.test_thesis_finalization -v
git diff --check
```

Expected: `OK`. متن پیوست نباید توضیح تکراری فصل چهارم یا فهرست غیرضروری تمام اسکریپت‌های تحلیل را داشته باشد.

- [ ] **Step 6: commit**

```powershell
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/appendix1.tex tests/test_thesis_finalization.py
git commit -m "docs: clarify selectors and streamline reproduction appendix"
```

---

### Task 5: ساخت کامل، بازبینی بصری و ثبت PDF نهایی

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`
- Verify: تمام فایل‌های Tasks 1–4.

**Interfaces:**
- Consumes: شکل‌های بازسازی‌شده، متن‌های عددی اصلاح‌شده، فصل چهارم و پیوست جدید.
- Produces: PDF کامل ۶۰ تا ۶۵ صفحه و شواهد آزمون/بازبینی.

- [ ] **Step 1: اجرای کل آزمون‌ها**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s tests -q
```

Expected: تمام آزمون‌ها با `OK` پایان یابند.

- [ ] **Step 2: ساخت کامل پایان‌نامه**

Run:

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\build_thesis.ps1'
```

Expected: `Thesis PDF created successfully`، بدون ارجاع حل‌نشده و بدون `Overfull \hbox` جدید.

- [ ] **Step 3: کنترل تعداد صفحات و استخراج محل‌های بازبینی**

Run:

```powershell
$pdf='قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir\AUTthesis.pdf'
pdfinfo $pdf
pdftotext -layout $pdf thesis-final.txt
```

Expected: `Pages: 60..65`. با جست‌وجوی متن، صفحات دمای `۰٫۷`، جدول انتخاب‌گرها، زیربخش ساختار پیاده‌سازی و عنوان پیوست را تعیین کن.

- [ ] **Step 4: رندر و بازبینی صفحات کلیدی**

صفحات تعیین‌شده را با `pdftoppm -r 180` رندر کن و کنترل کن:

- اعداد جدول‌ها و شکل‌ها فارسی و بدون وارونگی‌اند؛
- دما دقیقاً `۰٫۷` است، نه `۷٫۰`؛
- نام مدل‌ها لاتین و سالم‌اند؛
- جدول ساختار نرم‌افزار و پیوست قطع‌شدگی یا هم‌پوشانی ندارند؛
- شکل‌های فصل پنجم درصد و محور خوانا دارند؛
- پیوست در فهرست با عنوان جدید دیده می‌شود.

- [ ] **Step 5: گیت نهایی و commit PDF**

Run:

```powershell
git diff --check
git status --short
```

Expected: فقط `AUTthesis.pdf` از ساخت نهایی تغییر کرده باشد و artifact موقت رندر باقی نماند.

```powershell
git add قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf
git commit -m "build: refresh thesis after numeral and documentation fixes"
```

- [ ] **Step 6: بازبینی کل بازۀ تغییر**

بازبین مستقل باید کل diff از commit پایۀ اجرای برنامه تا HEAD را بررسی و verdict صریح `CLEAN` یا `NEEDS_FIXES` بدهد. هیچ finding بحرانی یا مهمی برای تحویل قابل قبول نیست.
