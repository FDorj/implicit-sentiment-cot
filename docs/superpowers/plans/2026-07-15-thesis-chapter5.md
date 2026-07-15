# Thesis Chapter 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** نگارش کامل فصل پنجم پایان‌نامه با تکیه بر نتایج واقعی پروژه، درج چهار شکل تأییدشده و حفظ انسجام با فصل‌های اول تا چهارم.

**Architecture:** فصل در یک فایل موجود نوشته می‌شود و شواهد آن مستقیماً از فایل‌های سنجه و پیش‌بینی `results/` می‌آیند. نتایج آزمون کامل `Qwen3 8B` از مقایسۀ تکمیلی روی زیرمجموعۀ مشترک `Qwen3 8B` و `Gemini 2.5 Flash` جدا نگه داشته می‌شوند. شکل‌ها به‌صورت PDF برداری موجود وارد قالب می‌شوند و شکل رفتار انتخاب‌گر عمداً درج نمی‌شود.

**Tech Stack:** XeLaTeX، کلاس موجود `AUTthesis`، بسته‌های موجود قالب، PDFهای برداری فصل پنجم، PowerShell و ابزارهای اعتبارسنجی مخزن.

## Global Constraints

- `AUTthesis.cls`، فایل‌های سبک، فونت‌ها و تنظیمات قالب تغییر نکنند.
- فقط `chapter5.tex` برای محتوای فصل تغییر کند؛ `AUTthesis.pdf` محصول کامپایل است.
- همۀ اعداد نثر و جدول فارسی باشند؛ اعداد فرمول‌ها و نام مدل‌ها می‌توانند انگلیسی بمانند.
- مسیر `Flan-T5/HuggingFace` و هر اجرای ارزیابی‌نشده ذکر نشود.
- کنترل‌گر تصمیم میانی و انتخاب‌گر منبع تولیدکنندۀ برچسب نهایی معرفی شود.
- فقط چهار شکل تأییدشده درج شوند و `ch5_selector_behavior.pdf` وارد فصل نشود.
- اعشار نثر فارسی با دستور موجود و امن راست‌به‌چپ نوشته شوند.

---

### Task 1: تثبیت شواهد عددی فصل

**Files:**
- Read: `results/etc_thor_originalish_sc3_selected_isa_metrics.txt`
- Read: `results/meta_selector_metrics.txt`
- Read: `results/final_pipeline_validation.txt`
- Read: `results/final_qualitative_summary.txt`
- Read: `results/qwen_modelcmp_subset_metrics.txt`
- Read: `results/gemini_modelcmp_*_metrics.txt`
- Read: `experiments/generate_thesis_result_figures.py`

**Interfaces:**
- Consumes: فایل‌های سنجه و پیش‌بینی ثبت‌شده در مخزن.
- Produces: فهرست قطعی اعداد آزمون کامل، زیرمجموعۀ مشترک، ماتریس‌های درهم‌ریختگی و رفتار انتخاب‌گر برای استفاده در متن.

- [ ] **Step 1: استخراج سنجه‌های روش‌های Qwen روی آزمون کامل**

Run:

```powershell
python -m unittest tests.test_generate_thesis_result_figures -v
```

Expected: تمام آزمون‌های تولید شکل موفق باشند و داده‌های شکل‌ها با فایل‌های نتایج هم‌تراز بمانند.

- [ ] **Step 2: کنترل زنجیرۀ نهایی و شمار نمونه‌ها**

Run:

```powershell
Get-Content results/final_pipeline_validation.txt
Get-Content results/etc_thor_originalish_sc3_selected_isa_metrics.txt
```

Expected: تعداد کل ۲۱۸۸، آموزش ۱۷۴۶، آزمون ۴۴۲، نبود کلید تکراری، و ۲۱۸۸ برچسب نهایی معتبر.

- [ ] **Step 3: کنترل مرز مقایسۀ Gemini**

Run:

```powershell
Get-Content results/qwen_modelcmp_subset_metrics.txt
Get-Content results/gemini_modelcmp_validation_tuned_selected_metrics.txt
```

Expected: زیرمجموعۀ مشترک شامل ۱۵۰ نمونۀ آموزش و ۹۰ نمونۀ آزمون باشد و سنجه‌ها فقط در همین محدودۀ مشترک تفسیر شوند.

### Task 2: نگارش متن و جدول‌های فصل پنجم

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`

**Interfaces:**
- Consumes: ساختار فصل چهارم، شواهد تثبیت‌شدۀ Task 1 و قرارداد اصطلاحات فصل‌های قبلی.
- Produces: فصل فارسی کامل با بخش‌های مقدمه، تنظیمات، معیارها، نتایج اصلی، تحلیل مؤلفه‌ای، تحلیل کیفی، مقایسۀ تکمیلی و جمع‌بندی.

- [ ] **Step 1: جایگزینی اسکلت با ساختار نهایی**

ساختار دقیق:

```latex
\chapter{نتایج آزمایش‌ها}
\section{مقدمه}
\section{تنظیمات آزمایش}
\section{روش‌های مورد مقایسه و معیارهای ارزیابی}
\section{نتایج اصلی مدل \lr{Qwen3 8B} روی آزمون کامل}
\section{تحلیل انتخاب‌گر و مؤلفه‌های سامانه}
\section{تحلیل کیفی و خطا}
\section{مقایسۀ تکمیلی با \lr{Gemini 2.5 Flash}}
\section{ملاحظات اجرایی و محدودیت‌ها}
\section{جمع‌بندی}
```

- [ ] **Step 2: نوشتن تنظیمات و معیارها بدون تکرار فصل چهارم**

در این بخش شمار داده‌ها، دو دامنه، مدل‌ها، زیرساخت و جداسازی آموزش و آزمون نوشته شود. فرمول‌های دقت، دقت و یادآوری هر کلاس، اف‌یک هر کلاس و اف‌یک ماکرو درج شوند. توضیح شود که اف‌یک ماکرو برای وزن برابر کلاس‌ها استفاده می‌شود.

- [ ] **Step 3: درج جدول نتایج اصلی و تحلیل آن**

جدول باید روش‌های زیر را به همین ترتیب نشان دهد:

1. پیش‌بینی مستقیم `Qwen3 8B`
2. `THOR` ساده‌شده
3. `THOR` نزدیک به روش اصلی با خودسازگاری سه‌اجرایی
4. بازبینی ساده
5. کنترل‌گر استاندارد
6. کنترل‌گر روی `THOR` سه‌اجرایی
7. سامانۀ نهایی انتخاب منبع

متن باید اختلاف آزمون مستقیم و نهایی را دقیق گزارش کند: دقت از ۰٫۶۷۸۷۳۳ به ۰٫۷۲۳۹۸۲ و اف‌یک ماکرو از ۰٫۶۷۴۰۷۵ به ۰٫۷۱۹۲۰۴ رسیده است.

- [ ] **Step 4: نوشتن تحلیل انتخاب‌گر بدون شکل حذف‌شده**

یک جدول کوچک شامل انتخاب مستقیم ۳۰۰، `THOR` برابر ۱۴۲ و تشخیصی برابر صفر در آزمون درج شود. جدول یا متن گذار صحت نیز شامل ۲۶ اصلاح، ۶ تضعیف، ۲۹۴ هر دو درست و ۱۱۶ هر دو نادرست باشد.

- [ ] **Step 5: نوشتن مطالعات مؤلفه‌ای و تحلیل کیفی**

کنترل‌گر، انتخاب‌گر نهایی، انتخاب‌گر فراسطح و اوراکل با تفکیک «روش قابل اجرا» از «کران تحلیلی» مقایسه شوند. نمونه‌های کیفی فقط از خروجی واقعی انتخاب شوند و از ادعای تعمیم‌پذیری خودداری شود.

- [ ] **Step 6: نوشتن مقایسۀ تکمیلی Gemini**

زیرمجموعۀ متوازن ۲۴۰ نمونه‌ای و آزمون مشترک ۹۰ نمونه‌ای معرفی شود. مقایسه فقط با سنجه‌های همین آزمون انجام شود و محدودیت اندازۀ نمونه صریح باشد.

### Task 3: درج چهار شکل و ارجاع‌های متنی

**Files:**
- Modify: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`
- Read: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/Images/Chapter5/*.pdf`

**Interfaces:**
- Consumes: چهار PDF برداری تأییدشده.
- Produces: چهار محیط `figure` دارای عنوان فارسی، برچسب یکتا و ارجاع پیش از شکل.

- [ ] **Step 1: درج شکل مقایسۀ روش‌های Qwen**

Use:

```latex
\includegraphics[width=\textwidth]{Images/Chapter5/ch5_qwen_full_test_methods.pdf}
```

- [ ] **Step 2: درج ماتریس‌های Qwen**

Use:

```latex
\includegraphics[width=\textwidth]{Images/Chapter5/ch5_direct_vs_final_confusion.pdf}
```

- [ ] **Step 3: درج مقایسۀ Qwen و Gemini**

Use:

```latex
\includegraphics[width=\textwidth]{Images/Chapter5/ch5_qwen_gemini_shared_subset.pdf}
```

- [ ] **Step 4: درج ماتریس‌های Gemini**

Use:

```latex
\includegraphics[width=\textwidth]{Images/Chapter5/ch5_gemini_direct_vs_selected_confusion.pdf}
```

- [ ] **Step 5: اثبات حذف شکل رفتار انتخاب‌گر**

Run:

```powershell
rg -n "ch5_selector_behavior" قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex
```

Expected: بدون خروجی.

### Task 4: بازبینی نگارشی و ساخت خروجی

**Files:**
- Modify if needed: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex`
- Generated: `قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/AUTthesis.pdf`

**Interfaces:**
- Consumes: فصل نوشته‌شده و چهار شکل درج‌شده.
- Produces: فصل کامپایل‌شده با ارجاع‌های حل‌شده و گزارش بازبینی.

- [ ] **Step 1: کنترل اعداد انگلیسی ناخواسته و اصطلاحات ممنوع**

Run:

```powershell
rg -n "Flan|HuggingFace|[0-9]" قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir/chapter5.tex
```

Expected: هیچ اشاره‌ای به مسیر ممنوع وجود نداشته باشد؛ اعداد انگلیسی فقط در فرمول‌ها، برچسب‌ها، مسیر فایل‌ها و نام مدل‌ها دیده شوند.

- [ ] **Step 2: کامپایل کامل دوبارۀ پایان‌نامه**

Run from the template directory:

```powershell
xelatex -interaction=nonstopmode -halt-on-error AUTthesis.tex
xelatex -interaction=nonstopmode -halt-on-error AUTthesis.tex
```

Expected: هر دو اجرا با کد خروج صفر پایان یابند و `AUTthesis.pdf` ساخته شود.

- [ ] **Step 3: کنترل خطاها و ارجاع‌های حل‌نشده**

Run:

```powershell
rg -n "LaTeX Error|Undefined control sequence|Reference .* undefined|Citation .* undefined|multiply defined" AUTthesis.log
```

Expected: بدون خروجی.

- [ ] **Step 4: بازبینی تصویری صفحات فصل پنجم**

صفحات فصل پنجم از PDF به PNG تبدیل و از نظر بریدگی شکل‌ها، وارونگی اعشار، تراکم جدول، قرارگیری عنوان‌ها و خوانایی چهار شکل بررسی شوند. هر مشکل مشاهده‌شده در `chapter5.tex` اصلاح و کامپایل تکرار شود.
