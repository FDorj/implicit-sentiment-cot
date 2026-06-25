# گزارش نهایی پروژه تحلیل احساس ضمنی

## 1. خلاصه پروژه

عنوان رسمی پروژه:

```text
Development of a System for Implicit Sentiment Analysis Using Multi-Step Chain-of-Thought Reasoning
```

هدف پروژه، طراحی و پیاده‌سازی یک سیستم برای **Implicit Sentiment Analysis (ISA)** است؛ یعنی تشخیص polarity نسبت به یک target، در حالتی که احساس معمولاً به‌صورت مستقیم بیان نشده و باید از متن و قرائن ضمنی فهمیده شود.

در این پروژه، ابتدا یک baseline مستقیم با Qwen3 8B ساخته شد. سپس یک pipeline چندمرحله‌ای الهام‌گرفته از THOR پیاده‌سازی شد. بعد از تحلیل خطا، مشخص شد THOR خام در setting فعلی از baseline مستقیم ضعیف‌تر است. بنابراین پروژه به سمت یک سیستم ترکیبی رفت که از Direct، THOR، diagnostic reflection و یک سیاست انتخاب منبع یادگرفته‌شده از train استفاده می‌کند.

نکته مهم: در نتایج فعلی هیچ fine-tuning انجام نشده است. مدل Qwen3 8B فقط به‌صورت prompt-based از طریق Ollama استفاده شده و وزن‌های مدل تغییر نکرده‌اند.

## 2. داده و مسئله

داده پروژه از نسخه SCAPT-labeled داده SemEval 2014 Laptop و Restaurant ساخته شده است. parser داده در `src/data_loader.py` نوشته شده و خروجی‌های پردازش‌شده در `data/processed/` قرار دارند.

فایل اصلی آزمایش‌ها:

```text
data/processed/semeval14_scapt_isa_only_clean.csv
```

آمار داده ISA-only clean:

| split | تعداد |
| --- | ---: |
| train | 1746 |
| test | 442 |
| کل | 2188 |

توزیع polarity در کل داده:

| polarity | تعداد |
| --- | ---: |
| neutral | 972 |
| negative | 619 |
| positive | 597 |

## 3. THOR چیست و چرا برای پروژه مهم بود؟

THOR یک رویکرد چندمرحله‌ای برای تحلیل احساس ضمنی است. ایده اصلی آن این است که به‌جای اینکه مدل مستقیماً polarity را پیش‌بینی کند، تصمیم را به چند گام reasoning تقسیم کند:

```text
aspect reasoning -> opinion reasoning -> polarity reasoning
```

این ایده برای ISA جذاب است، چون در احساس ضمنی معمولاً label از یک کلمه احساسی مستقیم به دست نمی‌آید. مدل باید بفهمد target از چه جنبه‌ای مطرح شده، چه opinion یا clue ضمنی وجود دارد، و آن clue چه polarityای نسبت به target می‌سازد.

اما نتیجه مهم پروژه این بود که **THOR خام الزاماً بهتر از Direct نیست**. در آزمایش‌های ما، THOR به‌تنهایی interpretability بیشتری داد، اما به دلیل خطاهای مرحله‌ای مانند aspect اشتباه، opinion نامرتبط یا over-interpretation، از Direct ضعیف‌تر شد.

## 4. تفاوت پروژه ما با THOR اصلی

این پروژه reproduction دقیق کد رسمی THOR نیست؛ بلکه یک پیاده‌سازی THOR-style / original-ish داخل سیستم خودمان است.

تفاوت‌های اصلی:

| مورد | THOR اصلی | پروژه فعلی |
| --- | --- | --- |
| مدل اصلی | setup مقاله، از جمله Flan-T5 و ارزیابی‌های جداگانه | Qwen3 8B از طریق Ollama |
| training | در مقاله مسیر fine-tuning هم وجود دارد | fine-tuning انجام نشده |
| هدف پیاده‌سازی | اجرای روش THOR | ساخت سیستم کامل ISA با THOR به‌عنوان یکی از منابع تصمیم |
| خروجی نهایی | معمولاً label حاصل از مسیر THOR | انتخاب بین Direct، THOR و diagnostic |
| افزونه‌های پروژه | ندارد یا جزو THOR خام نیست | Error-Type Reflection، Controller، Train-Calibrated Source Selection |

بنابراین اگر پرسیده شود «اگر کد THOR موجود بود، چرا خودمان پیاده‌سازی کردیم؟»، پاسخ این است که هدف پروژه فقط اجرای THOR نبود. هدف این بود که THOR را در setting مدل و داده خودمان بازسازی، ارزیابی و سپس با تحلیل خطا و controller به یک سیستم قوی‌تر تبدیل کنیم.

## 5. سیستم نهایی پیشنهادی

سیستم نهایی را نباید این‌طور فهمید که همه مراحل همیشه به‌صورت خطی و کورکورانه label قبلی را جایگزین می‌کنند. سیستم نهایی چند source می‌سازد و در پایان تصمیم می‌گیرد کدام source قابل اعتمادتر است.

نمای دقیق‌تر:

```text
Direct prediction
        \
         -> Train-Calibrated Source Selection -> Final prediction
        /
THOR original-ish SC3 prediction
        \
         -> Error-Type-Aware Reflection -> Controller / diagnostic signal
```

اجزای اصلی:

| جزء | نقش |
| --- | --- |
| Direct Qwen3 8B | baseline مستقیم و یکی از sourceهای اصلی تصمیم |
| THOR original-ish SC3 | reasoning چندمرحله‌ای با سه نمونه self-consistency |
| Error-Type-Aware Reflection | تشخیص نوع خطا، label پیشنهادی و confidence در موارد اختلاف |
| Controller | تصمیم rule-based اولیه بر اساس Direct، THOR و diagnostic |
| Train-Calibrated Source Selection | یادگیری از train برای انتخاب source مناسب در هر profile |

کلیدهای policy فعلی:

```text
direct_prediction
error_type
diagnostic_confidence
domain
```

این مرحله fine-tuning نیست. فقط یک جدول تصمیم‌گیری از split train ساخته می‌شود و روی test بدون استفاده از gold label اعمال می‌شود.

## 6. مسیر پیاده‌سازی

مهم‌ترین اجزای پیاده‌سازی:

| فایل/پوشه | نقش |
| --- | --- |
| `src/data_loader.py` | parse و آماده‌سازی داده |
| `src/prompt_runner.py` | اجرای prompt با backendهای Ollama و scaffold مربوط به HuggingFace |
| `src/thor_pipeline.py` | اجرای مسیر aspect، opinion، polarity reasoning و label |
| `src/reflection_pipeline.py` | reflection ساده و Error-Type-Aware Reflection |
| `src/controller.py` | منطق انتخاب label و error typeها |
| `experiments/run_direct.py` | اجرای baseline مستقیم |
| `experiments/run_thor.py` | اجرای THOR simplified |
| `experiments/run_thor_originalish.py` | اجرای THOR original-ish |
| `experiments/run_thor_self_consistency.py` | اجرای self-consistency |
| `experiments/run_error_type_controller.py` | اجرای ETC |
| `experiments/apply_etc_policy.py` | اعمال manual یا train-calibrated policy |
| `experiments/run_final_pipeline.py` | ساخت جدول نهایی و validation از خروجی‌های ذخیره‌شده |
| `experiments/extract_qualitative_examples.py` | استخراج نمونه‌های کیفی |
| `tests/test_core_logic.py` | تست‌های کوچک برای منطق‌های اصلی |

## 7. نتایج نهایی

نتایج اصلی باید روی split تست گزارش شوند، چون train برای policy selection استفاده شده است.

| روش | test accuracy | test macro-F1 |
| --- | ---: | ---: |
| Direct Qwen3 8B | 0.678733 | 0.674075 |
| THOR simplified | 0.599548 | 0.578437 |
| Simple reflection | 0.615385 | 0.606732 |
| ETC standard | 0.660633 | 0.659430 |
| THOR original-ish SC3 | 0.590498 | 0.600690 |
| ETC over original-ish SC3 | 0.660633 | 0.662108 |
| Final selected pipeline | 0.723982 | 0.719204 |

برداشت اصلی:

- Direct baseline قوی‌تر از THOR خام بود.
- THOR original-ish SC3 به‌تنهایی بهترین روش نبود.
- Error-Type Reflection و Controller خروجی THOR را بهتر کردند، اما کافی نبودند.
- بهترین نتیجه با Train-Calibrated Source Selection به دست آمد.
- سیستم نهایی نسبت به Direct، از نظر test macro-F1 حدود `+0.045129` بهتر شد.

## 8. تحلیل کیفی نسبت به Direct

برای اینکه نتیجه فقط یک عدد کلی نباشد، خروجی نهایی با Direct روی split تست مقایسه شد.

| وضعیت | تعداد |
| --- | ---: |
| Direct غلط بود و سیستم نهایی درست کرد | 26 |
| Direct درست بود و سیستم نهایی غلط کرد | 6 |
| هر دو درست بودند | 294 |
| هر دو غلط بودند | 116 |

این نشان می‌دهد سیستم نهایی فقط از نظر metric بهتر نشده، بلکه در نمونه‌های واقعی هم تعداد بیشتری از خطاهای Direct را اصلاح کرده تا اینکه خطای جدید ایجاد کند.

نمونه‌های دقیق در فایل‌های زیر ذخیره شده‌اند:

```text
results/final_qualitative_examples.csv
thesis_notes/final_qualitative_examples_fa.md
```

## 9. اعتبارسنجی pipeline نهایی

اسکریپت `experiments/run_final_pipeline.py` مدل را دوباره اجرا نمی‌کند. این اسکریپت خروجی‌های ذخیره‌شده را validate می‌کند و جدول‌های نهایی را می‌سازد.

نتایج validation:

| مورد | مقدار |
| --- | --- |
| row_count | 2188 |
| all_row_counts_equal | True |
| duplicate keys | 0 |
| direct alignment mismatches | 0 |
| thor alignment mismatches | 0 |
| selected valid labels | 2188 |

این یعنی فایل‌های Direct، THOR، ETC و final selected از نظر ردیف‌ها و کلیدهای اصلی با هم هم‌تراز هستند و خروجی نهایی label نامعتبر ندارد.

## 10. چرا الان fine-tune نکردیم؟

fine-tune کردن از نظر فنی ممکن است، اما در وضعیت فعلی بهترین مسیر اصلی پروژه نبود.

دلایل:

- setup فعلی با Qwen3 8B و Ollama برای inference طراحی شده، نه fine-tuning.
- برای fine-tuning باید مسیر HuggingFace/LoRA/QLoRA جدا ساخته شود.
- داده ISA-only فقط 2188 نمونه دارد و train آن 1746 نمونه است؛ برای fine-tune کردن LLM، ریسک overfitting وجود دارد.
- برای آموزش کامل مسیر THOR، gold label برای aspect reasoning، opinion reasoning، polarity reasoning و diagnostic error type نداریم.
- اگر reasoningها را pseudo-label کنیم، کیفیت training به کیفیت خروجی مدل تولیدکننده وابسته می‌شود.
- سیستم فعلی بدون fine-tuning توانسته از Direct و THOR خام بهتر شود، که برای پروژه کارشناسی یک نتیجه قابل دفاع است.

بنابراین fine-tuning بهتر است به‌عنوان future work مطرح شود، نه هسته اصلی نسخه فعلی.

## 11. محدودیت‌ها

- مدل زبانی fine-tune نشده است.
- Flan-T5/HuggingFace در کد scaffold دارد، اما در نتایج فعلی تست یا ارزیابی نشده است.
- OpenAI API یا API خارجی در نتایج فعلی استفاده نشده است.
- THOR پیاده‌سازی‌شده reproduction دقیق کد رسمی THOR نیست؛ نسخه THOR-style / original-ish سازگار با پروژه است.
- `run_final_pipeline.py` و `extract_qualitative_examples.py` inference جدید انجام نمی‌دهند و فقط خروجی‌های ذخیره‌شده را summarize و validate می‌کنند.

## 12. کارهای آینده

مسیرهای منطقی برای ادامه:

- اجرای baseline fine-tuned با Flan-T5 برای مقایسه با THOR اصلی
- اجرای LoRA/QLoRA برای Qwen در صورت داشتن زمان و GPU مناسب
- مقایسه اختیاری با API models
- ساخت gold یا pseudo-gold reasoning labels برای آموزش مرحله‌ای
- تحلیل خطای عمیق‌تر برای موارد `both_wrong`
- بهبود policy selection با validation split جداگانه

## 13. پاسخ‌های آماده برای دفاع

### چرا THOR خودش کافی نبود؟

چون در setting فعلی، THOR خام از Direct ضعیف‌تر شد. دلیل احتمالی این است که decomposition خطاهای مرحله‌ای ایجاد می‌کند: aspect اشتباه، opinion نامرتبط، over-interpretation یا ناسازگاری reasoning و label. بنابراین THOR برای تولید trace مفید بود، اما برای label نهایی به تنهایی کافی نبود.

### پروژه نسبت به THOR چه چیز جدیدی دارد؟

پروژه THOR را به‌عنوان یکی از sourceهای تصمیم استفاده می‌کند، نه جواب نهایی قطعی. نوآوری عملی پروژه در ترکیب Direct، THOR، Error-Type Reflection، Controller و Train-Calibrated Source Selection است.

### چرا از کد رسمی THOR مستقیم استفاده نشد؟

چون هدف پروژه فقط اجرای کد مقاله نبود. مدل، داده، backend و هدف سیستم متفاوت بود. برای اتصال THOR به Qwen/Ollama، reflection، controller، policy selection، validation و تحلیل کیفی، پیاده‌سازی سازگار با پروژه لازم بود.

### آیا پروژه zero-shot است یا fine-tuned؟

مدل زبانی fine-tune نشده است. اجرای مدل prompt-based است. اما policy نهایی از train برای انتخاب source استفاده می‌کند؛ این train-calibrated source selection است، نه fine-tuning مدل زبانی.

### آیا پیاده‌سازی برای پروژه کارشناسی کافی است؟

بله. پروژه شامل آماده‌سازی داده، baseline، THOR-style reasoning، self-consistency، reflection، diagnostic parsing، controller، policy ablation، train-calibrated source selection، جدول نتایج، validation، تست‌های کوچک و تحلیل کیفی است.

## 14. دستورهای بازتولیدپذیر

اجرای تست‌ها:

```powershell
python -B -m unittest discover -s tests -v
```

ساخت جدول نهایی و validation:

```powershell
python -B experiments/run_final_pipeline.py
```

استخراج نمونه‌های کیفی:

```powershell
python -B experiments/extract_qualitative_examples.py
```

## 15. جمع‌بندی

نتیجه نهایی پروژه این است که در تحلیل احساس ضمنی، reasoning چندمرحله‌ای به‌تنهایی تضمین‌کننده عملکرد بهتر نیست. THOR به مدل کمک می‌کند trace قابل توضیح بسازد، اما ممکن است خطاهای مرحله‌ای ایجاد کند. سیستم پیشنهادی پروژه با استفاده از تحلیل خطا، reflection ساختاریافته، controller و انتخاب منبع کالیبره‌شده با train، از Direct و THOR خام بهتر عمل کرده است.

بهترین نتیجه فعلی روی test:

```text
Final selected pipeline
Accuracy: 0.723982
Macro-F1: 0.719204
```
