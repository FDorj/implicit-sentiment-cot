# گزارش نهایی پروژه تحلیل احساس ضمنی

## 1. خلاصه پروژه

عنوان رسمی پروژه:

```text
Development of a System for Implicit Sentiment Analysis Using Multi-Step Chain-of-Thought Reasoning
```

هدف پروژه، طراحی و پیاده‌سازی یک سیستم برای **Implicit Sentiment Analysis (ISA)** است؛ یعنی تشخیص polarity نسبت به یک target، در حالتی که احساس معمولاً به‌صورت مستقیم بیان نشده و باید از متن و قرائن ضمنی فهمیده شود.

در این پروژه، ابتدا یک baseline مستقیم با Qwen3 8B ساخته شد. سپس یک pipeline چندمرحله‌ای الهام‌گرفته از THOR پیاده‌سازی و ارزیابی شد تا نقش reasoning مرحله‌ای در ISA بررسی شود. نتایج نشان دادند که reasoning چندمرحله‌ای در این تنظیمات به‌تنهایی بهترین عملکرد عددی را ایجاد نمی‌کند، اما trace قابل تحلیل و سیگنال مکمل ارزشمندی فراهم می‌کند. به همین دلیل، نسخه نهایی پروژه به شکل یک سیستم ترکیبی طراحی شد که از Direct، THOR-style reasoning، diagnostic reflection و یک سیاست انتخاب منبع یادگرفته‌شده از train استفاده می‌کند.

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

## 3. جایگاه THOR در پروژه

THOR یک رویکرد چندمرحله‌ای برای تحلیل احساس ضمنی است. ایده اصلی آن این است که به‌جای اینکه مدل مستقیماً polarity را پیش‌بینی کند، تصمیم را به چند گام reasoning تقسیم کند:

```text
aspect reasoning -> opinion reasoning -> polarity reasoning
```

این ایده برای ISA جذاب است، چون در احساس ضمنی معمولاً label از یک کلمه احساسی مستقیم به دست نمی‌آید. مدل باید بفهمد target از چه جنبه‌ای مطرح شده، چه opinion یا clue ضمنی وجود دارد، و آن clue چه polarityای نسبت به target می‌سازد.

در پروژه حاضر، THOR به‌عنوان مبنای reasoning و یکی از baselineهای تحلیلی استفاده شد. خروجی این مرحله علاوه بر label، traceهایی مانند aspect، opinion و reasoning تولید می‌کند. همین traceها بعداً برای reflection، تحلیل خطا و تصمیم‌گیری چندمنبعی استفاده شدند. از نظر عددی، نسخه THOR-style در این setting نسبت به Direct عملکرد پایین‌تری داشت؛ این نتیجه به‌عنوان نشانه‌ای برای طراحی مرحله‌های تکمیلی مانند reflection، controller و source selection استفاده شد.

## 4. نسبت پروژه با THOR اصلی

این پروژه reproduction دقیق کد رسمی THOR نیست؛ بلکه یک پیاده‌سازی THOR-style / original-ish است که با داده، مدل و زیرساخت اجرایی پروژه حاضر سازگار شده است. هدف، استفاده از ایده reasoning مرحله‌ای و سپس بررسی این بود که چگونه می‌توان آن را در یک سیستم کامل‌تر برای ISA به کار گرفت.

تفاوت‌های اصلی:

| مورد | THOR اصلی | پروژه فعلی |
| --- | --- | --- |
| مدل اصلی | setup مقاله، از جمله Flan-T5 و ارزیابی‌های جداگانه | Qwen3 8B از طریق Ollama |
| training | در مقاله مسیر fine-tuning هم وجود دارد | fine-tuning انجام نشده |
| هدف پیاده‌سازی | اجرای روش THOR | ساخت سیستم کامل ISA با THOR به‌عنوان یکی از منابع تصمیم |
| خروجی نهایی | معمولاً label حاصل از مسیر THOR | انتخاب بین Direct، THOR و diagnostic |
| افزونه‌های پروژه | تمرکز اصلی روی reasoning مرحله‌ای | Error-Type Reflection، Controller، Train-Calibrated Source Selection |

به همین دلیل، برای نیازهای این پروژه یک پیاده‌سازی سازگار با محیط آزمایش انتخاب شد: مدل، backend، داده پردازش‌شده، خروجی‌های diagnostic و سیاست انتخاب نهایی در این پروژه متفاوت‌اند. پیاده‌سازی مستقل باعث شد هر مرحله با فایل‌های خروجی، تحلیل خطا، validation و گزارش نهایی همین پروژه یکپارچه شود.

## 5. سیستم نهایی پیشنهادی

سیستم نهایی به‌صورت یک معماری چندمنبعی طراحی شده است. ابتدا چند سیگنال پیش‌بینی تولید می‌شوند و سپس مرحله انتخاب نهایی تصمیم می‌گیرد برای هر profile کدام source قابل اعتمادتر است.

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

## 6. منطق انتخاب‌های طراحی

در این پروژه هر جزء برای پاسخ دادن به یک نیاز مشخص اضافه شده است:

| انتخاب طراحی | دلیل |
| --- | --- |
| Direct baseline | لازم بود یک مرجع ساده و قابل مقایسه داشته باشیم تا ارزش مراحل بعدی سنجیده شود. |
| THOR-style reasoning | برای ISA مناسب است، چون مسئله نیازمند فهم aspect، opinion ضمنی و polarity است، نه فقط تشخیص کلمه احساسی. |
| Self-consistency | برای کاهش نوسان خروجی reasoning، مسیر THOR original-ish چند بار اجرا و با majority vote جمع‌بندی شد. |
| Error-Type Reflection | به‌جای یک reflection آزاد، خروجی diagnostic ساختاریافته تولید شد تا نوع خطا، label پیشنهادی و confidence قابل استفاده باشند. |
| Controller | reflection به‌تنهایی ممکن است تصمیم‌های ناپایدار بسازد؛ controller منطق تصمیم‌گیری را قابل توضیح و قابل کنترل کرد. |
| Train-Calibrated Source Selection | نتایج نشان داد هیچ sourceای در همه حالت‌ها بهترین نیست؛ بنابراین انتخاب source بر اساس الگوهای train انجام شد. |
| Test-only reporting | چون policy از train استفاده می‌کند، نتیجه اصلی باید روی test گزارش شود تا ارزیابی منصفانه باشد. |

این ساختار باعث می‌شود پروژه فقط یک prompt یا یک baseline نباشد، بلکه یک pipeline قابل تحلیل و قابل بازتولید برای ISA باشد.

## 7. مسیر پیاده‌سازی

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

## 8. نتایج نهایی

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

تفسیر نتیجه:

- Direct baseline مرجع قدرتمندی برای مقایسه فراهم کرد.
- THOR-style reasoning در این setting به‌تنهایی بهترین عدد را نداد، اما trace و سیگنال مکمل تولید کرد.
- Error-Type Reflection و Controller نسبت به reasoning تنها، تصمیم‌گیری را ساختاریافته‌تر کردند.
- بهترین نتیجه زمانی به دست آمد که سیستم بین Direct، THOR و diagnostic بر اساس الگوهای train انتخاب کرد.
- سیستم نهایی نسبت به Direct، از نظر test macro-F1 حدود `+0.045129` بهبود داشت.

## 9. تحلیل کیفی نسبت به Direct

برای اینکه نتیجه فقط یک عدد کلی نباشد، خروجی نهایی با Direct روی split تست مقایسه شد.

| وضعیت | تعداد |
| --- | ---: |
| Direct نادرست بود و سیستم نهایی درست پیش‌بینی کرد | 26 |
| Direct درست بود و سیستم نهایی نادرست پیش‌بینی کرد | 6 |
| هر دو درست بودند | 294 |
| هر دو غلط بودند | 116 |

این تحلیل نشان می‌دهد بهبود نهایی فقط در سطح metric کلی نیست؛ در split تست، تعداد مواردی که سیستم نهایی خطای Direct را اصلاح کرده، بیش از مواردی است که تصمیم Direct را به خطا تبدیل کرده است.

نمونه‌های دقیق در فایل‌های زیر ذخیره شده‌اند:

```text
results/final_qualitative_examples.csv
thesis_notes/final_qualitative_examples_fa.md
```

## 10. اعتبارسنجی pipeline نهایی

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

## 11. جایگاه fine-tuning در پروژه

fine-tune کردن از نظر فنی ممکن است، اما در نسخه فعلی به‌عنوان مسیر اصلی انتخاب نشد. دلیل این تصمیم فقط محدودیت اجرایی نبود؛ بلکه به طراحی پژوهش هم مربوط است.

دلایل:

- setup فعلی با Qwen3 8B و Ollama برای inference طراحی شده، نه fine-tuning.
- برای fine-tuning باید مسیر HuggingFace/LoRA/QLoRA جدا ساخته شود.
- داده ISA-only فقط 2188 نمونه دارد و train آن 1746 نمونه است؛ برای fine-tune کردن LLM، ریسک overfitting وجود دارد.
- برای آموزش کامل مسیر THOR، gold label برای aspect reasoning، opinion reasoning، polarity reasoning و diagnostic error type نداریم.
- اگر reasoningها را pseudo-label کنیم، کیفیت training به کیفیت خروجی مدل تولیدکننده وابسته می‌شود.
- سیستم فعلی بدون fine-tuning توانسته نسبت به baseline مستقیم و THOR-style reasoning نتیجه بهتری بدهد؛ بنابراین contribution اصلی نسخه فعلی روی طراحی pipeline و انتخاب منبع متمرکز است.

بنابراین fine-tuning بهتر است به‌عنوان future work مطرح شود، نه هسته اصلی نسخه فعلی.

## 12. محدودیت‌ها

- مدل زبانی fine-tune نشده است.
- Flan-T5/HuggingFace در کد scaffold دارد، اما در نتایج فعلی تست یا ارزیابی نشده است.
- API خارجی فقط برای مقایسه تکمیلی Gemini روی subset متوازن استفاده شده است؛ نتیجه اصلی full-test پروژه با Qwen/Ollama است.
- THOR پیاده‌سازی‌شده reproduction دقیق کد رسمی THOR نیست؛ نسخه THOR-style / original-ish سازگار با پروژه است.
- `run_final_pipeline.py` و `extract_qualitative_examples.py` inference جدید انجام نمی‌دهند و فقط خروجی‌های ذخیره‌شده را summarize و validate می‌کنند.

## 13. کارهای آینده

مسیرهای منطقی برای ادامه:

- اجرای baseline fine-tuned با Flan-T5 برای مقایسه با THOR اصلی
- اجرای LoRA/QLoRA برای Qwen در صورت داشتن زمان و GPU مناسب
- گسترش مقایسه Gemini/API از subset متوازن به چند subset مستقل یا اجرای کامل‌تر در صورت داشتن بودجه
- ساخت gold یا pseudo-gold reasoning labels برای آموزش مرحله‌ای
- تحلیل خطای عمیق‌تر برای موارد `both_wrong`
- بهبود policy selection با validation split جداگانه

## 13.1. مقایسه تکمیلی با Gemini 2.5 Flash

بعد از تثبیت نتیجه اصلی Qwen، یک آزمایش تکمیلی با `Gemini 2.5 Flash` از طریق backend سازگار با OpenAI اجرا شد. به دلیل هزینه و زمان زیاد THOR self-consistency، این مقایسه روی یک subset متوازن انجام شد:

```text
data/processed/gemini_model_comparison_subset_train150_test90.csv
```

منطق ساخت subset:

- از split رسمی train، برای هر ترکیب `domain/polarity` تعداد `25` نمونه انتخاب شد.
- از split رسمی test، برای هر ترکیب `domain/polarity` تعداد `15` نمونه انتخاب شد.
- دو domain وجود دارد: `laptop` و `restaurant`.
- سه polarity وجود دارد: `positive`, `negative`, `neutral`.
- بنابراین train برابر `2 * 3 * 25 = 150` و test برابر `2 * 3 * 15 = 90` شد.
- seed انتخاب نمونه‌ها `20260709` بود.

این کار train/test را با هم قاطی نمی‌کند. فقط subset کوچک‌تر و متوازن‌تری از همان splitهای رسمی ساخته می‌شود.

نتایج Gemini روی subset:

| روش | overall accuracy | overall macro-F1 | test accuracy | test macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Gemini direct | 0.754167 | 0.741729 | 0.811111 | 0.804886 |
| Gemini THOR original-ish SC3 | 0.787500 | 0.775404 | 0.733333 | 0.716109 |
| Gemini ETC controller | 0.758333 | 0.745482 | 0.811111 | 0.804886 |
| Gemini validation-tuned selected profile | 0.825000 | 0.818858 | 0.811111 | 0.808340 |

همچنین یک `Regularized Logistic Regression` به‌عنوان offline meta-selector آزمایش شد. این مدل روی subset Gemini در عمل همه نمونه‌ها را به THOR سپرد و روی test به macro-F1 برابر `0.716109` رسید؛ بنابراین از profile selection بهتر نبود.

برداشت: Gemini به‌عنوان مدل قوی‌تر، direct baseline بهتری نسبت به Qwen روی این subset داد. اما THOR با Gemini روی test افت کرد و diagnostic هم با وجود اصلاح format، به‌عنوان source نهایی انتخاب نشد. بهترین نتیجه Gemini از validation-tuned profile به دست آمد، ولی بهبود آن نسبت به Gemini direct کوچک بود. بنابراین این آزمایش برای مقایسه مدل‌ها مفید است، اما نتیجه اصلی full-test پروژه همچنان Qwen final pipeline است.

## 14. پرسش‌های احتمالی و پاسخ‌های کوتاه

### نقش THOR در سیستم چیست؟

THOR ایده اصلی reasoning مرحله‌ای را فراهم می‌کند. در پروژه حاضر، این مسیر برای تولید trace و یک source پیش‌بینی استفاده شده است. نتیجه عددی آن به‌تنهایی بهترین نبود، اما برای reflection، تحلیل خطا و انتخاب منبع نهایی نقش مهمی داشت.

### افزوده اصلی پروژه چیست؟

افزوده اصلی پروژه، طراحی یک pipeline چندمنبعی است که Direct، THOR-style reasoning و diagnostic reflection را کنار هم قرار می‌دهد و سپس با Train-Calibrated Source Selection تصمیم نهایی را انتخاب می‌کند.

### چرا از کد رسمی THOR مستقیم استفاده نشد؟

چون هدف پروژه فقط اجرای کد مقاله نبود. مدل، داده، backend و هدف سیستم متفاوت بود. برای اتصال THOR-style reasoning به Qwen/Ollama، reflection، controller، policy selection، validation و تحلیل کیفی، پیاده‌سازی سازگار با پروژه لازم بود.

### آیا پروژه zero-shot است یا fine-tuned؟

مدل زبانی fine-tune نشده است. اجرای مدل prompt-based است. اما policy نهایی از train برای انتخاب source استفاده می‌کند؛ این train-calibrated source selection است، نه fine-tuning مدل زبانی.

### subset 240تایی Gemini چطور ساخته شد؟

از split رسمی train و test نمونه‌برداری stratified انجام شد. برای train از هر ترکیب `domain/polarity` تعداد 25 نمونه و برای test از هر ترکیب `domain/polarity` تعداد 15 نمونه انتخاب شد. چون 2 domain و 3 polarity داریم، train برابر 150 و test برابر 90 شد. نمونه‌های test در ساخت profile دیده نشده‌اند.

### آیا عوض کردن نسبت train/test می‌تواند نتیجه را بهتر کند؟

ممکن است عددها را تغییر بدهد، اما اگر بعد از دیدن test این نسبت را عوض کنیم، از نظر علمی شبیه test leakage می‌شود. راه درست این است که یا همین split از قبل تعریف‌شده را نگه داریم، یا چند subset/seed مستقل را از قبل مشخص کنیم و میانگین بگیریم. train بیشتر می‌تواند profile را پایدارتر کند، ولی test کوچک‌تر ارزیابی را پرنوسان‌تر می‌کند.

### آیا پیاده‌سازی برای پروژه کارشناسی کافی است؟

بله. پروژه شامل آماده‌سازی داده، baseline، THOR-style reasoning، self-consistency، reflection، diagnostic parsing، controller، policy ablation، train-calibrated source selection، جدول نتایج، validation، تست‌های کوچک و تحلیل کیفی است.

## 15. دستورهای بازتولیدپذیر

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

## 16. جمع‌بندی

نتیجه نهایی پروژه این است که در تحلیل احساس ضمنی، reasoning چندمرحله‌ای زمانی بیشترین ارزش را دارد که در کنار baseline مستقیم، reflection ساختاریافته، controller و انتخاب منبع استفاده شود. THOR-style reasoning trace قابل توضیح فراهم کرد؛ Direct baseline مرجع عددی قدرتمندی بود؛ و Train-Calibrated Source Selection توانست از این سیگنال‌ها برای تصمیم نهایی استفاده کند.

بهترین نتیجه فعلی روی test:

```text
Final selected pipeline
Accuracy: 0.723982
Macro-F1: 0.719204
```
