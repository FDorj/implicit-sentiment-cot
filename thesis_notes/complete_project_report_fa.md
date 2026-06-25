# گزارش کامل روند توسعه پروژه ISA

> یادداشت مهم: این فایل نقش گزارش تاریخی روند توسعه را دارد. برای متن نهایی و دفاع‌محور پروژه، فایل `thesis_notes/final_project_report_fa.md` مرجع اصلی است. بخش‌های قدیمی‌تر این فایل مسیر رسیدن به نتیجه را توضیح می‌دهند، ولی برای ارائه نهایی باید اعداد و ادعاهای فایل گزارش نهایی استفاده شوند.

## نسخه اول‌شخص برای ارائه به استاد

اگر بخواهم کل مسیر کار را از ابتدا و با جزئیات توضیح بدهم، من این پروژه را با هدف پیاده‌سازی و ارزیابی یک سیستم تحلیل احساس ضمنی روی داده‌های SemEval 2014 شروع کردم. مسیرم از یک baseline خیلی ساده شروع شد، بعد به سمت یک pipeline چندمرحله‌ای شبیه THOR رفت، بعد reflection ساده را امتحان کردم، بعد reflection آگاه از نوع خطا و controller را اضافه کردم، و در نهایت به سمت policy selection و نسخه‌های original-ish و self-consistency حرکت کردم. در طول این مسیر سعی کردم هر مرحله هم از نظر مهندسی کد منظم باشد و هم از نظر تجربی با فایل نتیجه و متریک قابل ارجاع باشد.

این سند هم روایت پژوهشی کار را پوشش می‌دهد، هم تاریخچه توسعه کد را، هم فایل‌هایی را که در هر مرحله تغییر کرده‌اند، و هم وضعیت نهایی پروژه را تا تاریخ `2026-06-25`.

## 1. هدف پروژه و سوالی که دنبال کردم

عنوان رسمی پروژه در README این‌طور ثبت شده است:

`Development of a System for Implicit Sentiment Analysis Using Multi-Step Chain-of-Thought Reasoning`

مسئله اصلی من این بود که برای **Implicit Sentiment Analysis (ISA)**، فقط یک پیش‌بینی تک‌مرحله‌ای نداشته باشم، بلکه بررسی کنم آیا شکستن فرایند تصمیم‌گیری به چند گام reasoning، و بعد اضافه کردن reflection و controller، می‌تواند کیفیت پیش‌بینی را بهتر کند یا نه.

به‌صورت عملی، مسیر تحقیق من این بود:

1. داده‌های SCAPT-labeled SemEval14 را parse و تمیز کنم.
2. یک baseline مستقیم بسازم.
3. یک pipeline چندمرحله‌ای شبیه THOR بسازم.
4. خطاهای direct و THOR را تحلیل کنم.
5. reflection ساده را امتحان کنم.
6. یک reflection ساختاریافته آگاه از نوع خطا بسازم.
7. روی خروجی reflection یک controller تصمیم‌گیر تعریف کنم.
8. policyهای مختلف controller را با هم مقایسه کنم.
9. نسخه original-ish از promptهای THOR و self-consistency را اضافه کنم.
10. در آخر به‌جای rule ثابت، policy انتخابیِ یادگرفته‌شده از train را امتحان کنم.
11. pipeline نهایی را validate کنم و جدول نهایی بسازم.
12. چند تست کوچک برای منطق controller، parsing، policy و گزارش نهایی اضافه کنم.
13. نمونه‌های کیفی استخراج کنم تا مشخص شود سیستم نهایی دقیقاً کجا نسبت به Direct بهتر یا بدتر شده است.

## 2. وضعیت داده‌ها و آماده‌سازی آن‌ها

پایه کل پروژه روی داده‌های SCAPT-labeled SemEval14 بنا شده است. من parser را طوری نوشتم که از XMLهای اصلی، اطلاعات aspect-level را بیرون بکشد و بعد چند نسخه پردازش‌شده برای استفاده در آزمایش‌ها بسازد.

### فایل هسته‌ای این بخش

- `src/data_loader.py`

### ستون‌هایی که من از XML استخراج کردم

- `id`
- `source_sentence_id`
- `domain`
- `split`
- `sentence`
- `target`
- `from`
- `to`
- `polarity`
- `is_implicit`
- `opinion_words`
- `source_file`

### منطق پاک‌سازی

من بعد از parse اولیه، دو سطح داده ساختم:

- نسخه کامل: همه نمونه‌های aspect-level
- نسخه ISA-only: فقط نمونه‌هایی که `is_implicit == 1` دارند

بعد برای ساخت نسخه clean، این فیلترها را اعمال کردم:

- فقط polarityهای `positive`, `negative`, `neutral` نگه داشته شدند.
- فقط ردیف‌هایی که `is_implicit` آن‌ها معتبر بود نگه داشته شدند.
- ستون `is_implicit` به `int` تبدیل شد.

### خروجی‌های پردازش داده

- `data/processed/laptop_train.csv`
- `data/processed/laptop_test.csv`
- `data/processed/restaurant_train.csv`
- `data/processed/restaurant_test.csv`
- `data/processed/semeval14_scapt_all.csv`
- `data/processed/semeval14_scapt_all_clean.csv`
- `data/processed/semeval14_scapt_isa_only.csv`
- `data/processed/semeval14_scapt_isa_only_clean.csv`

### آمار نهایی داده clean

| مجموعه | تعداد ردیف | توضیح |
| --- | ---: | --- |
| `semeval14_scapt_all_clean.csv` | 7673 | کل داده تمیزشده |
| `semeval14_scapt_isa_only_clean.csv` | 2188 | فقط نمونه‌های implicit sentiment |

### توزیع ISA-only clean بر اساس دامنه و split

| domain | split | count |
| --- | --- | ---: |
| laptop | train | 716 |
| laptop | test | 175 |
| restaurant | train | 1030 |
| restaurant | test | 267 |

### توزیع ISA-only clean بر اساس polarity

| polarity | count |
| --- | ---: |
| neutral | 972 |
| negative | 619 |
| positive | 597 |

### نکته مهندسی مهم

من `data/raw/` را در `.gitignore` نگه داشتم، ولی خروجی‌های پردازش‌شده و فایل‌های نتیجه را version کردم. دلیلش این بود که:

- داده خام حجیم و بیرون از repo است.
- آزمایش‌ها باید روی فایل‌های پردازش‌شده ثابت و reproducible اجرا شوند.
- نتایج و متریک‌ها باید برای مقایسه commit به commit داخل repo بمانند.

## 3. خط زمانی کامل توسعه پروژه

## 3.1. ساخت اسکلت اولیه پروژه

- تاریخ: `2026-03-28`
- commit: `510c814`
- عنوان commit: `Initialize project structure`

در این مرحله من هنوز وارد پیاده‌سازی نشده بودم و اول ساختار پروژه را تعریف کردم تا مسیر توسعه منظم باشد.

### چیزهایی که ساختم

- پوشه‌های `data/raw`, `data/processed`, `data/outputs`, `logs`, `results`, `thesis_notes`
- اسکریپت‌های اولیه در `experiments/`
- فایل‌های prompt در `prompts/`
- فایل‌های هسته‌ای در `src/`

### فایل‌های اصلی ایجادشده در این مرحله

- `experiments/run_direct.py`
- `experiments/run_thor.py`
- `experiments/run_simple_reflection.py`
- `experiments/run_error_type_controller.py`
- `prompts/direct_prompt.txt`
- `prompts/error_type_reflection.txt`
- `prompts/simple_reflection.txt`
- `prompts/thor_aspect.txt`
- `prompts/thor_opinion.txt`
- `prompts/thor_polarity.txt`
- `src/controller.py`
- `src/data_loader.py`
- `src/evaluator.py`
- `src/prompt_runner.py`
- `src/reflection_pipeline.py`
- `src/thor_pipeline.py`
- `src/utils.py`

### توضیحی که من می‌توانم به استاد بدهم

در شروع کار، تمرکزم روی architecture پروژه بود، نه نتیجه. چون می‌خواستم از همان ابتدا مسیرهای direct baseline، THOR، reflection و controller از هم جدا باشند و بعداً اضافه‌کردن آزمایش‌ها باعث آشفتگی کد نشود.

## 3.2. parse داده‌های SCAPT و ساخت datasetهای قابل استفاده

- تاریخ: `2026-03-30`
- commit: `5b78117`
- عنوان commit: `Add SCAPT parser and processed SemEval14 ISA datasets`

این commit اولین مرحله‌ای بود که پروژه را از حالت skeleton خارج کرد. من parser XML را نوشتم، خروجی CSV ساختم، و داده‌های clean و ISA-only را تولید کردم.

### فایل‌های اصلی تغییرکرده

- `.gitignore`
- `src/data_loader.py`
- `data/processed/laptop_test.csv`
- `data/processed/laptop_train.csv`
- `data/processed/restaurant_test.csv`
- `data/processed/restaurant_train.csv`
- `data/processed/semeval14_scapt_all.csv`
- `data/processed/semeval14_scapt_all_clean.csv`
- `data/processed/semeval14_scapt_isa_only.csv`
- `data/processed/semeval14_scapt_isa_only_clean.csv`

### کاری که دقیقاً انجام دادم

- XMLهای laptop و restaurant را برای train و test خواندم.
- aspect termها را sentence به sentence استخراج کردم.
- `implicit_sentiment` را به مقدار باینری نرمال کردم.
- مجموعه clean را ساختم تا conflict و برچسب‌های نامعتبر وارد آزمایش نشوند.
- زیرمجموعه ISA-only را جدا کردم تا مسئله دقیقاً روی احساس ضمنی متمرکز شود.

### اهمیت این مرحله

اگر این مرحله درست انجام نمی‌شد، بقیه آزمایش‌ها قابل اعتماد نبودند. در واقع foundation کل پروژه همین فایل `src/data_loader.py` است.

## 3.3. نوشتن README و تثبیت هدف پروژه

- تاریخ: `2026-03-30`
- commit: `24bc435`
- عنوان commit: `Create README.md`

در این مرحله README را اضافه کردم تا هدف پروژه، ساختار پوشه‌ها، وضعیت فعلی و baselineهای برنامه‌ریزی‌شده روشن شوند.

### فایل تغییرکرده

- `README.md`

### نکته مهم

در README صریحاً ثبت کردم که عنوان رسمی thesis نباید تغییر کند. این برای ثابت نگه داشتن framing پژوهش مهم بود.

## 3.4. ساخت direct prompt و آماده‌سازی اجرای محلی

- تاریخ: `2026-03-31`
- commit: `200ab34`
- عنوان commit: `Add direct prompt and local runtime requirements`

قبل از هر reasoning پیچیده، من یک baseline مستقیم لازم داشتم. برای همین هم prompt baseline را نوشتم، هم dependencyهای پایه را مشخص کردم.

### فایل‌های تغییرکرده

- `prompts/direct_prompt.txt`
- `requirements.txt`

### منطق direct baseline

در این baseline، مدل فقط sentence و target را می‌بیند و باید مستقیماً یکی از سه label زیر را بدهد:

- `positive`
- `negative`
- `neutral`

### dependencyهای پایه

- `pandas`
- `scikit-learn`
- `tqdm`
- `requests`

## 3.5. اجرای direct baseline با Qwen3 8B

- تاریخ: `2026-04-03`
- commit: `13d45da`
- عنوان commit: `Run direct ISA baseline with qwen3 8b`

این اولین baseline کامل پروژه بود. من اسکریپت اجرایی را کامل کردم، abstraction اجرای prompt را اضافه کردم، ارزیابی را پیاده کردم، و اولین فایل‌های predictions و metrics را تولید کردم.

### فایل‌های اصلی تغییرکرده

- `.gitignore`
- `experiments/run_direct.py`
- `results/direct_isa_metrics.txt`
- `results/direct_isa_predictions.csv`
- `src/evaluator.py`
- `src/prompt_runner.py`
- `src/utils.py`

### چیزهایی که در کد اضافه کردم

- `PromptRunner` برای اجرای prompt
- `normalize_label` برای اینکه خروجی‌های آزاد مدل به سه label استاندارد map شوند
- `evaluate_predictions` برای محاسبه `accuracy` و `macro_f1`
- ذخیره هم‌زمان predictions و metrics

### نتیجه عددی direct baseline

| روش | n | accuracy | macro-F1 |
| --- | ---: | ---: | ---: |
| Direct baseline | 2188 | 0.657221 | 0.660807 |

### برداشت من از این مرحله

این baseline برای من خیلی مهم بود، چون یک خط مبنا داد که بعداً بتوانم هر معماری reasoning را با آن مقایسه کنم. نکته مهم‌تر این بود که direct baseline از همان ابتدا baseline ضعیفی نبود.

## 3.6. ساخت pipeline چندمرحله‌ای THOR

- تاریخ: `2026-04-15`
- commit: `f80d6f8`
- عنوان commit: `Add THOR ISA pipeline and baseline results`

در این مرحله من از پیش‌بینی تک‌مرحله‌ای به reasoning چندمرحله‌ای رفتم. pipeline را به چهار بخش شکستم:

1. تشخیص aspect
2. استخراج opinion clue
3. ساخت polarity reasoning
4. تبدیل reasoning به label نهایی

### فایل‌های اصلی تغییرکرده

- `experiments/run_direct.py`
- `experiments/run_thor.py`
- `prompts/thor_aspect.txt`
- `prompts/thor_opinion.txt`
- `prompts/thor_polarity.txt`
- `prompts/thor_polarity_label.txt`
- `results/thor_freeform_qwen8b_pilot_metrics.txt`
- `results/thor_freeform_qwen8b_pilot_predictions.csv`
- `results/thor_isa_metrics.txt`
- `results/thor_isa_predictions.csv`
- `src/__init__.py`
- `src/prompt_runner.py`
- `src/thor_pipeline.py`
- `test_runner.py`

### ویژگی‌های مهندسی THOR که من اضافه کردم

- کلاس `THORPipeline`
- تابع `clean_short_text` برای کوتاه و قابل‌کنترل نگه داشتن aspect
- fallback برای aspectهای نامعتبر به `general`
- fallback برای opinion خالی به `no clear opinion`
- نرمال‌سازی label نهایی
- قابلیت `RESUME` و `SAVE_EVERY` در اسکریپت اجرا برای jobهای طولانی

### promptهای THOR چه می‌کردند

- `thor_aspect.txt`: aspect implied را با یک عبارت کوتاه استخراج می‌کرد.
- `thor_opinion.txt`: clue یا opinion phrase را بیرون می‌کشید.
- `thor_polarity.txt`: از روی aspect و opinion reasoning می‌ساخت.
- `thor_polarity_label.txt`: reasoning را به یکی از سه label تبدیل می‌کرد.

### پایلوت اولیه THOR

یک خروجی pilot هم داشتم:

| روش | n | accuracy | macro-F1 |
| --- | ---: | ---: | ---: |
| THOR freeform pilot | 20 | 0.550000 | 0.533333 |

### نتیجه full-run THOR

| روش | n | accuracy | macro-F1 |
| --- | ---: | ---: | ---: |
| THOR simplified | 2188 | 0.585923 | 0.588127 |

### برداشت من از این مرحله

از نظر پژوهشی، THOR برای من مهم بود چون reasoning را explict می‌کرد. اما از نظر عملکرد خام، از direct baseline ضعیف‌تر شد. این برای ادامه مسیر خیلی تعیین‌کننده بود، چون نشان داد decomposition لزوماً به بهبود منجر نمی‌شود و ممکن است propagation error بسازد.

## 3.7. تحلیل خطای Direct در مقابل THOR

- تاریخ: `2026-04-15`
- commit: `bd0729d`
- عنوان commit: `Add Direct vs THOR error analysis`

بعد از دیدن اینکه THOR از direct ضعیف‌تر است، من صرفاً به metric نهایی اکتفا نکردم و یک analysis جدا نوشتم تا ببینم اختلاف دقیقاً کجاست.

### فایل‌های اصلی تغییرکرده

- `experiments/analyze_baseline_errors.py`
- `results/baseline_error_analysis_metrics.txt`
- `results/baseline_error_analysis_predictions.csv`
- `results/baseline_error_examples_predictions.csv`

### کار این اسکریپت

- خروجی direct و THOR را روی کلیدهای مشترک merge می‌کند.
- درستی هر کدام را نسبت به gold برچسب می‌زند.
- چهار گروه می‌سازد:
  - `both_correct`
  - `direct_correct_thor_wrong`
  - `thor_correct_direct_wrong`
  - `both_wrong`
- confusion matrix می‌سازد.
- چند مثال منتخب برای بررسی دستی ذخیره می‌کند.

### یافته‌های اصلی

| شاخص | مقدار |
| --- | ---: |
| هر دو درست | 976 |
| direct درست، THOR غلط | 462 |
| THOR درست، direct غلط | 306 |
| هر دو غلط | 444 |
| میزان توافق دو مدل | 1349 از 2188 |
| میزان عدم‌توافق | 839 از 2188 |

### برداشت من

این تحلیل نشان داد direct baseline فقط به‌طور اتفاقی بهتر نشده بود؛ واقعاً در تعداد زیادی نمونه، THOR باعث افت شده بود. بنابراین لازم بود به فکر مکانیسم اصلاح خطا باشم، نه صرفاً prompt بیشتر.

## 3.8. اضافه کردن Simple Reflection

- تاریخ: `2026-04-15`
- commit: `864d8cc`
- عنوان commit: `Add simple reflection baseline pilot`

در این مرحله من یک reflection ساده ساختم که روی خروجی THOR یک بازبینی دیگر انجام دهد و label نهایی را اصلاح کند.

### فایل‌های اصلی تغییرکرده

- `experiments/run_simple_reflection.py`
- `prompts/simple_reflection.txt`
- `results/simple_reflection_isa_metrics.txt`
- `results/simple_reflection_isa_predictions.csv`
- `src/reflection_pipeline.py`

### منطق reflection ساده

مدل دوباره این موارد را می‌بیند:

- sentence
- target
- aspect
- opinion
- polarity reasoning
- label اولیه THOR

و باید فقط یکی از سه label را برگرداند.

### دلیل انجام این مرحله

می‌خواستم ببینم آیا فقط با یک pass اضافه می‌شود بخشی از خطاهای THOR را بدون اضافه کردن controller پیچیده‌تر اصلاح کرد یا نه.

## 3.9. اجرای full-run و تحلیل خطای Simple Reflection

- تاریخ: `2026-04-16`
- commit: `8bd7f9a`
- عنوان commit: `Add simple reflection full-run results and analysis`

بعد از pilot، reflection را روی کل داده اجرا کردم و analysis جداگانه نوشتم.

### فایل‌های اصلی تغییرکرده

- `experiments/analyze_reflection_errors.py`
- `results/reflection_error_analysis_metrics.txt`
- `results/reflection_error_analysis_predictions.csv`
- `results/reflection_error_examples_predictions.csv`
- `results/simple_reflection_isa_metrics.txt`
- `results/simple_reflection_isa_predictions.csv`

### نتیجه عددی

| روش | n | accuracy | macro-F1 |
| --- | ---: | ---: | ---: |
| Simple reflection | 2188 | 0.600548 | 0.605101 |

### نسبت به THOR چه اتفاقی افتاد

- THOR: `macro_f1 = 0.588127`
- Simple reflection: `macro_f1 = 0.605101`

یعنی reflection ساده از THOR بهتر شد، ولی هنوز از direct baseline پایین‌تر بود.

### تحلیل رفتاری reflection

| شاخص | مقدار |
| --- | ---: |
| تعداد مواردی که reflection خروجی THOR را تغییر داد | 74 |
| unchanged_correct | 1265 |
| unchanged_wrong | 849 |
| fixed_by_reflection | 49 |
| broken_by_reflection | 17 |
| changed_still_wrong | 8 |

### برداشت من

reflection ساده مفید بود، ولی خیلی محافظه‌کار عمل می‌کرد. فقط در 74 نمونه از 2188 نمونه نظر THOR را عوض کرد. این یعنی برای بهبود بیشتر، من به reflection ساختاریافته‌تر نیاز داشتم.

## 3.10. ساخت Error-Type Aware Reflection و Controller

- تاریخ: `2026-04-16`
- commit: `aec5fad`
- عنوان commit: `Add backbone-aware ETC-ISA controller pilot`

این یکی از مهم‌ترین مراحل پروژه بود. در اینجا من reflection را از یک بازبینی ساده به یک تشخیص ساختاریافته تبدیل کردم.

### فایل‌های اصلی تغییرکرده

- `experiments/analyze_baseline_errors.py`
- `experiments/analyze_reflection_errors.py`
- `experiments/run_direct.py`
- `experiments/run_error_type_controller.py`
- `experiments/run_simple_reflection.py`
- `experiments/run_thor.py`
- `prompts/error_type_reflection.txt`
- `requirements-flan.txt`
- `results/etc_isa_metrics.txt`
- `results/etc_isa_predictions.csv`
- `src/controller.py`
- `src/experiment_config.py`
- `src/prompt_runner.py`
- `src/reflection_pipeline.py`

### دو تغییر معماری خیلی مهم در این مرحله

#### الف) generic شدن runner

در `src/prompt_runner.py` من فقط به Ollama محدود نماندم و backend مبتنی بر Hugging Face seq2seq را هم اضافه کردم:

- `OllamaPromptRunner`
- `HFSeq2SeqPromptRunner`
- wrapper نهایی `PromptRunner`

این باعث شد backend از خود منطق آزمایش جدا شود.

#### ب) ساخت controller واقعی

در `src/controller.py` من:

- مجموعه error typeهای correctable را تعریف کردم.
- confidence را نرمال کردم.
- تابع `select_final_label` را نوشتم.

### error typeهایی که controller می‌شناسد

- `missed_implicit_negative`
- `missed_implicit_positive`
- `neutral_overinterpretation`
- `target_scope_shift`
- `aspect_opinion_mismatch`
- `reasoning_label_inconsistency`
- `no_error`
- `insufficient_evidence`

### منطق کلی ETC

در `experiments/run_error_type_controller.py` من این جریان را ساختم:

1. خروجی direct و THOR را merge می‌کنم.
2. اگر policy بگوید diagnostic لازم است، reflection ساختاریافته را اجرا می‌کنم.
3. reflection سه خط می‌دهد:
   - `error_type=...`
   - `label=...`
   - `confidence=...`
4. controller با توجه به:
   - direct label
   - THOR label
   - label پیشنهادی reflection
   - error type
   - confidence
   - fallback policy
   تصمیم نهایی را می‌گیرد.

### نتیجه full-run ETC

| روش | n | accuracy | macro-F1 |
| --- | ---: | ---: | ---: |
| ETC-ISA | 2188 | 0.650366 | 0.653890 |

### برداشت من

ETC از THOR و simple reflection بهتر شد، ولی هنوز کمی پایین‌تر از direct baseline بود. با این حال فاصله خیلی کم شد و این نشان داد diagnostic + controller مسیر امیدبخشی است.

## 3.11. تحلیل خطای ETC

- تاریخ: `2026-04-17`
- commit: `22600f6`
- عنوان commit: `Add ETC-ISA full-run results and analysis`

بعد از اجرای ETC، من برای این مدل هم تحلیل خطا نوشتم تا مشخص شود controller در عمل چه‌قدر trigger شده و کجا سود یا ضرر ایجاد کرده است.

### فایل‌های اصلی تغییرکرده

- `experiments/analyze_etc_errors.py`
- `results/etc_error_analysis_metrics.txt`
- `results/etc_error_analysis_predictions.csv`
- `results/etc_error_examples_predictions.csv`
- `results/etc_isa_metrics.txt`
- `results/etc_isa_predictions.csv`

### یافته‌های کلیدی

| شاخص | مقدار |
| --- | ---: |
| diagnostic trigger | 839 |
| not triggered | 1349 |
| `accept_missed_implicit_negative` | 142 |
| `accept_missed_implicit_positive` | 13 |
| `fallback_use_direct` | 684 |
| `agreement_keep_shared_label` | 1349 |

### تغییرات ETC نسبت به THOR

| گروه | مقدار |
| --- | ---: |
| unchanged_correct | 1002 |
| unchanged_wrong | 426 |
| fixed_by_etc | 421 |
| broken_by_etc | 280 |
| changed_still_wrong | 59 |

### برداشت من

این تحلیل یک نکته خیلی مهم را نشان داد: ETC تعداد قابل‌توجهی خطا را اصلاح می‌کرد، ولی در عین حال در بعضی نقاط over-correction هم داشت. بنابراین واضح شد که خود controller هم نیاز به calibration دارد.

## 3.12. ablation روی policyهای controller و ساخت selected controller

- تاریخ: `2026-04-17`
- commit: `8262bb7`
- عنوان commit: `Add ETC-ISA policy ablation and selected controller`

در این مرحله من به‌جای اینکه فقط یک policy را به‌صورت شهودی نگه دارم، یک ablation سیستماتیک ساختم.

### فایل‌های اصلی تغییرکرده

- `experiments/ablate_etc_policies.py`
- `experiments/apply_etc_policy.py`
- `experiments/run_error_type_controller.py`
- `results/etc_policy_ablation_metrics.txt`
- `results/etc_policy_ablation_predictions.csv`
- `results/etc_selected_isa_metrics.txt`
- `results/etc_selected_isa_predictions.csv`
- `src/controller.py`

### کاری که `ablate_etc_policies.py` انجام می‌دهد

چند policy مختلف را روی همان خروجی ETC اعمال می‌کند و برای هر کدام:

- accuracy
- macro-F1
- تغییر نسبت به THOR
- تغییر نسبت به direct

را می‌سنجد.

### policyهای مهمی که مقایسه شدند

- `direct_baseline`
- `thor_baseline`
- `etc_direct_medium_no_trust`
- `etc_direct_high_no_trust`
- `etc_direct_high_positive_only`
- `etc_direct_high_negative_only`
- `etc_direct_medium_trust_no_error`
- `etc_thor_medium_no_trust`
- `etc_thor_high_no_trust`

### نتیجه مهم ablation در نسخه commit‌شده

بهترین policy rule-based روی خروجی ETC استاندارد این بود:

| policy | accuracy | macro-F1 |
| --- | ---: | ---: |
| `etc_direct_high_positive_only` | 0.659506 | 0.663444 |

این policy از direct baseline هم کمی بهتر شد:

- direct baseline: `macro_f1 = 0.660807`
- selected manual policy: `macro_f1 = 0.663444`

### برداشت من

این مرحله نشان داد اگر controller را خیلی محافظه‌کار و انتخابی طراحی کنم، می‌توانم از direct baseline هم کمی عبور کنم. این یکی از نقاط مهم دفاع پروژه است، چون نشان می‌دهد reflection آزاد به‌تنهایی کافی نیست، ولی reflection ساختاریافته + policy مناسب می‌تواند سود بدهد.

## 3.13. ساخت promptهای THOR original-ish و self-consistency

- تاریخ: `2026-04-24`
- commit: `d0f549c`
- عنوان commit: `Add THOR original-ish prompts and self-consistency runs`

در این مرحله من یک branch فکری جدید را وارد پروژه کردم: نزدیک‌تر کردن promptها به فرم original-ish و اضافه کردن self-consistency.

### فایل‌های اصلی تغییرکرده

- `experiments/run_thor_originalish.py`
- `experiments/run_thor_self_consistency.py`
- `prompts/thor_originalish_aspect.txt`
- `prompts/thor_originalish_opinion.txt`
- `prompts/thor_originalish_polarity.txt`
- `prompts/thor_originalish_polarity_label.txt`
- `results/thor_originalish_isa_metrics.txt`
- `results/thor_originalish_isa_predictions.csv`
- `results/thor_originalish_sc3_isa_metrics.txt`
- `results/thor_originalish_sc3_isa_predictions.csv`
- `src/thor_pipeline.py`

### تفاوت conceptually مهم با THOR قبلی

در نسخه original-ish:

- promptها بازتر و توضیحی‌تر شدند.
- سقف tokenها بیشتر شد.
- aspect و opinion اجازه داشتند عبارت‌های طبیعی‌تری تولید کنند.
- reasoning اجازه داشت کمی بسط پیدا کند.

### نتیجه pilot نسخه deterministic original-ish

| روش | n | accuracy | macro-F1 |
| --- | ---: | ---: | ---: |
| THOR original-ish deterministic pilot | 20 | 0.750000 | 0.650794 |

این فقط pilot بود و full-run deterministic در repo commit‌شده وجود ندارد.

### نتیجه full-run self-consistency با 3 نمونه

| روش | n | accuracy | macro-F1 |
| --- | ---: | ---: | ---: |
| THOR original-ish SC3 | 2188 | 0.612431 | 0.615445 |

### منطق self-consistency

در `run_thor_self_consistency.py` من:

- برای هر نمونه `SC_N` بار pipeline را اجرا می‌کنم.
- labelهای خروجی را جمع می‌کنم.
- با majority vote تصمیم می‌گیرم.
- اگر tie باشد، اولین label tied را انتخاب می‌کنم.
- یک run representative را هم برای ذخیره aspect/opinion/reasoning نگه می‌دارم.

### برداشت من

self-consistency نسخه original-ish نسبت به THOR simplified بهتر شد، ولی هنوز از direct baseline پایین‌تر ماند. با این حال این مرحله یک پایه جدید برای controllerهای بعدی ساخت.

## 4. پیوست تاریخی: تغییرات محلی تا تاریخ 2026-06-06

این بخش برای ثبت مسیر تاریخی پروژه نگه داشته شده است. برای دفاع و گزارش نهایی، فایل `thesis_notes/final_project_report_fa.md` مرجع اصلی است.

## 4.1. بازطراحی `apply_etc_policy.py` از policy دستی به policy یادگرفته‌شده

مهم‌ترین تغییر local فعلی در فایل زیر است:

- `experiments/apply_etc_policy.py`

### قبل از این تغییر

نسخه commit‌شده فقط یک policy rule-based را اعمال می‌کرد:

- fallback مستقیم یا THOR
- threshold روی confidence
- مجموعه مشخصی از accepted error typeها

### بعد از این تغییر

من دو mode مجزا اضافه کردم:

- `manual`
- `train_calibrated`

### ایده mode جدید `train_calibrated`

در این mode، من به‌جای این‌که از قبل بگویم همیشه direct یا diagnostic را ترجیح بدهیم، روی split train یاد می‌گیرم که برای هر الگوی خطا، بهترین source prediction کدام است:

- `direct`
- `thor`
- `diagnostic`

### کلیدهای policy که فعلاً برای یادگیری استفاده کرده‌ام

- `direct_prediction`
- `error_type`
- `diagnostic_confidence`
- `domain`

### چرا این تغییر مهم است

این تغییر از نظر علمی مهم است چون selected controller را از یک rule دستی به یک selection mechanism داده‌محور تبدیل می‌کند.

## 4.2. نتیجه local روی ETC استاندارد

فایل‌های local اصلاح‌شده:

- `results/etc_selected_isa_metrics.txt`
- `results/etc_selected_isa_predictions.csv`
- `results/etc_selected_manual_check_metrics.txt`
- `results/etc_selected_manual_check_predictions.csv`

### مقایسه manual و train-calibrated روی ETC استاندارد

| نسخه | accuracy | macro-F1 |
| --- | ---: | ---: |
| direct baseline | 0.657221 | 0.660807 |
| selected manual policy | 0.659506 | 0.663444 |
| selected train-calibrated policy | 0.665905 | 0.669348 |

### تفسیر

یعنی policy یادگرفته‌شده:

- از direct baseline بهتر شده است.
- از selected policy دستی هم بهتر شده است.
- با فقط `152` تغییر نسبت به direct، `78` gain و `59` loss ایجاد کرده است.

### توزیع source انتخاب‌شده در نسخه train-calibrated استاندارد

- `direct`: 1811
- `diagnostic`: 352
- `thor`: 25

برداشت من از این الگو این است که در نسخه استاندارد، direct همچنان ستون اصلی پیش‌بینی است، ولی در بعضی profileهای خاص diagnostic واقعاً ارزش افزوده دارد.

## 4.3. اجرای ETC روی خروجی THOR original-ish SC3

فایل‌های local جدید:

- `results/etc_thor_originalish_sc3_isa_metrics.txt`
- `results/etc_thor_originalish_sc3_isa_predictions.csv`
- `results/etc_thor_originalish_sc3_pilot20_isa_metrics.txt`
- `results/etc_thor_originalish_sc3_pilot20_isa_predictions.csv`

### نتیجه full-run ETC روی THOR original-ish SC3

| روش | accuracy | macro-F1 |
| --- | ---: | ---: |
| THOR original-ish SC3 | 0.612431 | 0.615445 |
| ETC روی THOR original-ish SC3 | 0.643510 | 0.645832 |

### برداشت

اینجا controller روی backbone reasoning جدید واقعاً سود داده و THOR original-ish SC3 را به‌طور واضح بالا کشیده است.

## 4.4. اجرای selected train-calibrated روی ETC حاصل از THOR original-ish SC3

فایل‌های local جدید:

- `results/etc_thor_originalish_sc3_selected_isa_metrics.txt`
- `results/etc_thor_originalish_sc3_selected_isa_predictions.csv`

### بهترین نتیجه فعلی کل پروژه

| روش | accuracy | macro-F1 |
| --- | ---: | ---: |
| Selected train-calibrated روی ETC حاصل از THOR original-ish SC3 | 0.680987 | 0.683829 |

### این نتیجه نسبت به direct baseline چه‌قدر بهتر است

- بهبود accuracy: `+0.023766`
- بهبود macro-F1: `+0.023022`

### تغییرات رفتاری نسبت به direct روی کل داده overall

- `changed_vs_direct = 150`
- `gain_vs_direct = 90`
- `loss_vs_direct = 38`

نکته: برای گزارش نهایی و دفاع، مقایسه اصلی باید روی split تست گزارش شود. مقدار test-only در فایل نهایی جدید آمده است: `gain_vs_direct = 26` و `loss_vs_direct = 6`.

### توزیع source انتخاب‌شده در این نسخه

- `direct`: 1566
- `thor`: 621
- `diagnostic`: 1

### برداشت من

این الگو خیلی جالب است. بر خلاف نسخه ETC استاندارد، در اینجا policy یادگرفته‌شده بیشتر به direct و thor تکیه می‌کند و تقریباً diagnostic را کنار می‌گذارد. یعنی diagnostic بیشتر نقش signal برای انتخاب source دارد، نه این‌که خودش همیشه label نهایی را تعیین کند.

## 4.5. ablation به‌روزشده روی policyها برای نسخه original-ish SC3

فایل‌های local اصلاح‌شده:

- `results/etc_policy_ablation_metrics.txt`
- `results/etc_policy_ablation_predictions.csv`

در ablation جدید، من policyهای rule-based را روی خروجی `etc_thor_originalish_sc3` هم مقایسه کرده‌ام.

### نکته مهم

در این نسخه، بهترین policy rule-based دیگر لزوماً از direct baseline عبور نکرده است. این دقیقاً همان جایی است که policy یادگرفته‌شده ارزش خود را نشان می‌دهد، چون:

- ruleهای ثابت روی همه profileها خوب عمل نمی‌کنند.
- train-calibrated selection از همه policyهای ثابت بهتر شده است.

## 4.6. وضعیت فایل‌های تغییرکرده در working tree فعلی

### فایل‌های modified

- `experiments/apply_etc_policy.py`
- `results/etc_policy_ablation_metrics.txt`
- `results/etc_policy_ablation_predictions.csv`
- `results/etc_selected_isa_metrics.txt`
- `results/etc_selected_isa_predictions.csv`

### فایل‌های untracked

- `results/etc_selected_manual_check_metrics.txt`
- `results/etc_selected_manual_check_predictions.csv`
- `results/etc_thor_originalish_sc3_isa_metrics.txt`
- `results/etc_thor_originalish_sc3_isa_predictions.csv`
- `results/etc_thor_originalish_sc3_pilot20_isa_metrics.txt`
- `results/etc_thor_originalish_sc3_pilot20_isa_predictions.csv`
- `results/etc_thor_originalish_sc3_selected_isa_metrics.txt`
- `results/etc_thor_originalish_sc3_selected_isa_predictions.csv`

## 5. معماری نهایی کد و نقش هر فایل

این بخش را اگر استاد از من بپرسد که "الان کد نهایی‌ات از چه اجزایی تشکیل شده؟" می‌توانم این‌طور توضیح بدهم.

### هسته `src/`

#### `src/data_loader.py`

مسئول parse داده خام XML و ساخت همه نسخه‌های processed dataset است.

#### `src/prompt_runner.py`

abstraction اجرای مدل است. در حال حاضر دو backend دارد:

- `ollama`
- `hf`

#### `src/utils.py`

- بارگذاری prompt از فایل
- نرمال‌سازی label مدل به `positive/negative/neutral`

#### `src/evaluator.py`

- محاسبه `accuracy`
- محاسبه `macro_f1`
- ساخت `classification_report`

#### `src/thor_pipeline.py`

pipeline چندمرحله‌ای THOR را اجرا می‌کند:

- aspect
- opinion
- polarity reasoning
- final label

#### `src/reflection_pipeline.py`

دو مسیر reflection را پوشش می‌دهد:

- `SimpleReflectionPipeline`
- `ErrorTypeReflectionPipeline`

و خروجی diagnostic ساختاریافته را parse می‌کند.

#### `src/controller.py`

منطق تصمیم‌گیری نهایی controller را نگه می‌دارد:

- تعریف error typeها
- تعریف confidence levelها
- تصمیم نهایی با `select_final_label`

#### `src/experiment_config.py`

لایه تنظیمات اجرایی برای experimentها است:

- `DEBUG_N`
- `EXPERIMENT_ID`
- `PROMPT_BACKEND`
- `result_path`
- `describe_runtime`

این فایل باعث شده result naming و runtime configuration در اسکریپت‌ها تکراری نشود.

### پوشه `experiments/`

#### اسکریپت‌های run

- `run_direct.py`
- `run_thor.py`
- `run_simple_reflection.py`
- `run_error_type_controller.py`
- `run_thor_originalish.py`
- `run_thor_self_consistency.py`

#### اسکریپت‌های analysis

- `analyze_baseline_errors.py`
- `analyze_reflection_errors.py`
- `analyze_etc_errors.py`

#### اسکریپت‌های policy

- `ablate_etc_policies.py`
- `apply_etc_policy.py`

### پوشه `prompts/`

در این پوشه من promptها را از کد جدا نگه داشتم تا:

- experimentation سریع‌تر شود
- تغییر prompt نیازمند دست‌زدن به منطق pipeline نباشد
- نسخه simplified و original-ish از هم جدا بمانند

## 6. جدول جمع‌بندی نتایج

## 6.1. نتایج commit‌شده

| روش | وضعیت | n | accuracy | macro-F1 | توضیح |
| --- | --- | ---: | ---: | ---: | --- |
| Direct baseline | commit‌شده | 2188 | 0.657221 | 0.660807 | baseline مستقیم |
| THOR freeform pilot | commit‌شده | 20 | 0.550000 | 0.533333 | پایلوت اولیه |
| THOR simplified | commit‌شده | 2188 | 0.585923 | 0.588127 | pipeline چندمرحله‌ای |
| Simple reflection | commit‌شده | 2188 | 0.600548 | 0.605101 | بازبینی ساده روی THOR |
| ETC-ISA | commit‌شده | 2188 | 0.650366 | 0.653890 | reflection ساختاریافته + controller |
| Selected manual policy | commit‌شده | 2188 | 0.659506 | 0.663444 | بهترین policy rule-based روی ETC استاندارد |
| THOR original-ish deterministic | commit‌شده | 20 | 0.750000 | 0.650794 | فقط pilot |
| THOR original-ish SC3 | commit‌شده | 2188 | 0.612431 | 0.615445 | self-consistency full-run |

## 6.2. نتایج local فعلی

| روش | وضعیت | n | accuracy | macro-F1 | توضیح |
| --- | --- | ---: | ---: | ---: | --- |
| Selected train-calibrated روی ETC استاندارد | local | 2188 | 0.665905 | 0.669348 | انتخاب source بر اساس train |
| ETC روی THOR original-ish SC3 | local | 2188 | 0.643510 | 0.645832 | controller روی THOR original-ish |
| Selected train-calibrated روی ETC حاصل از THOR original-ish SC3 | local | 2188 | 0.680987 | 0.683829 | بهترین نتیجه فعلی کل پروژه |

## 7. برداشت علمی من از کل مسیر

اگر بخواهم جمع‌بندی علمی پروژه را از زبان خودم بگویم، به این شکل توضیح می‌دهم:

اول، direct baseline برخلاف انتظار اولیه خیلی قوی بود. این نکته مهمی بود چون نشان داد هر نوع reasoning چندمرحله‌ای لزوماً از یک پیش‌بینی مستقیم بهتر نیست.

دوم، THOR به من interpretability و trace reasoning داد، ولی در نسخه simplified باعث افت عملکرد شد. بنابراین صرفاً decomposing the task کافی نبود و error propagation ایجاد می‌کرد.

سوم، simple reflection نشان داد یک pass اضافی می‌تواند بخشی از خطاها را اصلاح کند، اما چون reflection تغییرات کمی اعمال می‌کرد، سقف بهبودش محدود بود.

چهارم، ETC-ISA به‌عنوان reflection آگاه از نوع خطا ایده بهتری بود، چون به‌جای خروجی آزاد، model را مجبور می‌کرد نوع خطا، label پیشنهادی و confidence را ساختاریافته بدهد. این ساختار برای controller خیلی مهم بود.

پنجم، analysisها نشان دادند که controller اگر بیش از حد تهاجمی باشد، over-correction می‌کند. به همین دلیل، policy design و بعداً policy selection به بخش اصلی پروژه تبدیل شد.

ششم، بهترین نتیجه وقتی به دست آمد که من از policy ثابت فاصله گرفتم و selection را از روی train یاد گرفتم. یعنی به‌جای این‌که بگویم همیشه direct یا همیشه diagnostic بهتر است، اجازه دادم سیستم برای هر profile از خطا تصمیم بگیرد کدام source prediction قابل‌اعتمادتر است.

## 8. اگر بخواهم contribution پروژه را در چند جمله کوتاه بگویم

اگر استاد از من بخواهد contribution اصلی را خیلی خلاصه بگویم، می‌توانم این‌طور بگویم:

من پروژه را از یک baseline مستقیم شروع کردم، بعد یک pipeline reasoning چندمرحله‌ای شبیه THOR ساختم، بعد نشان دادم که THOR ساده به‌تنهایی از baseline مستقیم بهتر نیست. بعد با تحلیل خطا به این نتیجه رسیدم که باید reflection را ساختاریافته کنم. برای همین Error-Type Aware Reflection و controller را طراحی کردم. بعد روی policyهای controller ablation انجام دادم و در نهایت selected controller یادگرفته‌شده از train را ساختم که در بهترین نسخه فعلی، macro-F1 را به `0.683829` رسانده است و از direct baseline بهتر عمل می‌کند.

## 9. نکاتی که در جلسه دفاع باید حواسم به آن‌ها باشد

### اگر پرسیدند چرا direct از THOR بهتر شد

می‌گویم چون decomposition باعث شد خطاهای مرحله‌ای ایجاد شود؛ مثلاً aspect اشتباه، opinion نامرتبط، یا reasoning-label inconsistency. پس interpretability بالا رفت ولی performance خام افت کرد.

### اگر پرسیدند چرا reflection ساده کافی نبود

می‌گویم چون فقط در `74` نمونه از `2188` نمونه خروجی THOR را تغییر داد. یعنی از نظر رفتاری خیلی محدود بود و برای اصلاح systematic خطاها کافی نبود.

### اگر پرسیدند controller چرا لازم بود

می‌گویم چون reflection ساختاریافته بدون منطق تصمیم‌گیری نهایی هنوز می‌تواند over-correct کند. controller باعث شد diagnostic فقط وقتی و فقط به شکلی که policy اجازه می‌دهد وارد تصمیم نهایی شود.

### اگر پرسیدند بهترین ایده فعلی پروژه چیست

می‌گویم بهترین ایده فعلی این است که به‌جای اعتماد مطلق به یک source، بین `direct`, `thor`, و `diagnostic` به‌صورت condition-based انتخاب کنیم. این دقیقاً همان چیزی است که در نسخه train-calibrated selected policy انجام داده‌ام.

## 10. فهرست زمانی commitها برای ارجاع سریع

| تاریخ | commit | خلاصه |
| --- | --- | --- |
| 2026-03-28 | `510c814` | ساخت اسکلت اولیه پروژه |
| 2026-03-30 | `5b78117` | parse داده‌های SCAPT و ساخت datasetهای processed |
| 2026-03-30 | `24bc435` | ساخت README |
| 2026-03-31 | `200ab34` | direct prompt و requirementهای اجرا |
| 2026-04-03 | `13d45da` | direct baseline با Qwen3 8B |
| 2026-04-15 | `f80d6f8` | THOR pipeline و baseline نتایج |
| 2026-04-15 | `bd0729d` | تحلیل خطای direct در برابر THOR |
| 2026-04-15 | `864d8cc` | simple reflection pilot |
| 2026-04-16 | `8bd7f9a` | simple reflection full-run و analysis |
| 2026-04-16 | `aec5fad` | ETC-ISA pilot و controller |
| 2026-04-17 | `22600f6` | ETC full-run و analysis |
| 2026-04-17 | `8262bb7` | policy ablation و selected controller |
| 2026-04-24 | `d0f549c` | THOR original-ish و self-consistency |

## 11. جمع‌بندی نهایی

اگر بخواهم خیلی دقیق و صادقانه جمع‌بندی کنم، مسیر این پروژه برای من این‌طور بوده است:

من از یک baseline مستقیم شروع کردم، بعد reasoning چندمرحله‌ای را اضافه کردم، بعد متوجه شدم که reasoning خام به‌تنهایی کافی نیست، بعد reflection و analysis را وارد کردم، بعد controller و policy design را ساختاربندی کردم، و در نهایت به یک selected policy داده‌محور رسیدم که فعلاً بهترین عملکرد پروژه را داده است.

بنابراین خروجی نهایی من فقط یک اسکریپت یا یک عدد نیست؛ بلکه یک مسیر کامل پژوهشی است که در آن:

- داده را خودم parse و تمیز کرده‌ام،
- baseline ساخته‌ام،
- pipeline reasoning پیاده کرده‌ام،
- خطاها را تحلیل کرده‌ام،
- reflection و controller طراحی کرده‌ام،
- policyها را ablate کرده‌ام،
- و یک نسخه بهترِ train-calibrated برای selection ساخته‌ام.

اگر بخواهم یک جمله آخر برای ارائه بگویم:

**دستاورد اصلی من این بوده که نشان بدهم در ISA، reasoning چندمرحله‌ای وقتی مفید می‌شود که با تحلیل خطا، controller، و policy selection دقیق ترکیب شود؛ وگرنه خود reasoning به‌تنهایی حتی می‌تواند از baseline مستقیم ضعیف‌تر باشد.**
