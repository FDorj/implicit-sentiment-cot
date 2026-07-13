# مسیر نهایی پروژه ISA

این سند نسخه دقیق pipeline نهایی پروژه را توضیح می‌دهد. نکته مهم این است که مدل زبانی fine-tune نشده و آزمایش‌های معتبر فعلی با Qwen3 8B از طریق Ollama انجام شده‌اند. کد HuggingFace/Flan-T5 در runner وجود دارد، اما تست و ارزیابی نشده و نباید جزو نتایج پروژه ادعا شود.

## روش نهایی

روش نهایی پروژه یک ترکیب چندمنبعی است. علامت `+` در اینجا یعنی این مؤلفه‌ها در تصمیم نهایی استفاده می‌شوند. هر مؤلفه یک سیگنال یا مرحله تحلیلی ایجاد می‌کند و تصمیم نهایی با source selection گرفته می‌شود.

```text
Direct Qwen3 8B
+ THOR original-ish self-consistency (SC3)
+ Error-Type-Aware Reflection
+ Controller
+ Train-Calibrated Source Selection
```

نمای دقیق‌تر جریان تصمیم:

```text
Direct prediction
        \
         -> Train-Calibrated Source Selection -> Final prediction
        /
THOR original-ish SC3 prediction
        \
         -> Error-Type-Aware Reflection -> Controller / diagnostic signal
```

این pipeline به شکل چند مرحله و چند فایل خروجی اجرا شده است، نه یک اجرای مستقیم end-to-end که دوباره مدل را صدا بزند. اسکریپت `experiments/run_final_pipeline.py` مدل را دوباره اجرا نمی‌کند؛ خروجی‌های موجود را اعتبارسنجی می‌کند و جدول نهایی را می‌سازد.

## ترتیب فایل‌ها

```text
results/direct_isa_predictions.csv
        +
results/thor_originalish_sc3_isa_predictions.csv
        |
        v
results/etc_thor_originalish_sc3_isa_predictions.csv
        |
        v
results/etc_thor_originalish_sc3_selected_isa_predictions.csv
```

مرحله ETC فقط وقتی diagnostic reflection را اجرا کرده که Direct و THOR original-ish SC3 اختلاف داشته‌اند. بنابراین diagnostic برای همه نمونه‌ها اجرا نشده؛ فقط برای disagreementها فعال شده است.

## نقش هر مرحله

`Direct Qwen3 8B`

مدل فقط جمله و target را می‌بیند و مستقیم یکی از سه label را تولید می‌کند.

`THOR original-ish SC3`

مدل با promptهای نزدیک‌تر به THOR، سه بار مسیر reasoning را اجرا می‌کند و با majority vote label نهایی THOR را می‌دهد. خروجی این مرحله شامل aspect، opinion، reasoning، labelهای چند run و vote count است.

`Error-Type-Aware Reflection`

وقتی Direct و THOR اختلاف دارند، مدل trace مربوط به THOR را بررسی می‌کند و خروجی ساختاریافته می‌دهد:

```text
error_type=...
label=...
confidence=...
```

`Controller`

یک منطق rule-based است که با استفاده از direct label، THOR label، diagnostic label، error type و confidence تصمیم اولیه می‌گیرد.

`Train-Calibrated Source Selection`

این مرحله از split train یاد می‌گیرد که در هر profile بهتر است به کدام source اعتماد شود:

```text
direct
thor
diagnostic
```

کلیدهای profile فعلی این‌ها هستند:

```text
direct_prediction
error_type
diagnostic_confidence
domain
```

این مرحله fine-tuning نیست، چون وزن‌های مدل زبانی تغییر نمی‌کنند. فقط یک جدول تصمیم‌گیری از روی train ساخته می‌شود و روی test بدون استفاده از gold label اعمال می‌شود.

## نتایج نهایی

جدول کامل در این فایل‌ها تولید می‌شود:

- `results/final_results_table.csv`
- `results/final_results_table.md`

نتیجه اصلی برای گزارش باید test-only باشد، چون policy نهایی از split train برای انتخاب source استفاده می‌کند:

| روش | test accuracy | test macro-F1 |
| --- | ---: | ---: |
| Direct Qwen3 8B | 0.678733 | 0.674075 |
| THOR simplified | 0.599548 | 0.578437 |
| Simple reflection | 0.615385 | 0.606732 |
| ETC standard | 0.660633 | 0.659430 |
| THOR original-ish SC3 | 0.590498 | 0.600690 |
| ETC over original-ish SC3 | 0.660633 | 0.662108 |
| Final selected pipeline | 0.723982 | 0.719204 |

تفسیر نتیجه: `THOR original-ish SC3` در این setting به‌تنهایی بهترین عدد را نمی‌دهد، اما یک source reasoning و diagnostic trace فراهم می‌کند. بهبود نهایی از ترکیب چند source و انتخاب داده‌محور بین Direct، THOR و diagnostic به دست آمده است.

## تحلیل کیفی نسبت به Direct

روی split تست:

| گروه | تعداد |
| --- | ---: |
| Direct نادرست، سیستم نهایی درست | 26 |
| Direct درست، سیستم نهایی نادرست | 6 |
| هر دو درست | 294 |
| هر دو غلط | 116 |

نمونه‌ها در این فایل‌ها آمده‌اند:

- `results/final_qualitative_examples.csv`
- `thesis_notes/final_qualitative_examples_fa.md`

## اعتبارسنجی زنجیره

`experiments/run_final_pipeline.py` این موارد را بررسی می‌کند:

- همه فایل‌های اصلی 2188 ردیف دارند.
- keyهای نهایی duplicate ندارند.
- Direct و THOR با ETC کامل هم‌تراز هستند.
- مقدار `direct_prediction` داخل ETC با خروجی Direct برابر است.
- مقدار `thor_prediction` داخل ETC با خروجی THOR original-ish SC3 برابر است.
- همه `selected_prediction`ها label معتبر دارند.

خروجی validation در این فایل ذخیره می‌شود:

```text
results/final_pipeline_validation.txt
```

## به‌روزرسانی مدل Gemini روی subset متوازن

بعد از نتیجه اصلی Qwen، یک مقایسه محدود با `Gemini 2.5 Flash` هم انجام شد. این مقایسه روی کل test رسمی اجرا نشده، چون اجرای THOR self-consistency با Gemini زمان و هزینه زیادی داشت. برای همین یک subset متوازن ساخته شد:

```text
data/processed/gemini_model_comparison_subset_train150_test90.csv
```

ساختار subset:

| بخش | laptop negative | laptop neutral | laptop positive | restaurant negative | restaurant neutral | restaurant positive | کل |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 25 | 25 | 25 | 25 | 25 | 25 | 150 |
| test | 15 | 15 | 15 | 15 | 15 | 15 | 90 |

یعنی split رسمی train/test جابه‌جا نشده است؛ فقط از هر ترکیب `split/domain/polarity` تعداد ثابتی نمونه با seed ثابت `20260709` انتخاب شده است. بنابراین نمونه‌های test در ساخت profile دیده نشده‌اند.

نتایج مهم Gemini روی این subset:

| روش | overall macro-F1 | test macro-F1 |
| --- | ---: | ---: |
| Gemini direct | 0.741729 | 0.804886 |
| Gemini THOR original-ish SC3 | 0.775404 | 0.716109 |
| Gemini ETC controller | 0.745482 | 0.804886 |
| Gemini validation-tuned selected profile | 0.818858 | 0.808340 |

در نسخه validation-tuned، policy نهایی `rich_unguarded` انتخاب شد و از 240 نمونه، `211` بار direct و `29` بار THOR را انتخاب کرد. `diagnostic` به‌عنوان source نهایی انتخاب نشد. علت این است که اصلاح parser و prompt باعث شد خروجی diagnostic ساختاریافته‌تر شود، اما کیفیت labelهای diagnostic هنوز برای اعتماد مستقیم کافی نبود.

## چیزهایی که نباید ادعا شوند

- fine-tuning انجام نشده است.
- Flan-T5/HuggingFace تست و ارزیابی نشده است.
- نتیجه اصلی full-test پروژه با Qwen/Ollama است. Gemini فقط به‌عنوان مقایسه subset متوازن گزارش می‌شود.
- `run_final_pipeline.py` مدل را دوباره اجرا نمی‌کند؛ فقط خروجی‌های موجود را validate و summarize می‌کند.

## جمله پیشنهادی برای گزارش یا ارائه

در این پروژه یک سامانه آزمایشی برای تحلیل احساس ضمنی پیاده‌سازی شده است که ابتدا direct و THOR را به عنوان دو منبع پیش‌بینی تولید می‌کند، سپس در موارد اختلاف از reflection آگاه از نوع خطا و controller استفاده می‌کند، و در نهایت با یک source-selection policy یادگرفته‌شده از train تصمیم می‌گیرد در هر profile به direct، THOR یا diagnostic اعتماد کند. بهترین نسخه فعلی بدون fine-tuning مدل زبانی، روی test به macro-F1 برابر با 0.719204 رسیده است.
