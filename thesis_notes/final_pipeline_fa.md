# مسیر نهایی پروژه ISA

این سند نسخه دقیق و قابل دفاع pipeline نهایی پروژه را توضیح می‌دهد. نکته مهم این است که مدل زبانی fine-tune نشده و آزمایش‌های معتبر فعلی با Qwen3 8B از طریق Ollama انجام شده‌اند. کد HuggingFace/Flan-T5 در runner وجود دارد، اما تست و ارزیابی نشده و نباید جزو نتایج پروژه ادعا شود.

## روش نهایی

روش نهایی پروژه یک ترکیب چندمنبعی است. علامت `+` در اینجا یعنی این مؤلفه‌ها در تصمیم نهایی استفاده می‌شوند، نه اینکه هر مرحله همیشه و برای همه نمونه‌ها خروجی مرحله قبل را کورکورانه جایگزین کند.

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

نتیجه اصلی قابل دفاع باید test-only باشد:

| روش | test accuracy | test macro-F1 |
| --- | ---: | ---: |
| Direct Qwen3 8B | 0.678733 | 0.674075 |
| THOR simplified | 0.599548 | 0.578437 |
| Simple reflection | 0.615385 | 0.606732 |
| ETC standard | 0.660633 | 0.659430 |
| THOR original-ish SC3 | 0.590498 | 0.600690 |
| ETC over original-ish SC3 | 0.660633 | 0.662108 |
| Final selected pipeline | 0.723982 | 0.719204 |

نکته مهم برای دفاع: `THOR original-ish SC3` به‌تنهایی از Direct ضعیف‌تر است. بهبود نهایی از اینجا آمده که سیستم نهایی همیشه THOR را انتخاب نمی‌کند؛ بین Direct، THOR و diagnostic بر اساس policy یادگرفته‌شده از train انتخاب می‌کند.

## تحلیل کیفی نسبت به Direct

روی split تست:

| گروه | تعداد |
| --- | ---: |
| Direct غلط، سیستم نهایی درست | 26 |
| Direct درست، سیستم نهایی غلط | 6 |
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

## چیزهایی که نباید ادعا شوند

- fine-tuning انجام نشده است.
- Flan-T5/HuggingFace تست و ارزیابی نشده است.
- OpenAI API یا مدل API در نتایج فعلی استفاده نشده است.
- `run_final_pipeline.py` مدل را دوباره اجرا نمی‌کند؛ فقط خروجی‌های موجود را validate و summarize می‌کند.

## جمله پیشنهادی برای دفاع

در این پروژه یک سامانه آزمایشی برای تحلیل احساس ضمنی پیاده‌سازی شده است که ابتدا direct و THOR را به عنوان دو منبع پیش‌بینی تولید می‌کند، سپس در موارد اختلاف از reflection آگاه از نوع خطا و controller استفاده می‌کند، و در نهایت با یک source-selection policy یادگرفته‌شده از train تصمیم می‌گیرد در هر profile به direct، THOR یا diagnostic اعتماد کند. بهترین نسخه فعلی بدون fine-tuning مدل زبانی، روی test به macro-F1 برابر با 0.719204 رسیده است.
