# جمع‌بندی نهایی پروژه

## سیستم نهایی

سیستم نهایی پروژه یک مدل fine-tune شده نیست؛ وزن‌های Qwen3 8B تغییر نکرده‌اند. روش فعلی به صورت zero-shot / prompt-based اجرا شده و خروجی چند مسیر مختلف را با یک سیاست انتخاب نهایی ترکیب می‌کند:

```text
Direct Qwen3 8B
+ THOR original-ish self-consistency (SC3)
+ Error-Type-Aware Reflection
+ Controller
+ Train-Calibrated Source Selection
```

## نتیجه اصلی

روی split تست، بهترین خروجی فعلی مربوط به `Final selected pipeline` است:

| روش | Accuracy تست | Macro-F1 تست |
| --- | ---: | ---: |
| Direct Qwen3 8B | 0.678733 | 0.674075 |
| Final selected pipeline | 0.723982 | 0.719204 |

این یعنی pipeline نهایی نسبت به baseline مستقیم، هم در Accuracy و هم در Macro-F1 بهتر شده است.

## تفسیر فنی

در این پروژه، Direct baseline یک مرجع عددی قوی فراهم کرد و THOR-style reasoning یک مسیر reasoning قابل تحلیل ایجاد کرد. بهترین عملکرد زمانی به دست آمد که reasoning مرحله‌ای در کنار Direct، diagnostic reflection و انتخاب منبع استفاده شد.

بنابراین سیستم نهایی صرفاً اجرای پشت سر هم چند prompt نیست. مرحله آخر، یعنی `Train-Calibrated Source Selection`، از split train یاد می‌گیرد در چه وضعیت‌هایی کدام منبع قابل اعتمادتر است: `direct`، `thor` یا `diagnostic`.

## نسبت با THOR اصلی

THOR الهام اصلی بخش reasoning پروژه است، اما پیاده‌سازی فعلی با Qwen3 8B و Ollama انجام شده و با setup رسمی THOR، که مسیرهای Flan-T5/fine-tuning هم دارد، یکی نیست. دلیل پیاده‌سازی مستقل این بود که باید THOR-style reasoning با داده، backend، reflection، controller، policy selection، validation و تحلیل کیفی همین پروژه یکپارچه می‌شد.

جمله کوتاه برای توضیح:

> THOR در پروژه ما یک منبع reasoning است، نه کل سیستم نهایی.

## شاهد کیفی

در split تست، مقایسه سیستم نهایی با Direct این وضعیت را نشان می‌دهد:

| گروه | تعداد |
| --- | ---: |
| مواردی که Direct نادرست بود و سیستم نهایی درست پیش‌بینی کرد | 26 |
| مواردی که Direct درست بود و سیستم نهایی نادرست پیش‌بینی کرد | 6 |
| هر دو درست | 294 |
| هر دو غلط | 116 |

این برای گزارش ارزشمند است، چون نشان می‌دهد بهبود نهایی فقط یک عدد کلی نیست؛ در تعدادی از نمونه‌های test، سیستم نهایی تصمیم Direct را اصلاح کرده است. نمونه‌های دقیق در فایل `thesis_notes/final_qualitative_examples_fa.md` و نسخه CSV در `results/final_qualitative_examples.csv` آمده‌اند.

## محدودیت‌ها

- fine-tuning انجام نشده است.
- backend مربوط به HuggingFace/Flan-T5 در کد وجود دارد، اما در نتایج فعلی تست یا ارزیابی نشده است.
- نتیجه اصلی full-test با Qwen/Ollama است. Gemini 2.5 Flash فقط روی subset متوازن 240تایی برای مقایسه مدل بزرگ‌تر اجرا شده است.
- اسکریپت‌های نهایی `run_final_pipeline.py` و `extract_qualitative_examples.py` مدل را دوباره اجرا نمی‌کنند؛ فقط خروجی‌های ذخیره‌شده را اعتبارسنجی، خلاصه و تحلیل می‌کنند.

## مقایسه محدود با Gemini

برای بررسی اینکه آیا مدل بزرگ‌تر می‌تواند به مسیر THOR/profile کمک کند، یک subset متوازن ساخته شد:

```text
train: 150 = 25 نمونه برای هر ترکیب domain/polarity
test: 90 = 15 نمونه برای هر ترکیب domain/polarity
```

این subset از split رسمی train/test جدا نشده؛ یعنی نمونه‌های train از train اصلی و نمونه‌های test از test اصلی آمده‌اند. نتیجه مهم این بود که Gemini direct روی test قوی‌تر از Qwen direct شد، اما THOR Gemini روی test از direct پایین‌تر بود. بهترین نسخه Gemini با validation-tuned profile به test macro-F1 برابر `0.808340` رسید، کمی بالاتر از Gemini direct با `0.804886`.

نتیجه دفاعی: Gemini نشان می‌دهد مدل قوی‌تر به baseline کمک می‌کند، اما contribution اصلی پروژه همچنان در انتخاب منبع و کنترل خطای reasoning است، نه صرفاً عوض کردن مدل.

## نکته اجرایی

برای پروژه کارشناسی، بخش پیاده‌سازی فقط یک prompt ساده نیست. پروژه شامل آماده‌سازی داده، baseline مستقیم، reasoning چندمرحله‌ای، reflection، diagnostic error type، controller، self-consistency، سیاست انتخاب کالیبره‌شده با train، ارزیابی عددی، تست‌های کوچک، و تحلیل کیفی است.

متن کامل‌تر و آماده‌تر برای گزارش در `thesis_notes/final_project_report_fa.md` قرار دارد.
