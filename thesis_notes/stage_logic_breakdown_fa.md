# منطق هر مرحله از پروژه ISA

این سند برای این نوشته شده که اگر بخواهم **لاجیک دقیق هر مرحله** را توضیح بدهم، فقط نتیجه‌ها را نگویم و روشن کنم:

- ورودی هر مرحله چه بوده
- روی ورودی چه پردازشی انجام شده
- تصمیم نهایی در آن مرحله چطور گرفته شده
- خروجی هر مرحله چه بوده
- ضعف یا ریسک اصلی همان مرحله چه بوده

## چند اصطلاح پایه

`target`
- همان aspect term داخل دیتاست است.
- مثال: در جمله `Going to bring it to service today.`، اگر target برابر `service` باشد، ما sentiment را فقط نسبت به `service` می‌سنجیم.

`aspect`
- چیزی است که THOR از روی sentence و target استنتاج می‌کند.
- این با target یکی نیست.
- مثال: target ممکن است `cord` باشد، ولی aspect استنتاج‌شده `battery life` باشد.

`direct_prediction`
- برچسبی که baseline مستقیم می‌دهد.

`thor_prediction`
- برچسبی که pipeline چندمرحله‌ای THOR می‌دهد.

`diagnostic_label`
- برچسب پیشنهادی reflection ساختاریافته.

`controller_prediction`
- برچسب نهایی بعد از اعمال controller روی خروجی reflection ساختاریافته.

## مرحله 1: parse داده خام و ساخت datasetهای clean

### فایل‌های اصلی

- `src/data_loader.py`

### ورودی

- فایل‌های XML مربوط به SCAPT-labeled SemEval14 برای دو دامنه:
  - laptop
  - restaurant
- هر دو split:
  - train
  - test

### لاجیک

1. XML sentence به sentence خوانده می‌شود.
2. اگر sentence اصلاً `aspectTerms` نداشته باشد، کنار گذاشته می‌شود.
3. برای هر `aspectTerm` یک ردیف ساخته می‌شود.
4. از هر `aspectTerm` این فیلدها استخراج می‌شوند:
   - `term -> target`
   - `polarity`
   - `from`, `to`
   - `implicit_sentiment`
   - `opinion_words`
5. ردیف‌ها برای چهار فایل domain/split ساخته می‌شوند.
6. همه ردیف‌ها با هم concat می‌شوند.
7. نسخه ISA-only با شرط `is_implicit == 1` ساخته می‌شود.
8. نسخه clean با این فیلترها ساخته می‌شود:
   - polarity فقط یکی از `positive`, `negative`, `neutral` باشد
   - `is_implicit` معتبر باشد

### خروجی

- `semeval14_scapt_all.csv`
- `semeval14_scapt_all_clean.csv`
- `semeval14_scapt_isa_only.csv`
- `semeval14_scapt_isa_only_clean.csv`

### ریسک این مرحله

- اگر parse اشتباه باشد، کل pipeline بعدی روی داده غلط می‌چرخد.
- اگر targetها درست map نشده باشند، evaluation بی‌معنا می‌شود.

### مثال از دیتا

- Sentence: `Purchased a Toshiba Lap top it worked good until just after the warrenty went out.`
- Target: `warrenty`
- Gold polarity: `negative`
- این نمونه بعد از parse به یک ردیف aspect-level تبدیل می‌شود و بعد وارد dataset `ISA-only clean` می‌شود.

## مرحله 2: direct baseline

### فایل‌های اصلی

- `experiments/run_direct.py`
- `prompts/direct_prompt.txt`
- `src/prompt_runner.py`
- `src/utils.py`

### ورودی

- `data/processed/semeval14_scapt_isa_only_clean.csv`

### لاجیک

1. هر ردیف dataset جداگانه خوانده می‌شود.
2. فقط `sentence` و `target` داخل prompt قرار می‌گیرند.
3. مدل باید مستقیماً یکی از سه label زیر را بدهد:
   - `positive`
   - `negative`
   - `neutral`
4. خروجی خام مدل با `normalize_label` نرمال می‌شود.
5. prediction ذخیره می‌شود.
6. در پایان روی کل داده `accuracy` و `macro-F1` محاسبه می‌شود.

### تصمیم نهایی چطور گرفته می‌شود

- هیچ reasoning چندمرحله‌ای وجود ندارد.
- هیچ reflection یا controller وجود ندارد.
- prediction نهایی همان خروجی نرمال‌شده مدل است.

### مزیت

- ساده، سریع، و خطای مرحله‌ای ندارد.

### ضعف

- هیچ trace تفسیری به ما نمی‌دهد.
- اگر مدل implicit cue را نگیرد، هیچ مکانیزم اصلاحی ندارد.

### مثال از دیتا

- Sentence: `Going to bring it to service today.`
- Target: `service`
- Gold: `neutral`
- Direct prediction: `positive`
- این مثال نشان می‌دهد direct baseline گاهی از خود جمله بیش‌برداشت می‌کند.

## مرحله 3: THOR simplified

### فایل‌های اصلی

- `experiments/run_thor.py`
- `src/thor_pipeline.py`
- `prompts/thor_aspect.txt`
- `prompts/thor_opinion.txt`
- `prompts/thor_polarity.txt`
- `prompts/thor_polarity_label.txt`

### ورودی

- همان dataset ISA-only clean

### لاجیک

THOR هر نمونه را در 4 گام پردازش می‌کند:

1. `infer_aspect`
   - از روی sentence و target، یک aspect phrase کوتاه استنتاج می‌شود.
   - اگر خروجی خالی یا نامعتبر باشد، `general` گذاشته می‌شود.

2. `infer_opinion`
   - از روی sentence و target و aspect، opinion clue ساخته می‌شود.
   - اگر خالی باشد، `no clear opinion` گذاشته می‌شود.

3. `infer_polarity_reasoning`
   - از روی sentence و target و aspect و opinion، یک reasoning متنی کوتاه ساخته می‌شود.

4. `infer_polarity_label`
   - reasoning متنی به یک label نهایی تبدیل می‌شود.

### تصمیم نهایی چطور گرفته می‌شود

- prediction نهایی همان label مرحله 4 است.

### مزیت

- trace reasoning قابل‌دیدن می‌دهد:
  - aspect
  - opinion
  - polarity_reasoning

### ضعف

- خطای هر مرحله می‌تواند به مرحله بعد propagate شود.
- اگر aspect اشتباه شود، opinion و reasoning هم ممکن است اشتباه شوند.

### مثال از دیتا

- Sentence: `I charge it at night and skip taking the cord with me because of the good battery life.`
- Target: `cord`
- Gold: `neutral`
- THOR aspect: `battery life`
- THOR opinion: `good battery life`
- THOR prediction: `positive`
- اینجا THOR از target فاصله گرفته و sentiment مربوط به `battery life` را به `cord` نسبت داده است.

## مرحله 4: مقایسه خطای Direct و THOR

### فایل‌های اصلی

- `experiments/analyze_baseline_errors.py`

### ورودی

- خروجی direct baseline
- خروجی THOR

### لاجیک

1. دو فایل prediction روی کلیدهای مشترک merge می‌شوند.
2. برای هر ردیف بررسی می‌شود:
   - direct درست است یا نه
   - THOR درست است یا نه
3. هر نمونه داخل یکی از چهار گروه می‌افتد:
   - `both_correct`
   - `direct_correct_thor_wrong`
   - `thor_correct_direct_wrong`
   - `both_wrong`

### هدف

- بفهمیم THOR در چه مواردی به تصمیم بهتر کمک کرده و در چه مواردی فقط trace reasoning تولید کرده است.
- بفهمیم disagreementها کجا هستند.

### نکته منطقی مهم

- این مرحله خودش prediction جدید نمی‌سازد.
- فقط diagnosis تحلیلی می‌دهد برای تصمیم‌گیری مرحله بعد.

### مثال از دیتا

- Sentence: `The tech guy then said ... I have to direct my concern to the "sales" team ...`
- Target: `"sales" team`
- Gold: `negative`
- Direct prediction: `negative`
- THOR prediction: `neutral`
- این نمونه در analysis داخل گروه `direct_correct_thor_wrong` می‌افتد.

## مرحله 5: simple reflection

### فایل‌های اصلی

- `experiments/run_simple_reflection.py`
- `src/reflection_pipeline.py`
- `prompts/simple_reflection.txt`

### ورودی

- خروجی THOR:
  - sentence
  - target
  - aspect
  - opinion
  - polarity_reasoning
  - thor label

### لاجیک

1. مدل دوباره trace کامل THOR را می‌بیند.
2. از مدل فقط خواسته می‌شود یک label نهایی جدید بدهد.
3. خروجی نرمال می‌شود.

### تصمیم نهایی چطور گرفته می‌شود

- prediction نهایی همان `reflection_prediction` است.

### مزیت

- یک فرصت دوم برای اصلاح THOR می‌دهد.

### ضعف

- نوع خطا را مشخص نمی‌کند.
- confidence نمی‌دهد.
- controller ندارد.

### مثال از دیتا

- Sentence: `HOW DOES THE POWER SUPPLY NOT WORK!!!`
- Target: `POWER SUPPLY`
- Gold: `negative`
- THOR prediction: `neutral`
- Reflection prediction: `negative`
- این نمونه نشان می‌دهد simple reflection گاهی فقط با یک بازبینی ساده می‌تواند THOR را اصلاح کند.

## مرحله 6: تحلیل خطای simple reflection

### فایل‌های اصلی

- `experiments/analyze_reflection_errors.py`

### لاجیک

1. خروجی direct، THOR، و reflection با هم merge می‌شوند.
2. بررسی می‌شود reflection آیا THOR را تغییر داده یا نه.
3. نمونه‌ها به گروه‌هایی مثل این تقسیم می‌شوند:
   - `fixed_by_reflection`
   - `broken_by_reflection`
   - `unchanged_wrong`

### هدف

- بفهمیم reflection واقعاً کجا مفید بوده و کجا نه.

### مثال از دیتا

- Sentence: `The materials that came with the computer did not include the right # anywhere.`
- Target: `materials`
- Gold: `negative`
- THOR prediction: `neutral`
- Reflection prediction: `negative`
- این نمونه در گروه `fixed_by_reflection` قرار می‌گیرد.

## مرحله 7: reflection ساختاریافته یا Error-Type Reflection

### فایل‌های اصلی

- `src/reflection_pipeline.py`
- `prompts/error_type_reflection.txt`

### ورودی

- sentence
- target
- direct label
- THOR aspect
- THOR opinion
- THOR reasoning
- THOR label

### لاجیک

این بار از مدل فقط label نهایی نمی‌خواهیم. از او می‌خواهیم:

- نوع خطا را تشخیص دهد
- label پیشنهادی بدهد
- confidence بدهد

خروجی باید دقیقاً این سه خط باشد:

- `error_type=...`
- `label=...`
- `confidence=...`

بعد این خروجی parse و normalize می‌شود.

### مزیت

- خروجی از حالت آزاد به خروجی ساختاریافته تبدیل می‌شود.

### ضعف

- confidence در این پروژه عددی محاسبه نمی‌شود؛ مدل خودش low/medium/high می‌گوید.

### مثال از دیتا

- Sentence: `Purchased a Toshiba Lap top it worked good until just after the warrenty went out.`
- Target: `warrenty`
- THOR label: `neutral`
- Structured reflection output:
  - `error_type = missed_implicit_negative`
  - `label = negative`
  - `confidence = medium`

## مرحله 8: controller

### فایل‌های اصلی

- `src/controller.py`
- `experiments/run_error_type_controller.py`

### لاجیک

controller یک لایه تصمیم‌گیری rule-based است.

### ورودی controller

- `direct_label`
- `thor_label`
- `proposed_label`
- `error_type`
- `confidence`
- policy params

### قوانین اصلی

1. اگر direct و THOR با هم موافق باشند، همان label نگه داشته می‌شود.
2. اگر error type جزو correctableها باشد و confidence به threshold برسد، `proposed_label` قبول می‌شود.
3. اگر policy اجازه بدهد، در بعضی شرایط `no_error` باعث نگه داشتن THOR می‌شود.
4. وگرنه بر اساس fallback policy:
   - یا direct نگه داشته می‌شود
   - یا THOR

### خروجی

- `controller_prediction`
- `controller_decision`

### مزیت

- تصمیم نهایی متمرکز و قابل‌کنترل می‌شود.

### ضعف

- اگر policy خوب تنظیم نشود، over-correction رخ می‌دهد.

### مثال از دیتا

- Sentence: `I love WIndows 7 which is a vast improvment over Vista.`
- Target: `Vista`
- Gold: `negative`
- Direct label: `negative`
- THOR label: `positive`
- Diagnostic label: `negative`
- Confidence: `medium`
- Controller decision: `accept_missed_implicit_negative`
- Final prediction: `negative`

## مرحله 9: ETC-ISA

### فایل‌های اصلی

- `experiments/run_error_type_controller.py`

### لاجیک کامل ETC

1. direct و THOR merge می‌شوند.
2. اگر direct و THOR disagreement داشته باشند، diagnostic trigger می‌شود.
3. reflection ساختاریافته اجرا می‌شود.
4. controller با توجه به error type و confidence تصمیم نهایی را می‌گیرد.
5. prediction نهایی به‌صورت `controller_prediction` ذخیره می‌شود.

### ایده اصلی

- فقط روی disagreementها انرژی محاسباتی اضافی بگذار.
- هر disagreement را هم با یک diagnostic ساختاریافته بررسی کن.

### مثال از دیتا

- Sentence: `Purchased a Toshiba Lap top it worked good until just after the warrenty went out.`
- Target: `warrenty`
- Direct: `negative`
- THOR: `neutral`
- Diagnostic: `negative`
- Controller: `negative`
- در این نمونه disagreement باعث trigger شدن ETC شده و خروجی نهایی THOR را اصلاح کرده است.

## مرحله 10: تحلیل خطای ETC

### فایل‌های اصلی

- `experiments/analyze_etc_errors.py`

### لاجیک

1. direct، THOR و ETC merge می‌شوند.
2. بررسی می‌شود ETC نسبت به THOR:
   - کجا باعث اصلاح پیش‌بینی شده
   - کجا باعث افت تصمیم شده
   - کجا بدون تغییر مانده

### هدف

- تشخیص اینکه controller واقعاً سود داده یا فقط labelها را زیاد تغییر داده است.

### مثال از دیتا

- Sentence: `In the shop, these MacBooks are encased ... you will never know about the razor edge ...`
- Target: `edge`
- Gold: `negative`
- Direct: `negative`
- THOR: `positive`
- ETC: `negative`
- Error type: `missed_implicit_negative`
- این نمونه در analysis داخل `fixed_by_etc` قرار می‌گیرد.

## مرحله 11: policy ablation

### فایل‌های اصلی

- `experiments/ablate_etc_policies.py`

### لاجیک

در این مرحله خود ETC دوباره اجرا نمی‌شود.  
بلکه روی همان خروجی ETC، چند policy مختلفِ تصمیم‌گیری اعمال می‌شود.

### کارهای این اسکریپت

1. فایل ETC را می‌خواند.
2. برای هر policy، روی همه ردیف‌ها prediction جدید می‌سازد.
3. برای هر policy این‌ها را مقایسه می‌کند:
   - accuracy
   - macro-F1
   - تغییر نسبت به direct
   - تغییر نسبت به THOR

### هدف

- بفهمیم کدام policy بهتر است.
- بفهمیم حساسیت سیستم به تنظیمات controller چقدر است.

### مثال از دیتا

- Sentence: `although its windows vista compared to windows xp sucks.`
- Target: `windows xp`
- Gold: `positive`
- Direct: `neutral`
- THOR: `negative`
- Diagnostic: `positive`
- در فایل ablation دیده می‌شود که policyهای مختلف برای همین نمونه خروجی‌های متفاوت می‌دهند:
  - `direct_baseline -> neutral`
  - `thor_baseline -> negative`
  - `etc_direct_medium_no_trust -> positive`

## مرحله 12: selected manual policy

### لاجیک

بعد از ablation، بهترین policy rule-based را انتخاب می‌کنیم و آن را به‌عنوان selected manual policy نگه می‌داریم.

### نکته

- این هنوز policy ثابت است.
- هنوز چیزی از train یاد نمی‌گیرد.

### مثال از دیتا

- Sentence: `It's applications are terrific, including the replacements for Microsoft office.`
- Target: `Microsoft office`
- Gold: `positive`
- Direct: `negative`
- THOR: `positive`
- Diagnostic: `positive`
- Selected manual policy: `positive`
- Decision: `accept_missed_implicit_positive`

## مرحله 13: THOR original-ish

### فایل‌های اصلی

- `experiments/run_thor_originalish.py`
- `prompts/thor_originalish_*`

### لاجیک

- همان THOR چهارمرحله‌ای حفظ می‌شود
- ولی promptها original-ish می‌شوند
- token budget بزرگ‌تر می‌شود
- wording آزادتر و طبیعی‌تر می‌شود

### هدف

- ببینیم آیا فقط با تغییر prompt variant می‌توان quality reasoning را بهتر کرد یا نه.

### مثال از دیتا

- Sentence: `I charge it at night and skip taking the cord with me because of the good battery life.`
- Target: `cord`
- Gold: `neutral`
- Original-ish aspect: `battery life`
- Original-ish opinion: `No sentiment evidence about the cord itself`
- Original-ish prediction: `neutral`
- این نمونه نشان می‌دهد original-ish prompt گاهی target را محافظه‌کارانه‌تر می‌خواند.

## مرحله 14: self-consistency با 3 sample

### فایل‌های اصلی

- `experiments/run_thor_self_consistency.py`

### لاجیک

1. برای هر نمونه، THOR سه بار اجرا می‌شود.
2. هر بار یک label می‌دهد.
3. با majority vote برنده انتخاب می‌شود.
4. یک run representative هم برای ذخیره trace نگه داشته می‌شود.

### هدف

- به‌جای اتکا به یک مسیر reasoning، چند مسیر گرفته شود و رأی‌گیری شود.

### ضعف

- هزینه محاسباتی 3 برابر می‌شود.

### مثال از دیتا

- Sentence: `I charge it at night and skip taking the cord with me because of the good battery life.`
- Target: `cord`
- Gold: `neutral`
- SC labels: `negative, positive, negative`
- Vote counts: `negative:2 ; positive:1`
- Final SC3 prediction: `negative`
- این نمونه نشان می‌دهد self-consistency همیشه به معنی درست‌تر شدن نیست.

## مرحله 15: ETC روی THOR original-ish SC3

### لاجیک

این مرحله از نظر ساختاری مثل ETC قبلی است، فقط به‌جای THOR simplified از خروجی `THOR original-ish SC3` استفاده می‌کند.

### ایده

- backbone reasoning را عوض می‌کنیم
- بعد همان diagnostic + controller را روی آن سوار می‌کنیم

### مثال از دیتا

- Sentence: `Going to bring it to service today.`
- Target: `service`
- Gold: `neutral`
- Direct: `positive`
- THOR original-ish SC3: `neutral`
- Diagnostic: `neutral`
- ETC output: `positive`
- Controller decision: `fallback_use_direct`
- این نمونه نشان می‌دهد ETC روی backbone جدید هم در بعضی profileها ممکن است به fallback کم‌دقت‌تر برگردد.

## مرحله 16: selected train-calibrated policy

### فایل‌های اصلی

- `experiments/apply_etc_policy.py`

### ایده اصلی

به‌جای policy ثابت، از train یاد بگیریم در هر profile کدام source بهتر است:

- direct
- thor
- diagnostic

### کلیدهایی که policy بر اساس آن‌ها یاد گرفته می‌شود

- `direct_prediction`
- `error_type`
- `diagnostic_confidence`
- `domain`

### لاجیک

1. فقط ردیف‌های train جدا می‌شوند.
2. ردیف‌های train بر اساس keyهای بالا group می‌شوند.
3. در هر group بررسی می‌شود کدام source بیشترین prediction درست را دارد:
   - direct
   - thor
   - diagnostic
4. همان source به‌عنوان policy آن group ذخیره می‌شود.
5. بعد این policy یادگرفته‌شده روی همه داده‌ها اعمال می‌شود.

### تصمیم نهایی چطور گرفته می‌شود

- اگر policy group بگوید `thor` بهتر است، `thor_prediction` انتخاب می‌شود.
- اگر بگوید `direct` بهتر است، `direct_prediction` انتخاب می‌شود.
- اگر بگوید `diagnostic` بهتر است، `diagnostic_label` انتخاب می‌شود.

### تفاوت با manual policy

- manual policy = rule ثابت
- train-calibrated policy = انتخاب source بر اساس رفتار واقعی train

### مثال از دیتا

- Sentence: `Going to bring it to service today.`
- Target: `service`
- Gold: `neutral`
- Direct: `positive`
- THOR original-ish SC3: `neutral`
- ETC output: `positive`
- Selected source: `thor`
- Final selected prediction: `neutral`
- این نمونه دقیقاً نشان می‌دهد train-calibrated policy چطور از بین sourceها انتخاب می‌کند و یک خطای ETC را اصلاح می‌کند.

## جمع‌بندی منطق کل پروژه در یک خط

منطق کل پروژه این است:

1. اول baseline مستقیم ساخته شد.
2. بعد reasoning چندمرحله‌ای THOR اضافه شد.
3. بعد با تحلیل خطا مشخص شد THOR به‌تنهایی کافی نیست.
4. بعد reflection ساده امتحان شد.
5. بعد reflection ساختاریافته + controller ساخته شد.
6. بعد policyهای controller با ablation مقایسه شدند.
7. در آخر به‌جای policy ثابت، selected train-calibrated policy ساخته شد تا انتخاب بین direct و thor و diagnostic داده‌محور شود.
