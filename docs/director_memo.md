# Director memo: this month's retention budget

**All numbers are from a simulated subscriber base. Save rate, contact cost and horizon are labelled assumptions.**

## Recommendation
Contact about **1,470 subscribers (8.9% of the 16,503 active base)** at the 2024-12-31 cutoff, ranked by expected value rather than by probability alone. Stop there: beyond that, each extra contact costs more than it is expected to save.

## What it buys
- The top decile by risk holds subscribers with an average 7.7% chance of churning next month, against 1.5% for the base. Across the test months (Sep-Nov) contacting the top 10% reaches about **49% of the subscribers who then churned** (lift about 5.0x).
- Backtest, November cutoff: contacting the 1,363 subscribers the value ranking selects gave **38,382 SDG of net value** on the simulated outcomes. The model expected 46,147 SDG, so expected value overstates the realised figure by about 17% in this quiet month; it expected 54 churners in the list and 52 occurred. Use the realised figure, not the expected one, as the planning number until a pilot confirms it.
- The same budget spent on the highest probabilities alone realised only **16,458 SDG**. The two lists share 34% of names: the difference is that some high-risk subscribers are low value.

## Assumptions (in `churn_model/config.yaml`)
Save rate 20% of contacted would-be churners; cost 20 SDG per contact; value = 6 months of average spend. Prepaid spend from recharges, postpaid from ARPU. Change them in the dashboard sliders.

## Risks
1. The model is trained and tested on simulated data; real performance is unknown.
2. A regime shift (outage, price change) moves the base rate and therefore the calibration. The monthly drift check flags it; refit the calibration whenever the model is retrained.
3. Regional outages distort scores: Kassala is flagged at 52% and Nyala at 34% of subscribers, while their observed churn rates are 6.6% and 9.6%. Pair regional lists with the network hand-off.
4. Reasons show what the model uses, not what causes churn.

## Pilot needed to confirm
A 50/50 randomised holdout of top-decile prepaid subscribers, 30-day churn as the primary KPI. At the assumed 20% relative reduction it needs about 3,282 subscribers per arm (`docs/pilot_design.md`).

---

<div dir="rtl">

# مذكرة للمدير: ميزانية الاحتفاظ بالعملاء لهذا الشهر

**جميع الأرقام مأخوذة من قاعدة مشتركين محاكاة. نسبة النجاح وتكلفة التواصل والأفق الزمني افتراضات معلنة وليست نتائج.**

## التوصية
التواصل مع نحو **1,470 مشتركاً (8.9% من القاعدة النشطة البالغة 16,503)** في تاريخ 2024-12-31، بترتيبهم حسب القيمة المتوقعة وليس حسب احتمال المغادرة فقط. نتوقف عند هذا الحد، لأن كل تواصل إضافي تتجاوز تكلفته الوفر المتوقع منه.

## ما الذي نحصل عليه
- الشريحة العُشرية الأعلى خطراً متوسط احتمال مغادرتها في الشهر القادم 7.7%، مقابل 1.5% لكامل القاعدة. وعلى أشهر الاختبار (سبتمبر إلى نوفمبر) يصل التواصل مع أعلى 10% إلى نحو **49% من المشتركين الذين غادروا فعلاً** (نسبة رفع نحو 5.0 مرات).
- الاختبار الرجعي عند نهاية نوفمبر: التواصل مع 1,363 مشتركاً اختارهم ترتيب القيمة حقق **38,382 جنيهاً سودانياً** صافي قيمة على النتائج المحاكاة، بينما توقّع النموذج 46,147 جنيهاً. أي أن القيمة المتوقعة تبالغ بنحو 17% في شهر هادئ، وقد توقّع النموذج 54 مغادراً في القائمة وغادر فعلاً 52. يُعتمد الرقم المتحقق في التخطيط إلى أن تؤكده التجربة التجريبية.
- إنفاق الميزانية نفسها على أعلى الاحتمالات فقط حقق **16,458 جنيهاً** فقط. القائمتان تشتركان في 34% من الأسماء، والفرق أن بعض المشتركين شديدي الخطر منخفضو القيمة.

## الافتراضات
نسبة النجاح 20% ممن كانوا سيغادرون، وتكلفة التواصل 20 جنيهاً للمشترك، والقيمة = متوسط الإنفاق لستة أشهر. يمكن تغييرها من أشرطة التحكم في لوحة المتابعة.

## المخاطر
1. النموذج مُدرَّب ومُختبَر على بيانات محاكاة، والأداء الحقيقي غير معروف.
2. أي تغيّر في الظروف (انقطاع في الشبكة أو تغيير في الأسعار) يغيّر معدل المغادرة وبالتالي المعايرة. يرصد فحص الانحراف الشهري ذلك، ويُعاد ضبط المعايرة عند كل إعادة تدريب.
3. انقطاعات الشبكة الإقليمية تشوّه الدرجات: تُصنَّف كسلا بنسبة 52% ونيالا بنسبة 34% من مشتركيهما، بينما معدل المغادرة الفعلي 6.6% و9.6%. لذا تُقرن القوائم الإقليمية بتقرير الشبكة.
4. الأسباب المعروضة تبيّن ما يعتمد عليه النموذج، وليس ما يسبب المغادرة.

## التجربة المطلوبة للتأكد
تجربة عشوائية 50/50 مع مجموعة ضابطة للمشتركين مسبقي الدفع في الشريحة العُشرية الأعلى، ومعيارها الأساسي معدل المغادرة خلال 30 يوماً. عند انخفاض نسبي متوقع قدره 20% نحتاج نحو 3,282 مشتركاً في كل مجموعة.

</div>

*Arabic text: native review needed before it goes to a director.*
