# Pilot design: does contacting top-risk subscribers reduce churn?

The model predicts risk; it does not prove that contacting people changes the outcome. A randomised pilot does.

## Population and assignment
Active prepaid subscribers in the top-risk decile at each monthly cutoff, randomised 50/50 into treatment and holdout, stratified by region. Holdout receives nothing beyond normal service.

## Treatment
The action mapped to each subscriber's top reason: recharge bonus (recharge gap, inactivity), usage or data bundle offer (usage decline), payment plan (unpaid bills), welcome call (new subscriber), network follow-up call (network complaints).

## Outcomes
- **Primary KPI:** 30-day churn rate, treatment minus holdout.
- **Secondary:** value retained per contact, cost per saved subscriber.
- **Guardrails:** complaints and opt-outs.

## Analysis
Intention-to-treat, two-proportion test at alpha 0.05, stratified by region. The pilot outcome also checks the model's calibration: expected churn in the holdout should match observed.

## Sample size
Baseline churn in the top decile is **9.80%**, measured on the test months (Sep-Nov) of this simulation; replace with the operator's value. Two-sided alpha 0.05, power 80%.

| Relative reduction from treatment | Subscribers per arm | Total |
|---|---|---|
| 15% | 5,983 | 11,966 |
| 20% | 3,282 | 6,564 |
| 25% | 2,046 | 4,092 |
| 30% | 1,383 | 2,766 |

The simulated base yields about 1,650 top-decile subscribers per month, so even the 30% case needs roughly two months of enrolment and the 15% case about seven. A real pilot needs a real operator base or several months of enrolment. The assumed save rate in the budget model is 20%, so plan around the 20% row (3,282 per arm, about four months of enrolment at this base size).

## Risks
Contamination if holdout subscribers are contacted by other campaigns; regime shifts such as a regional outage mid-pilot (stratify by region and monitor PSI); enrolment months with different churn baselines.
