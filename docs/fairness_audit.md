# Fairness audit (simulated data)

Age is not a feature. Metrics on test cutoffs (Sep-Nov). `Flag rate` = share above the global top-10% threshold; a large gap between groups means the model concentrates contact on one group.

## By age_band

| Group | Rows | Churners | Observed churn | Mean score | Flag rate | Lift@10% |
|---|---|---|---|---|---|---|
| 18-24 | 10,402 | 199 | 1.91% | 1.82% | 10.3% | 4.88 |
| 25-34 | 13,428 | 270 | 2.01% | 1.83% | 10.3% | 4.97 |
| 35-44 | 11,698 | 223 | 1.91% | 1.76% | 9.8% | 4.80 |
| 45-54 | 7,370 | 143 | 1.94% | 1.75% | 9.5% | 5.17 |
| 55-69 | 7,706 | 153 | 1.99% | 1.79% | 9.9% | 5.17 |

## By plan

| Group | Rows | Churners | Observed churn | Mean score | Flag rate | Lift@10% |
|---|---|---|---|---|---|---|
| postpaid | 14,704 | 244 | 1.66% | 1.44% | 6.8% | 4.67 |
| prepaid | 35,900 | 744 | 2.07% | 1.94% | 11.3% | 5.04 |

## By region

| Group | Rows | Churners | Observed churn | Mean score | Flag rate | Lift@10% |
|---|---|---|---|---|---|---|
| Atbara | 3,123 | 44 | 1.41% | 1.47% | 7.5% | 3.87 |
| El Obeid | 3,499 | 83 | 2.37% | 1.88% | 11.0% | 4.47 |
| Kassala | 3,773 | 249 | 6.60% | 7.62% | 51.9% | 3.01 |
| Khartoum | 18,618 | 197 | 1.06% | 1.07% | 4.4% | 4.98 |
| Nyala | 1,633 | 157 | 9.61% | 4.08% | 34.4% | 2.42 |
| Omdurman | 10,876 | 137 | 1.26% | 1.16% | 4.9% | 4.97 |
| Port Sudan | 4,130 | 52 | 1.26% | 1.32% | 6.4% | 3.27 |
| Wad Madani | 4,952 | 69 | 1.39% | 1.27% | 5.9% | 3.33 |

Overall churn 1.95%. Groups with fewer than 20 churners show no lift (too noisy).
