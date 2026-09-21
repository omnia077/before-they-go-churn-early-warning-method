# Feature dictionary

About 22 as-of features per subscriber per month-end cutoff `C`. Windows are (C - from, C - to].
Excluded on purpose: persona, is_churned, root_cause, classification_reason, churn_date,
days_silent and tickets_last_30d_before_silence (anchored to the label), raw msisdn,
gender, age, and region one-hot (ablation only).

| Feature | Group | Definition | Single-feature test AUC |
|---|---|---|---|
| `use_30d` | Usage | Usage sessions in the last 30 days | 0.764 |
| `use_prev60d` | Usage | Usage sessions in the 60 days before that (days 31-90) | 0.679 |
| `use_trend` | Usage | use_30d / (use_prev60d / 2 + 1); below 1 = declining | 0.697 |
| `data_mb_30d` | Usage | Data volume (MB) in the last 30 days | 0.748 |
| `voice_min_30d` | Usage | Voice minutes in the last 30 days | 0.732 |
| `days_since_use` | Usage | Days since last usage event (120 if none) | 0.713 |
| `days_since_rech` | Recharge | Days since last recharge (200 if none; -1 for postpaid) | 0.637 |
| `rech_30d` | Recharge | Recharges in the last 30 days | 0.629 |
| `rech_90d` | Recharge | Recharges in the last 90 days | 0.609 |
| `rech_amt_90d` | Recharge | Mean recharge amount (SDG), last 90 days | 0.552 |
| `rech_gap_ratio` | Recharge | Days since last recharge / (mean gap over 180d + 1); >1 = overdue | 0.619 |
| `tix_30d` | Network | Network tickets in the last 30 days | 0.537 |
| `tix_90d` | Network | Network tickets in the last 90 days | 0.571 |
| `hard_tix_30d` | Network | Hard-issue tickets (dropped call, no signal, slow data), 30 days | 0.535 |
| `region_tix_rate_30d` | Network | Tickets per 1,000 active subscribers in the region, 30 days | 0.678 |
| `unpaid_last3` | Billing | Unpaid bills among the last 3 bills | 0.610 |
| `late_last3` | Billing | Late-paid bills among the last 3 bills | 0.581 |
| `bill_ratio` | Billing | Latest bill / average of last 3 bills | 0.666 |
| `tenure_days` | Lifecycle | Days since activation | 0.536 |
| `tenure_lt90` | Lifecycle | 1 if tenure under 90 days | 0.500 |
| `is_prepaid` | Lifecycle | 1 if prepaid | 0.522 |
| `monthly_arpu_sdg` | Lifecycle | Static monthly ARPU (SDG) from the subscriber master | 0.503 |
