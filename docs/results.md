# Out-of-time results (simulated data)

Test cutoffs Sep-Nov, 50,604 rows, 988 churners, base rate 1.95%. 95% CIs from 200 customer-level bootstrap resamples.

| Model | PR-AUC (95% CI) | ROC-AUC | Top-10% lift (95% CI) | Precision @ top10% | Recall @ top10% | Brier |
|---|---|---|---|---|---|---|
| no-skill | 0.019 | 0.500 | 1.00 | - | - | - |
| rule heuristic | 0.054 (0.046-0.065) | 0.629 | 3.38 (3.02-3.68) | - | - | - |
| logistic | 0.134 (0.115-0.154) | 0.812 | 4.93 (4.64-5.24) | 0.101 | 0.481 | 0.0180 |
| lightgbm | 0.129 (0.112-0.149) | 0.813 | 4.97 (4.67-5.29) | 0.098 | 0.492 | 0.0181 |
| lightgbm + region (ablation) | 0.130 (0.112-0.150) | 0.810 | 4.92 (4.63-5.26) | - | - | - |
| ORACLE true hazard | 0.276 (0.245-0.306) | 0.880 | 6.55 (6.30-6.86) | - | - | - |
