# Model card: churn early-warning model

**Status:** portfolio prototype on simulated data. Not validated on real subscribers. Model version `churn-model-v1.0.0`.

## Purpose and permitted use
Score each active subscriber's probability of churning in the next calendar month, so a retention team can decide whom to contact and what it buys. **Permitted purpose: retention only.** Not for credit decisions, pricing discrimination or any use outside retention.

## Data
Simulated 20,000-subscriber base from `phase1b_generate_data.py`. Churn is a random draw from a hidden, evolving dissatisfaction state, never a rule on the features. Monthly churn is 1.8%. The generator plants a Nyala outage (Jun-Aug) and a Kassala outage (Sep-Oct). Ground truth (`churn_labels`, `latent_state`, `answer_key.json`) is used only to build the label and to validate. Exploratory analysis is in `notebooks/01_eda_churn_panel.ipynb`.

## Label and time design
Cutoff = end of month; features use data up to the cutoff only. Label = churn date in the next calendar month; subscribers already churned at the cutoff are excluded. Train cutoffs Mar-Jul, validation Aug, test Sep-Nov (Kassala outage inside the test window). Folds and bootstrap are grouped by subscriber.

## Model
22 as-of features (see `feature_dictionary.md`). Logistic regression and LightGBM (random search on the validation cutoff, early stopping on logloss). Probabilities are calibrated with Platt scaling fitted on **pooled out-of-fold predictions over all train and validation cutoffs (Mar-Aug)**. The registered primary model is **LightGBM**, chosen on validation PR-AUC (0.305 vs 0.296 for logistic). On the test months logistic is nominally ahead (0.134 vs 0.129 PR-AUC), but the confidence intervals overlap, so treat them as tied.

## Performance (test cutoffs Sep-Nov, 988 churners, base rate 1.95%)
See `results.md`. Headline: ROC-AUC 0.81, top-decile lift about 5.0x, PR-AUC about 0.13 (95% CI 0.11-0.15) against 0.02 base rate. The oracle (true hazard) reaches 0.88 ROC-AUC and 0.28 PR-AUC, so the model recovers roughly half of the oracle's PR-AUC. The rule heuristic (root-cause thresholds applied as of the cutoff) reaches only 0.63 ROC-AUC: the rule engine explains churn after it happens; this model predicts it before. They are complementary.

Gradient boosting roughly ties logistic because the planted signal is close to linear. On real data with interactions a gap is possible and should be tested, not assumed.

## Calibration
Test Brier 0.0181 against 0.0191 for a constant base-rate forecast; mean predicted churn on the test months is 1.8% against 1.95% observed. See `figures/reliability.png`.

An earlier version fitted an isotonic calibrator on the August validation month alone. August was busy (2.8% churn) and the test months were quieter (1.95%), so probabilities ran about 18% too high (mean 2.30%) and the test Brier score was worse than the uncalibrated model. Fitting on pooled out-of-fold predictions over all matured months removed most of that bias (compared offline before adopting it). At the November decision point the model now expects 54 churners in the contacted list and 52 occurred (it was 85 vs 65 before). Residual risk: a regime shift (outage, price change) still moves the base rate; the monthly PSI check and the retrain rule below are the safeguards, and calibration should be refitted whenever the model is retrained.

## Explanations
SHAP (log-odds), grouped into six reason groups; top-3 English and Arabic reasons per subscriber. **SHAP shows what the model uses, not what causes churn.** Answer-key check: usage decline is the largest group (43% of mean |SHAP|), then unpaid bills (17%), recharge gaps (16%), new subscriber (12%) and network complaints (8%); the sign of each driver is as planted (shorter tenure, more unpaid bills, lower usage trend raise risk). Spearman correlation of predicted probability with the true hazard on test rows: 0.70.

## Fairness audit (audit only; age is not a feature)
See `fairness_audit.md`. By age band, flag rates are 9.5-10.3% and lift 4.8-5.2x: no age band is treated very differently. **By region the model is not even:** Kassala is flagged at 51.9% (mean score 7.6% vs 6.6% observed) and Nyala at 34.4% (mean score 4.1% vs 9.6% observed), while Khartoum and Omdurman are flagged at about 5%. Kassala is over-predicted and Nyala under-predicted. The outages drive this, and the region one-hot ablation did not change test PR-AUC (0.130 vs 0.129), so simply adding region does not fix it. Use the regional view with the network hand-off, not as a per-region target.

## Monitoring
PSI on the score and top-5 features each month (`drift.py`). `tenure_days` is excluded (it ages by construction). At the 2024-12-31 cutoff the score PSI is 0.010 (stable) while `region_tix_rate_30d` alerts, because the regional outages ended: the monitor is doing its job. Retrain trigger: score PSI above 0.25, or above 0.1 in two consecutive months, or PR-AUC dropping materially once labels mature.

## Security and governance
Hashed subscriber IDs only (salted SHA-256, salt in `CHURN_HASH_SALT`, never in the repo). No msisdn, names, gender or age in the panel or the scores. `churn_scores` should sit behind role-based access, with a stated retention period.

## Limitations
All results are on simulated data and demonstrate the method, not real-world performance. Save rate, contact cost and horizon are assumptions. The label is a proxy for inactivity and some inactive subscribers return. Only ticket-based network signals exist. Sudan-specific shocks (displacement, multi-SIM behaviour) are only partly simulated. The production "active" filter (used within 60 days) differs slightly from the training exclusion, which uses the true churn date. The regional unevenness above is documented, not fixed.
