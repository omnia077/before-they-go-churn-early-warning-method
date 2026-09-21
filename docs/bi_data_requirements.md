# BI data requirements (for Power BI / Superset)

Tables served from the analytics mart (`analytics_mart.duckdb`, refreshed monthly by the `churn_scoring_monthly` DAG).

| Table | Grain | Key fields | Refresh | Owner |
|---|---|---|---|---|
| `churn_scores` | subscriber x cutoff | `sub_hash`, `cutoff_date`, `region`, `probability`, `decile` (1 = highest risk), `expected_value_sdg`, `reason_{1,2,3}_{group,en,ar}` | monthly, 3rd of month | Data science |
| `network_handoff` | region x cutoff | `region`, `cutoff`, `at_risk`, `network_share`, `network_attributable_ev_sdg`, ranks | monthly | Data science, NOC consumes |
| `model_monitoring` | cutoff | `score_psi`, feature PSI, `retrain_trigger`, model version | monthly | Data science |
| `budget_scenarios` | cutoff x assumption set | contacts, expected net, realised net (backtest only) | on assumption change | Retention |

## Rules
- Hashed IDs only. No msisdn, names or ages in any BI table. Age bands, if needed, come from a separate governed dimension.
- Access: retention team and NOC on `churn_scores` and `network_handoff`; everyone else aggregated views only. State the purpose (retention) and a retention period on the data source.
- Reasons are stored in both languages; the report switches language with a slicer. Arabic columns need right-to-left formatting.
- `decile` and `probability` are calibrated for the month scored; do not compare probabilities across models.
- Assumptions (save rate, cost per contact, horizon) live in `churn_model/config.yaml` and must be shown next to any expected-value figure.

## Suggested pages
Overview (churn trend, model version, drift status), risk list with reasons and language slicer, budget curve with assumption slicers, model performance, network hand-off by region.
