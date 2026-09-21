# Architecture Decision Record — Telecom Churn Platform

This document records the key design decisions made in this project,
the alternatives that were considered, and the reasoning behind each choice:


## Why Apache Iceberg?
Iceberg adds three things plain Parquet cannot:

1. **Snapshot isolation** — every write is atomic. A reader always sees one
   complete, consistent snapshot, never a partial write.
2. **Time travel** — every write creates a new snapshot. The table as it
   looked last week is still queryable even after ten overwrites since.
3. **Idempotent writes** — `iceberg_writer.py` uses `overwrite()` scoped to
   `reference_date` instead of unconditional `append()`. An Airflow retry
   replaces that date's rows rather than duplicating them.
---

## Why two fact tables ?



I found that the grains are genuinely different. A customer can raise zero, one, or five
tickets on a day that has exactly one churn-snapshot row. Merging them into
one table would produce a **fan-out**: every measure in the churn-snapshot
row (days_silent, arpu_drop_pct, etc.) would be repeated once per ticket,
and every `SUM()` on that table would silently inflate those measures by the
ticket count. This is one of the most common warehouse modeling mistakes —
and it produces wrong numbers with no error message.

Two fact tables sharing conformed dimensions (dim_customer, dim_region,
dim_date) is the correct fix. A dashboard can still answer "how many tickets
did churned customers in Khartoum raise?" by joining the two facts through
`dim_customer` — it just can't do it in a single scan of one table.

---

## Why SCD Type 2 on dim_customer?

Telecom customers change plans and move regions regularly. If we overwrote
`dim_customer` on every load, a churn event from Feb with a `postpaid`
plan would silently get attributed to the customer's April `prepaid` plan
in every report from April onward — wrong historical data with no warning.

---

## Why DuckDB for the analytics mart, not another Spark job?



**Why not Spark?**

The mart layer is read-heavy SQL aggregations over ~5,000 rows — not a
distributed compute problem. Spinning up a Spark context to run six
GROUP BY queries over a small warehouse adds minutes of JVM startup
overhead for milliseconds of actual work.

---

## Why is root cause classification a rule engine, not an ML model?

These reasons:

1. **Explainability.** A retention team calling a churned customer
   needs to explain why that customer was flagged. "Your rule says: 3+
   tickets before going silent = network churn" is actionable. "The model
   assigned 0.73 probability to class 'network'" is not.
2. **No labelled training data.** This project generates synthetic data.
   Training an ML classifier on synthetic labels to predict those same
   synthetic labels is circular — the classifier would learn the generator's
   logic, not real churn patterns.

---

This reasoning holds for root-cause classification after churn. Predicting *next-month* churn is a different question: see the next-but-one section, "Why add a predictive model".

## Why add a predictive model, and how is it kept honest?

The rule engine above explains churn *after* it happened. `churn_model/` adds a next-month churn model (logistic + LightGBM,
calibrated) that scores every active subscriber and turns the scores into a contact decision
(expected value = probability x value x save rate - cost). Applied as of a cutoff, the rule thresholds reach 0.63 ROC-AUC
as a predictor; the model reaches 0.81. The two are complementary: rules explain, the model ranks.

It is not the circular exercise the rule-engine section warns about:

1. The v2 generator (`phase1b_generate_data.py`) draws churn from a hidden, evolving hazard, never from a rule on the
   features, so the model cannot relearn the generator. An oracle ceiling (true hazard) and an answer key show how far the
   model is from the best possible score and whether it uses the planted drivers.
2. Evaluation is out of time (train Mar-Jul, validate Aug, test Sep-Nov) with subscriber-grouped folds and bootstrap.
3. Leakage is blocked structurally: as-of features only, banned columns asserted in tests, IDs hashed with a salt from the
   environment.

Trade-offs accepted: SHAP shows what the model uses, not what causes churn; calibration has to be refitted when the base
rate moves (see `docs/model_card.md`); results on simulated data prove the method, not real-world performance. A randomised
pilot (`docs/pilot_design.md`) is the step that would.

## Why manifests instead of a database for pipeline state?

**Why JSON files?**

- **Zero infrastructure.** A state database needs a running service.
  JSON files need nothing — they work the same on a laptop and in
  production.
- **Debuggable by humans.** A pipeline operator can `cat` a manifest to
  understand what ran, when, and what it produced — without querying a
  database schema they may not know.
