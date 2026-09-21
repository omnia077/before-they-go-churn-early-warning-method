"""
Fairness audit (audit only: age is NOT a model feature).

Lift@10%, mean score and observed churn rate on the test cutoffs by age band, plan and region.
Writes docs/fairness_audit.md for the model card.
Run: python -m churn_model.fairness
"""
import joblib
import numpy as np
import pandas as pd

from churn_model.common import ROOT, load_config, load_panel, out_dir, path, split
from churn_model.evaluate import lift_at


def main():
    cfg = load_config()
    od = out_dir(cfg)
    model = joblib.load(od / "models" / "primary.joblib")
    _, _, te = split(load_panel(cfg), cfg)
    cust = pd.read_parquet(path(cfg, "raw").parent / "sftp_drop" / "customers.parquet")[["customer_id", "age"]]
    te = te.merge(cust, on="customer_id", how="left").reset_index(drop=True)
    te["age_band"] = pd.cut(te["age"], [17, 24, 34, 44, 54, 69], labels=["18-24", "25-34", "35-44", "45-54", "55-69"])
    te["plan"] = np.where(te["is_prepaid"] == 1, "prepaid", "postpaid")
    te["p"] = model.predict_proba(te)
    raw = model.predict_raw(te)
    base = te["y"].mean()
    thr = np.quantile(te["p"], 0.90)

    lines = ["# Fairness audit (simulated data)\n",
             "Age is not a feature. Metrics on test cutoffs (Sep-Nov). `Flag rate` = share above the "
             "global top-10% threshold; a large gap between groups means the model concentrates contact on one group.\n"]
    for col in ["age_band", "plan", "region"]:
        rows = []
        for k, g in te.groupby(col, observed=True):
            gi = g.index.values
            rows.append((k, len(g), int(g["y"].sum()), g["y"].mean(), g["p"].mean(),
                         float((g["p"] >= thr).mean()),
                         lift_at(g["y"].values, raw[gi]) if g["y"].sum() >= 20 else np.nan))
        lines += [f"## By {col}\n", "| Group | Rows | Churners | Observed churn | Mean score | Flag rate | Lift@10% |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {r[0]} | {r[1]:,} | {r[2]} | {r[3]:.2%} | {r[4]:.2%} | {r[5]:.1%} | "
                  f"{'-' if np.isnan(r[6]) else f'{r[6]:.2f}'} |" for r in rows]
        lines.append("")
    lines.append(f"Overall churn {base:.2%}. Groups with fewer than 20 churners show no lift (too noisy).")
    (ROOT / "docs" / "fairness_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
