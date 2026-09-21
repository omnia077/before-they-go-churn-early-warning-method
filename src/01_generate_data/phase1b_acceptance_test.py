"""
Phase 1b acceptance test: is the simulated data learnable, but not trivially so?

Builds ~15 AS-OF features from the raw tables only (never from ground_truth),
uses out-of-time cutoffs, and reports:
  1. churn rate per cutoff month
  2. best SINGLE-feature ROC-AUC        (must stay well below 0.85)
  3. logistic regression / gradient boosting on out-of-time months
  4. ORACLE AUC using the true hazard   (the ceiling no model can beat)

Ground truth is used only to (a) build the label and (b) drop subscribers who
had already churned by the cutoff. Run:  python phase1b_acceptance_test.py --root ../data_v2
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_CUTOFFS = [2, 3, 4, 5, 6]     # month-end index (0 = Jan); needs 90d history
TEST_CUTOFFS = [8, 9, 10]           # validation month 7 is left free for tuning
SEED = 42


def month_end(m):
    return pd.Timestamp("2024-01-01") + pd.DateOffset(months=m + 1) - pd.Timedelta(days=1)


def lift_at(y, s, k=0.10):
    top = np.argsort(-s)[: int(len(y) * k)]
    return y[top].mean() / y.mean()


def build(root, cutoff_idx, T):
    """One row per customer active at the cutoff, features use data <= cutoff."""
    C = month_end(cutoff_idx)
    cust, usage, rech, tix, bill, lab, lat = (T[k] for k in
        ["cust", "usage", "rech", "tix", "bill", "lab", "lat"])

    base = cust[cust["activation_date"] <= C].merge(lab, on="customer_id", how="left")
    # exclude subscribers that had already churned before the cutoff
    base = base[~(base["churn_date"].notna() & (base["churn_date"] <= C))].copy()
    nxt_end = month_end(cutoff_idx + 1)
    base["y"] = (base["churn_date"].notna() & (base["churn_date"] > C)
                 & (base["churn_date"] <= nxt_end)).astype(int)

    def win(df, col, lo, hi):
        return df[(df[col] > C - pd.Timedelta(days=lo)) & (df[col] <= C - pd.Timedelta(days=hi))]

    u = usage[usage["event_date"] <= C]
    n30 = win(u, "event_date", 30, 0).groupby("customer_id").size().rename("use_30d")
    n60 = win(u, "event_date", 90, 30).groupby("customer_id").size().rename("use_prev60d")
    mb30 = win(u[u["usage_type"] == "data"], "event_date", 30, 0
               ).groupby("customer_id")["usage_value"].sum().rename("data_mb_30d")
    last_use = u.groupby("customer_id")["event_date"].max().rename("last_use")

    r = rech[rech["recharge_date"] <= C]
    r90 = win(r, "recharge_date", 90, 0)
    rc30 = win(r, "recharge_date", 30, 0).groupby("customer_id").size().rename("rech_30d")
    rc90 = r90.groupby("customer_id").size().rename("rech_90d")
    ramt = r90.groupby("customer_id")["amount_sdg"].mean().rename("rech_amt_90d")
    last_r = r.groupby("customer_id")["recharge_date"].max().rename("last_rech")

    t = tix[tix["ticket_date"] <= C]
    t30 = win(t, "ticket_date", 30, 0).groupby("customer_id").size().rename("tix_30d")
    t90 = win(t, "ticket_date", 90, 0).groupby("customer_id").size().rename("tix_90d")

    b = bill[bill["billing_month"] <= C.strftime("%Y-%m")].sort_values("billing_month")
    b3 = b.groupby("customer_id").tail(3)
    unpaid = (b3["payment_status"] == "unpaid").groupby(b3["customer_id"]).sum().rename("unpaid_last3")

    X = base.set_index("customer_id")
    for s_ in [n30, n60, mb30, last_use, rc30, rc90, ramt, last_r, t30, t90, unpaid]:
        X = X.join(s_, how="left")          # explicit left joins keep the base rows only
    for c in ["use_30d", "use_prev60d", "data_mb_30d", "rech_30d", "rech_90d",
              "tix_30d", "tix_90d", "unpaid_last3"]:
        X[c] = X[c].fillna(0)
    X["use_trend"] = X["use_30d"] / (X["use_prev60d"] / 2 + 1)
    X["days_since_use"] = (C - X["last_use"]).dt.days.fillna(120)
    prepaid = X["plan_type"] == "prepaid"
    X["days_since_rech"] = np.where(prepaid, (C - X["last_rech"]).dt.days.fillna(200), -1)
    X["rech_amt_90d"] = X["rech_amt_90d"].fillna(0)
    X["tenure_days"] = (C - X["activation_date"]).dt.days
    X["is_prepaid"] = prepaid.astype(int)
    X["cutoff"] = cutoff_idx
    # true hazard for the NEXT month (oracle ceiling, uses ground truth)
    hz = lat[lat["month"] == cutoff_idx + 2].set_index("customer_id")["true_hazard"]
    X["oracle"] = hz.reindex(X.index).values
    X = X.join(pd.get_dummies(X["region"], prefix="reg", dtype=int))
    return X.reset_index()


def main(root):
    rd = lambda n: pd.read_parquet(f"{root}/sftp_drop/{n}.parquet")
    T = {"cust": rd("customers"), "usage": rd("usage_events"), "rech": rd("recharge_records"),
         "tix": rd("network_tickets"), "bill": rd("billing_records")}
    T["cust"]["activation_date"] = pd.to_datetime(T["cust"]["activation_date"])
    T["usage"]["event_date"] = pd.to_datetime(T["usage"]["event_date"])
    T["rech"]["recharge_date"] = pd.to_datetime(T["rech"]["recharge_date"])
    T["tix"]["ticket_date"] = pd.to_datetime(T["tix"]["ticket_date"])
    lab = pd.read_parquet(f"{root}/ground_truth/churn_labels.parquet")
    lab["churn_date"] = pd.to_datetime(lab["churn_date"])
    T["lab"] = lab[["customer_id", "churn_date"]]
    T["lat"] = pd.read_parquet(f"{root}/ground_truth/latent_state.parquet")

    frames = {c: build(root, c, T) for c in TRAIN_CUTOFFS + TEST_CUTOFFS}
    tr = pd.concat([frames[c] for c in TRAIN_CUTOFFS])
    te = pd.concat([frames[c] for c in TEST_CUTOFFS])

    print("\nCHURN RATE (next-month) per cutoff")
    for c in TRAIN_CUTOFFS + TEST_CUTOFFS:
        f = frames[c]
        print(f"  cutoff month {c + 1:2d}: n={len(f):6,}  positives={f['y'].sum():4d}  rate={f['y'].mean():.2%}")

    feats = ["use_30d", "use_prev60d", "use_trend", "data_mb_30d", "days_since_use",
             "rech_30d", "rech_90d", "rech_amt_90d", "days_since_rech", "tix_30d",
             "tix_90d", "unpaid_last3", "tenure_days", "is_prepaid", "monthly_arpu_sdg"
             ] + [c for c in tr.columns if c.startswith("reg_")]

    print("\nSINGLE-FEATURE ROC-AUC on test months (direction-free)")
    single = {f: roc_auc_score(te["y"], te[f].fillna(0)) for f in feats}
    single = {f: max(v, 1 - v) for f, v in single.items()}
    for f, v in sorted(single.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {f:<18} {v:.3f}")
    print(f"  best single feature = {max(single.values()):.3f}  (limit 0.85)")

    print("\nMODELS, out-of-time (train cutoffs 3-7, test cutoffs 9-11)")
    y_tr, y_te = tr["y"].values, te["y"].values
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(tr[feats].fillna(0), y_tr)
    gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                        random_state=SEED)
    gb.fit(tr[feats].fillna(0), y_tr)
    rows = [("no-skill (base rate)", np.full(len(y_te), y_te.mean())),
            ("logistic", lr.predict_proba(te[feats].fillna(0))[:, 1]),
            ("gradient boosting", gb.predict_proba(te[feats].fillna(0))[:, 1]),
            ("ORACLE true hazard", te["oracle"].values)]
    print(f"  {'model':<22}{'ROC-AUC':>8}{'PR-AUC':>8}{'top-10% lift':>14}")
    for name, s in rows:
        roc = 0.5 if name.startswith("no-skill") else roc_auc_score(y_te, s)
        print(f"  {name:<22}{roc:8.3f}{average_precision_score(y_te, s):8.3f}"
              f"{(1.0 if name.startswith('no-skill') else lift_at(y_te, s)):14.2f}")
    print(f"\n  test positives = {int(y_te.sum())} of {len(y_te):,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data_v2")
    main(ap.parse_args().root)
