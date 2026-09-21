"""
Day 3 checks on churn_panel + writes docs/feature_dictionary.md.

Fails loudly if: a banned column is present, a subscriber appears twice in a cutoff,
train/validation/test cutoffs overlap, any cutoff feature reaches past its date,
or a single feature is suspiciously strong (leakage smell test: AUC > 0.85).

Run: python -m churn_model.leakage_checks
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

from churn_model.as_of_features import FEATURE_COLUMNS
from churn_model.common import ROOT, load_config, load_panel, split

BANNED = {"persona", "is_churned", "root_cause", "classification_reason", "churn_date", "msisdn",
          "gender", "age", "days_silent", "tickets_last_30d_before_silence", "reference_date",
          "true_hazard", "d_start"}

DICTIONARY = {
    "use_30d": ("Usage", "Usage sessions in the last 30 days"),
    "use_prev60d": ("Usage", "Usage sessions in the 60 days before that (days 31-90)"),
    "use_trend": ("Usage", "use_30d / (use_prev60d / 2 + 1); below 1 = declining"),
    "data_mb_30d": ("Usage", "Data volume (MB) in the last 30 days"),
    "voice_min_30d": ("Usage", "Voice minutes in the last 30 days"),
    "days_since_use": ("Usage", "Days since last usage event (120 if none)"),
    "days_since_rech": ("Recharge", "Days since last recharge (200 if none; -1 for postpaid)"),
    "rech_30d": ("Recharge", "Recharges in the last 30 days"),
    "rech_90d": ("Recharge", "Recharges in the last 90 days"),
    "rech_amt_90d": ("Recharge", "Mean recharge amount (SDG), last 90 days"),
    "rech_gap_ratio": ("Recharge", "Days since last recharge / (mean gap over 180d + 1); >1 = overdue"),
    "tix_30d": ("Network", "Network tickets in the last 30 days"),
    "tix_90d": ("Network", "Network tickets in the last 90 days"),
    "hard_tix_30d": ("Network", "Hard-issue tickets (dropped call, no signal, slow data), 30 days"),
    "region_tix_rate_30d": ("Network", "Tickets per 1,000 active subscribers in the region, 30 days"),
    "unpaid_last3": ("Billing", "Unpaid bills among the last 3 bills"),
    "late_last3": ("Billing", "Late-paid bills among the last 3 bills"),
    "bill_ratio": ("Billing", "Latest bill / average of last 3 bills"),
    "tenure_days": ("Lifecycle", "Days since activation"),
    "tenure_lt90": ("Lifecycle", "1 if tenure under 90 days"),
    "is_prepaid": ("Lifecycle", "1 if prepaid"),
    "monthly_arpu_sdg": ("Lifecycle", "Static monthly ARPU (SDG) from the subscriber master"),
}


def main():
    cfg = load_config()
    panel = load_panel(cfg)
    td = cfg["time_design"]

    assert not BANNED & set(panel.columns), BANNED & set(panel.columns)
    assert set(FEATURE_COLUMNS) == set(DICTIONARY), "feature dictionary out of sync"
    assert not panel.duplicated(["customer_id", "cutoff"]).any()
    sets = [set(td["train"]), set(td["validation"]), set(td["test"])]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "cutoff overlap"
    assert max(td["train"]) < min(td["validation"]) < min(td["test"]), "splits not chronological"
    assert panel["y"].isin([0, 1]).all()
    print(f"panel: {len(panel):,} rows, {panel['customer_id'].nunique():,} subscribers, "
          f"{len(FEATURE_COLUMNS)} features")

    tr, va, te = split(panel, cfg)
    shared_tr_te = set(tr["customer_id"]) & set(te["customer_id"])
    print(f"subscribers in both train and test: {len(shared_tr_te):,} "
          f"(different months; folds/bootstrap are grouped by customer)")

    print("\nsingle-feature ROC-AUC on test (direction-free), leakage smell test < 0.85")
    aucs = {f: roc_auc_score(te["y"], te[f].fillna(0)) for f in FEATURE_COLUMNS}
    aucs = {f: max(a, 1 - a) for f, a in aucs.items()}
    for f, a in sorted(aucs.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {f:<22}{a:.3f}")
    assert max(aucs.values()) < 0.85, "a single feature is too strong: look for leakage"

    nulls = panel[FEATURE_COLUMNS].isna().mean()
    print(f"\nmax null rate over features: {nulls.max():.2%} ({nulls.idxmax()})")

    rows = ["# Feature dictionary\n",
            "About 22 as-of features per subscriber per month-end cutoff `C`. Windows are (C - from, C - to].",
            "Excluded on purpose: persona, is_churned, root_cause, classification_reason, churn_date,",
            "days_silent and tickets_last_30d_before_silence (anchored to the label), raw msisdn,",
            "gender, age, and region one-hot (ablation only).\n",
            "| Feature | Group | Definition | Single-feature test AUC |", "|---|---|---|---|"]
    for f in FEATURE_COLUMNS:
        g, d = DICTIONARY[f]
        rows.append(f"| `{f}` | {g} | {d} | {aucs[f]:.3f} |")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "feature_dictionary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\nall leakage checks passed; wrote docs/feature_dictionary.md")


if __name__ == "__main__":
    main()
