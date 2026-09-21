"""
Top-3 bilingual reason codes per subscriber from grouped SHAP values.

A reason is chosen only if its group pushes risk UP (positive SHAP). plan_profile has no
reason template (not actionable). If a subscriber's raw values do not support the main
template (for example usage did not actually fall) a neutral fallback is used instead.
"""
import numpy as np
import pandas as pd
import yaml

from churn_model.common import ROOT

REASON_GROUPS = ["usage_decline", "recharge_gap", "network_complaints",
                 "unpaid_bills", "new_subscriber", "inactivity"]


def load_templates():
    with open(ROOT / "churn_model" / "reason_templates.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ar_days(n):
    """Arabic day count with correct number agreement (1, 2, 3-10, 11+)."""
    n = int(n)
    if n == 1:
        return "يوم واحد"
    if n == 2:
        return "يومين"
    return f"{n} أيام" if 3 <= n <= 10 else f"{n} يوماً"


def _fill(group, row, tpl):
    t = tpl[group]
    if group == "usage_decline":
        prev = row["use_prev60d"] / 2.0
        x = round(100 * (1 - row["use_30d"] / prev)) if prev > 0 else 0
        return (t["en"].format(x=x), t["ar"].format(x=x)) if x > 0 else (t["fallback_en"], t["fallback_ar"])
    if group == "recharge_gap":
        n = int(row["days_since_rech"])
        return (t["en"].format(n=n), t["ar"].format(days_ar=ar_days(n))) if n > 0 else (t["fallback_en"], t["fallback_ar"])
    if group == "network_complaints":
        n = int(row["tix_30d"])
        if n > 0:
            return t["en"].format(n=n).replace("(s)", "" if n == 1 else "s"), t["ar"].format(n=n)
        return t["fallback_en"], t["fallback_ar"]
    if group == "unpaid_bills":
        n = int(row["unpaid_last3"])
        if n > 0:
            return t["en"].format(n=n).replace("(s)", "" if n == 1 else "s"), t["ar"].format(n=n)
        return t["fallback_en"], t["fallback_ar"]
    if group == "new_subscriber":
        return (t["en"], t["ar"]) if row["tenure_days"] < 90 else (t["fallback_en"], t["fallback_ar"])
    if group == "inactivity":
        n = int(row["days_since_use"])
        return (t["en"].format(n=n), t["ar"].format(days_ar=ar_days(n))) if n > 0 else (t["fallback_en"], t["fallback_ar"])
    raise KeyError(group)


def build_reasons(df, group_shap, templates=None, top_k=3):
    """df: feature rows; group_shap: DataFrame with same row order (columns = reason groups).
    Returns DataFrame with reason_{i}_en / reason_{i}_ar / reason_{i}_group."""
    tpl = templates or load_templates()
    g = group_shap[REASON_GROUPS].to_numpy()
    order = np.argsort(-g, axis=1)[:, :top_k]
    rows = []
    for i in range(len(df)):
        r = df.iloc[i]
        rec = {}
        k = 0
        for j in order[i]:
            if g[i, j] <= 0:
                continue
            group = REASON_GROUPS[j]
            en, ar = _fill(group, r, tpl)
            k += 1
            rec[f"reason_{k}_group"], rec[f"reason_{k}_en"], rec[f"reason_{k}_ar"] = group, en, ar
        rows.append(rec)
    out = pd.DataFrame(rows, index=df.index)
    for k in range(1, top_k + 1):
        for c in (f"reason_{k}_group", f"reason_{k}_en", f"reason_{k}_ar"):
            if c not in out:
                out[c] = None
    return out[[f"reason_{k}_{s}" for k in range(1, top_k + 1) for s in ("group", "en", "ar")]]
