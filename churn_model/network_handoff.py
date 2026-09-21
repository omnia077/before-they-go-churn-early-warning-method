"""
Network hand-off for the NOC: how much of each region's churn risk is attributable to network signals?

For every scored subscriber: network share = positive SHAP of the network group divided by the
sum of positive group SHAP values. At-risk = top 10% by probability within the cutoff.
Aggregated by region x cutoff: at-risk subscribers, mean network share, network-attributable
expected value. Validated against the outages the generator planted:
    Nyala   months Jun-Aug, Kassala months Sep-Oct. A month-end cutoff's features already
    contain that month's tickets, so the affected cutoffs are Jun-Aug and Sep-Oct.
Limit: only ticket signals exist. Real cell KPIs (dropped-call rate, throughput) are the next step.
Note: train-cutoff rows are scored in-sample; the share is descriptive, not a performance claim.

Run: python -m churn_model.network_handoff
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn_model.common import ROOT, load_config, load_panel, out_dir
from churn_model.decision import add_ev, attach_value
from churn_model.explain import group_shap, shap_values

# outage episode -> (region, cutoff indices that precede the outage months)
PLANTED = {"Nyala": {"outage_months": "Jun-Aug", "cutoffs": [5, 6, 7]},
           "Kassala": {"outage_months": "Sep-Oct", "cutoffs": [8, 9]}}
MONTH = {2: "Mar", 3: "Apr", 4: "May", 5: "Jun", 6: "Jul", 7: "Aug", 8: "Sep", 9: "Oct", 10: "Nov"}


def compute(cfg, df=None, model=None):
    od = out_dir(cfg)
    model = model or joblib.load(od / "models" / "primary.joblib")
    df = load_panel(cfg) if df is None else df
    df = add_ev(attach_value(df, cfg), model.predict_proba(df), cfg).reset_index(drop=True)
    gs = group_shap(shap_values(model, df), model.features)
    pos = gs.clip(lower=0)
    df["network_share"] = pos["network_complaints"] / pos.sum(axis=1).replace(0, np.nan)
    df["network_share"] = df["network_share"].fillna(0)
    df["at_risk"] = df.groupby("cutoff")["p"].rank(ascending=False, pct=True) <= 0.10
    df["network_ev"] = df["ev"].clip(lower=0) * df["network_share"]
    return df


def aggregate(df):
    ar = df[df["at_risk"]]
    g = ar.groupby(["cutoff", "region"]).agg(
        at_risk=("p", "size"), network_share=("network_share", "mean"),
        network_attributable_ev_sdg=("network_ev", "sum"),
        actual_churners=("y", "sum")).reset_index()
    g["network_ev_rank"] = g.groupby("cutoff")["network_attributable_ev_sdg"].rank(ascending=False).astype(int)
    g["network_share_rank"] = g.groupby("cutoff")["network_share"].rank(ascending=False).astype(int)
    return g


def main():
    cfg = load_config()
    od = out_dir(cfg)
    df = compute(cfg)
    g = aggregate(df)
    g.to_parquet(od / "network_handoff.parquet", index=False)
    g.to_csv(od / "network_handoff.csv", index=False)

    checks = {}
    for region, meta in PLANTED.items():
        sub = g[(g["region"] == region) & (g["cutoff"].isin(meta["cutoffs"]))]
        rest = g[(g["region"] == region) & (~g["cutoff"].isin(meta["cutoffs"]))]
        checks[region] = {
            "outage_months": meta["outage_months"],
            "mean_network_share_during_outage": float(sub["network_share"].mean()),
            "mean_network_share_other_months": float(rest["network_share"].mean()),
            "rank_by_network_share_during_outage": sub["network_share_rank"].tolist(),
            "rank_by_network_ev_during_outage": sub["network_ev_rank"].tolist(),
            "cutoffs": meta["cutoffs"], "n_regions": int(g["region"].nunique())}
        checks[region]["flagged_as_top_region_by_network_ev"] = bool((sub["network_ev_rank"] == 1).all())
    (od / "network_handoff_validation.json").write_text(json.dumps(checks, indent=2))

    # heatmap: network-attributable EV by region x cutoff
    piv = g.pivot(index="region", columns="cutoff", values="network_attributable_ev_sdg").fillna(0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(piv.values / 1000, aspect="auto", cmap="Reds")
    ax.set_xticks(range(piv.shape[1])), ax.set_xticklabels([MONTH[c] for c in piv.columns])
    ax.set_yticks(range(piv.shape[0])), ax.set_yticklabels(piv.index)
    ax.set_xlabel("Cutoff month"), ax.set_title("Network-attributable expected value, top-10% risk (thousand SDG)", fontsize=9)
    fig.colorbar(im, ax=ax), fig.tight_layout()
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "docs" / "figures" / "network_handoff.png", dpi=150), plt.close(fig)

    latest = g[g["cutoff"] == g["cutoff"].max()].sort_values("network_attributable_ev_sdg", ascending=False)
    print(json.dumps(checks, indent=2))
    print("\nlatest cutoff ranked list for the NOC:\n", latest.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
